import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lpips
import mmcv
import trimesh

from copy import deepcopy
from glob import glob
import time
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from mmcv.runner import load_checkpoint
from mmgen.models.builder import MODULES, build_module
from mmgen.models.architectures.common import get_module_device

from ...core import custom_meshgrid, eval_psnr, eval_ssim_skimage, reduce_mean, rgetattr, rsetattr, extract_geometry, \
    module_requires_grad, get_cam_rays

LPIPS_BS = 32


@MODULES.register_module()
class TanhCode(nn.Module):
    def __init__(self, scale=1.0, eps=1e-5):
        super(TanhCode, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, code_, update_stats=False):
        return code_.tanh() if self.scale == 1 else code_.tanh() * self.scale

    def inverse(self, code):
        return code.clamp(min=-1 + self.eps, max=1 - self.eps).atanh() if self.scale == 1 \
            else (code / self.scale).clamp(min=-1 + self.eps, max=1 - self.eps).atanh()

@MODULES.register_module()
class IdentityCode(nn.Module):
    @staticmethod
    def forward(code_, update_stats=False):
        return code_

    @staticmethod
    def inverse(code):
        return code


@MODULES.register_module()
class NormalizedTanhCode(nn.Module):
    def __init__(self, mean=0.0, std=1.0, clip_range=1, eps=1e-5, momentum=0.001):
        super(NormalizedTanhCode, self).__init__()
        self.mean = mean
        self.std = std
        self.clip_range = clip_range
        self.register_buffer('running_mean', torch.tensor([0.0]))
        self.register_buffer('running_var', torch.tensor([std ** 2]))
        self.momentum = momentum
        self.eps = eps

    def forward(self, code_, update_stats=False):
        if update_stats and self.training:
            with torch.no_grad():
                var, mean = torch.var_mean(code_)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * reduce_mean(mean))
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * reduce_mean(var))
        scale = (self.std / (self.running_var.sqrt() + self.eps)).to(code_.device)
        return (code_ * scale + (self.mean - self.running_mean.to(code_.device) * scale)
                ).div(self.clip_range).tanh().mul(self.clip_range)

    def inverse(self, code):
        scale = ((self.running_var.sqrt() + self.eps) / self.std).to(code.device)
        return code.div(self.clip_range).clamp(min=-1 + self.eps, max=1 - self.eps).atanh().mul(
            self.clip_range * scale) + (self.running_mean.to(code.device) - self.mean * scale)


class BaseNeRF(nn.Module):
    def __init__(self,
                 code_size=(3, 8, 64, 64),
                 code_activation=dict(
                     type='TanhCode',
                     scale=1),
                 grid_size=64,
                 decoder=dict(
                     type='TriPlaneDecoder'),
                 decoder_use_ema=False,
                 bg_color=1,
                 pixel_loss=dict(
                     type='MSELoss'),
                 scale_loss_weight=0,
                 norm_reg_weight=0,
                 alpha_reg_weight=0,
                 reg_loss=None,
                 per_loss=None,
                 update_extra_interval=16,
                 use_lpips_metric=True,
                 init_from_mean=False,
                 init_scale=1e-4,
                 mean_ema_momentum=0.001,
                 mean_scale=1.0,
                 train_cfg=dict(),
                 test_cfg=dict(),
                 pretrained=None):
        super().__init__()
        self.code_size = code_size
        self.code_activation = build_module(code_activation)
        self.grid_size = grid_size
        self.decoder = build_module(decoder)
        self.decoder_use_ema = decoder_use_ema
        if self.decoder_use_ema:
            self.decoder_ema = deepcopy(self.decoder)
        self.bg_color = bg_color
        self.pixel_loss = build_module(pixel_loss)
        self.reg_loss = build_module(reg_loss) if reg_loss is not None else None
        self.per_loss = build_module(per_loss) if per_loss is not None else None
        self.scale_loss_weight = scale_loss_weight
        self.norm_reg_weight = norm_reg_weight
        self.alpha_reg_weight = alpha_reg_weight
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.update_extra_interval = update_extra_interval
        self.lpips = [] if use_lpips_metric else None  # use a list to avoid registering the LPIPS model in state_dict
        if init_from_mean:
            self.register_buffer('init_code', torch.zeros(code_size))
        else:
            self.init_code = None
        self.init_scale = init_scale
        self.mean_ema_momentum = mean_ema_momentum
        self.mean_scale = mean_scale
        if pretrained is not None and os.path.isfile(pretrained):
            load_checkpoint(self, pretrained, map_location='cpu')

        self.train_cfg_backup = dict()
        for key, value in self.test_cfg.get('override_cfg', dict()).items():
            self.train_cfg_backup[key] = rgetattr(self, key, None)

    def train(self, mode=True):
        if mode:
            for key, value in self.train_cfg_backup.items():
                rsetattr(self, key, value)
        else:
            for key, value in self.test_cfg.get('override_cfg', dict()).items():
                if self.training:
                    self.train_cfg_backup[key] = rgetattr(self, key)
                rsetattr(self, key, value)
        super().train(mode)
        return self

    def load_scene(self, data, load_density=False):
        device = get_module_device(self)
        code_list = []
        for code_state_single in data['code']:
            code_list.append(
                code_state_single['param']['code'] if 'code' in code_state_single['param']
                else self.code_activation(code_state_single['param']['code_']))
        code = torch.stack(code_list, dim=0).to(device)
        return code

    @staticmethod
    def save_scene(save_dir, code, scene_name):
        os.makedirs(save_dir, exist_ok=True)
        for scene_id, scene_name_single in enumerate(scene_name):
            results = dict(
                scene_name=scene_name_single,
                param=dict(
                    code=code.data[scene_id].cpu(),
                    ))
            torch.save(results, os.path.join(save_dir, scene_name_single) + '.pth')

    @staticmethod
    def save_mesh(save_dir, decoder, code, scene_name, mesh_resolution, mesh_threshold):
        os.makedirs(save_dir, exist_ok=True)
        for code_single, scene_name_single in zip(code, scene_name):
            vertices, triangles = extract_geometry(
                decoder,
                code_single,
                mesh_resolution,
                mesh_threshold)
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            mesh.export(os.path.join(save_dir, scene_name_single) + '.stl')

    def get_init_code_(self, num_scenes, device=None):
        code_ = torch.empty(
            self.code_size if num_scenes is None else (num_scenes, *self.code_size),
            device=device, requires_grad=True, dtype=torch.float32)
        if self.init_code is None:
            code_.data.uniform_(-self.init_scale, self.init_scale)
        else:
            code_.data[:] = self.code_activation.inverse(self.init_code * self.mean_scale)
        return code_
    
    def get_init_mask_(self, num_scenes, num_init, device=None):
        mask = torch.ones(num_init if num_scenes is None else (num_scenes, num_init), device=device, dtype=torch.bool)
        return mask

    def get_init_points_(self, num_scenes, init_pcd, device=None):
        if num_scenes == None:
            points = init_pcd
        else:
            points = init_pcd[None].expand(num_scenes, -1, -1)
        return points

    def get_init_density_grid(self, num_scenes, device=None):
        return torch.zeros(
            self.grid_size ** 3 if num_scenes is None else (num_scenes, self.grid_size ** 3),
            device=device, dtype=torch.float16)

    def get_init_density_bitfield(self, num_scenes, device=None):
        return torch.zeros(
            self.grid_size ** 3 // 8 if num_scenes is None else (num_scenes, self.grid_size ** 3 // 8),
            device=device, dtype=torch.uint8)

    @staticmethod
    def build_optimizer(code_, cfg):
        optimizer_cfg = cfg['optimizer'].copy()
        optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
        if isinstance(code_, list):
            code_optimizer = [
                optimizer_class([code_single_], **optimizer_cfg)
                for code_single_ in code_]
        else:
            code_optimizer = optimizer_class([code_], **optimizer_cfg)
        return code_optimizer

    @staticmethod
    def build_scheduler(code_optimizer, cfg):
        if 'lr_scheduler' in cfg:
            scheduler_cfg = cfg['lr_scheduler'].copy()
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_cfg.pop('type'))
            if isinstance(code_optimizer, list):
                code_scheduler = [
                    scheduler_class(code_optimizer_single, **scheduler_cfg)
                    for code_optimizer_single in code_optimizer]
            else:
                code_scheduler = scheduler_class(code_optimizer, **scheduler_cfg)
        else:
            code_scheduler = None
        return code_scheduler

    @staticmethod
    def ray_sample(cond_rays_o, cond_rays_d, cond_imgs, n_samples, sample_inds=None, ortho=True):
        """
        Args:
            cond_rays_o (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
            cond_rays_d (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
            cond_imgs (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
            n_samples (int): number of samples
            sample_inds (None | torch.Tensor): (num_scenes, n_samples)

        Returns:
            rays_o (torch.Tensor): (num_scenes, n_samples, 3)
            rays_d (torch.Tensor): (num_scenes, n_samples, 3)
            target_rgbs (torch.Tensor): (num_scenes, n_samples, 3)
        """
        device = cond_rays_o.device
        num_scenes, num_imgs, h, w, _ = cond_rays_o.size()
        num_scene_pixels = num_imgs * h * w
        rays_o = cond_rays_o.reshape(num_scenes, num_scene_pixels, 3)
        rays_d = cond_rays_d.reshape(num_scenes, num_scene_pixels, 3)
        target_rgbs = cond_imgs.reshape(num_scenes, num_scene_pixels, 3)
        
        if num_scene_pixels > n_samples:
            if sample_inds is None:
                sample_inds = [torch.randperm(
                    target_rgbs.size(1), device=device)[:n_samples] for _ in range(num_scenes)]
                sample_inds = torch.stack(sample_inds, dim=0)
            scene_arange = torch.arange(num_scenes, device=device)[:, None]
            rays_o = rays_o[scene_arange, sample_inds]
            rays_d = rays_d[scene_arange, sample_inds]
            target_rgbs = target_rgbs[scene_arange, sample_inds]
        return rays_o, rays_d, target_rgbs

    @staticmethod
    def get_raybatch_inds(cond_imgs, n_inverse_rays):
        device = cond_imgs.device
        num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        num_scene_pixels = num_imgs * h * w
        if num_scene_pixels > n_inverse_rays:
            raybatch_inds = [torch.randperm(num_scene_pixels, device=device) for _ in range(num_scenes)]
            raybatch_inds = torch.stack(raybatch_inds, dim=0).split(n_inverse_rays, dim=1)
            num_raybatch = len(raybatch_inds)
        else:
            raybatch_inds = num_raybatch = None
        return raybatch_inds, num_raybatch

    def loss(self, decoder, code, target_rgbs, cameras,   
            dt_gamma=0.0, smpl_params=None, return_decoder_loss=False, scale_num_ray=1.0,
             cfg=dict(), init=False, norm=None, **kwargs):
        num_imgs = target_rgbs.shape[1]
        outputs = decoder(
            code, self.grid_size, smpl_params, cameras,
            num_imgs, dt_gamma=dt_gamma, perturb=True, return_loss=return_decoder_loss, init=init, return_norm=(norm!=None))
        out_rgbs = outputs['image']
        out_offsets = outputs['offset']
        
        loss = 0
        scale = 1 - math.exp(-cfg['loss_coef'] * scale_num_ray) if 'loss_coef' in cfg else 1
        pixel_loss = self.pixel_loss(out_rgbs, target_rgbs, **kwargs) * (scale * 3) # 2.5
        if norm != None:
            out_norms = outputs['norm']
            pix_norm_loss = self.pixel_loss(out_norms, norm, **kwargs) * (scale * 3) # 2.5
            pixel_loss = pixel_loss * 0.7 + pix_norm_loss * 0.3
        loss = loss + pixel_loss
        loss_dict = dict(pixel_loss=pixel_loss)
        reg_offset = out_offsets
        loss = loss + reg_offset * cfg['offset_weight'] 
        loss_dict.update(reg_offset=reg_offset*cfg['offset_weight'])

        if self.reg_loss is not None:
            reg_loss = self.reg_loss(code, **kwargs)
            loss = loss + reg_loss
            loss_dict.update(reg_loss=reg_loss)
        if self.per_loss is not None:
            if self.per_loss.loss_weight > 0:
                per_loss = self.per_loss(out_rgbs[:, 0], target_rgbs[:, 0])
                if norm != None:
                    per_loss_norm = self.per_loss(out_norms[:, 0], target_rgbs[:, 0])
                    per_loss = per_loss * 0.7 + per_loss_norm * 0.3
                loss = loss + per_loss
                loss_dict.update(per_loss=per_loss)
        if self.scale_loss_weight > 0:
            out_scales = outputs['scales'][..., :2]
            scale_reg_loss = (out_scales.max(-1).values/out_scales.min(-1).values - 5).clip(min=0).mean() * self.scale_loss_weight
            loss = loss + scale_reg_loss
            loss_dict.update(scale_reg_loss=scale_reg_loss)
        if self.norm_reg_weight > 0:
            out_alphas = outputs['alphas'][..., 0]
            norm_reg_loss = (out_alphas.detach() * (1.0 - torch.sum(outputs['norm'] * outputs['norm_maps'], axis=-1))).mean() * self.norm_reg_weight
            loss = loss + norm_reg_loss
            loss_dict.update(norm_reg_loss=norm_reg_loss)
        if return_decoder_loss and outputs['decoder_reg_loss'] is not None:
            decoder_reg_loss = outputs['decoder_reg_loss']
            loss = loss + decoder_reg_loss
            loss_dict.update(decoder_reg_loss=decoder_reg_loss)
        return out_rgbs, loss, loss_dict

    def loss_decoder(self, decoder, code, cond_imgs, cameras, smpl_params=None, 
        dt_gamma=0.0, cfg=dict(), densify=False, init=False, cond_norm=None, **kwargs):
        decoder_training_prev = decoder.training
        decoder.train(True)
        assert smpl_params != None

        num_scenes, num_imgs, _, _, _ = cond_imgs.shape
        select_imgs = cond_imgs[:, 3::4]
        select_cameras = cameras[:, 3::4]
        if cond_norm != None:
            select_norm = cond_norm[:, 3::4]
        else:
            select_norm = None

        out_rgbs, loss, loss_dict = self.loss(
            decoder, code, select_imgs, select_cameras, 
            dt_gamma, smpl_params=smpl_params, return_decoder_loss=True, scale_num_ray=cond_imgs.shape[1:4].numel(),
            cfg=cfg, init=init, norm=select_norm, **kwargs)
        log_vars = dict()
        for key, val in loss_dict.items():
            log_vars.update({key: float(val)})

        decoder.train(decoder_training_prev)
        return loss, log_vars, out_rgbs, select_imgs

    def update_extra_state(self, decoder, code, masks, points,
                           iter_density, smpl_params=None, density_thresh=0.01, scale_thresh=0.05, decay=0.9, S=128, offset_thresh=0.005):
        with torch.no_grad():
            device = get_module_device(self)
            num_scenes, num_points = masks.shape
            if isinstance(decoder, DistributedDataParallel):
                decoder = decoder.module
            sigmas, _, _, _ = decoder.point_density_decode(
                                points, code, smpl_params=None)
            sigmas = sigmas.reshape(num_scenes, -1)
            masks = sigmas > density_thresh
            return masks, points


    def get_density(self, decoder, code, cfg=dict()):
        density_thresh = cfg.get('density_thresh', 0.005)
        density_step = cfg.get('density_step', 8)
        num_scenes = code.size(0)
        try:
            num_init = decoder.module.num_init
        except:
            num_init = decoder.num_init
        device = code.device
        masks = self.get_init_mask_(num_scenes, num_init, device)
        try:
            points = self.get_init_points_(num_scenes, decoder.module.init_pcd, device)
        except:
            points = self.get_init_points_(num_scenes, decoder.init_pcd, device)
        masks, points = self.update_extra_state(decoder, code, masks, points, 0,
                                density_thresh=density_thresh, decay=1.0)
        return masks, points

    def sample_imgs(self, cond_imgs, cameras, num_scenes, num_imgs, num_samples, device):
        sample_inds = [torch.randperm(num_imgs, device=device)[:num_samples] for _ in range(num_scenes)]
        sample_inds = torch.stack(sample_inds, dim=0)
        scene_arange = torch.arange(num_scenes, device=device)[:, None]
        cond_imgs_select = cond_imgs[scene_arange, sample_inds]
        cameras_select = cameras[scene_arange, sample_inds]
        return cond_imgs_select, cameras_select

    def inverse_code(self, decoder, cond_imgs, cameras, dt_gamma=0, smpl_params=None, cfg=dict(),
                     code_=None, iter_density=None, init=None, cond_norm=None,
                     code_optimizer=None, code_scheduler=None,
                     prior_grad=None, show_pbar=False, densify=False):
        """
        Obtain scene codes via optimization-based inverse rendering.
        """
        device = get_module_device(self)
        decoder_training_prev = decoder.training
        decoder.train(True)
        assert smpl_params != None

        with module_requires_grad(decoder, False):
            n_inverse_steps = cfg.get('n_inverse_steps', 4096)

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            assert n_inverse_steps * 4 <= num_imgs
            num_scene_pixels = num_imgs * h * w
            if code_ is None:
                code_ = self.get_init_code_(num_scenes, device=device)
            if iter_density is None:
                iter_density = 0

            if code_optimizer is None:
                assert code_scheduler is None
                code_optimizer = self.build_optimizer(code_, cfg)
            if code_scheduler is None:
                code_scheduler = self.build_scheduler(code_optimizer, cfg)

            assert n_inverse_steps > 0
            if show_pbar:
                pbar = mmcv.ProgressBar(n_inverse_steps)
            
            for inverse_step_id in range(n_inverse_steps):
                code = self.code_activation(
                    torch.stack(code_, dim=0) if isinstance(code_, list)
                    else code_)
                
                select_imgs = cond_imgs[:, inverse_step_id::4]
                select_cameras = cameras[:, inverse_step_id::4]
                if cond_norm != None:
                    select_norm = cond_norm[:, inverse_step_id::4]
                else:
                    select_norm = None

                out_rgbs, loss, loss_dict = self.loss(
                    decoder, code,
                    select_imgs, select_cameras, dt_gamma, smpl_params=smpl_params, scale_num_ray=num_scene_pixels,
                    cfg=cfg, init=init, norm=select_norm)
                
                if prior_grad is not None:
                    if isinstance(code_, list):
                        for code_single_, prior_grad_single in zip(code_, prior_grad):
                            code_single_.grad.copy_(prior_grad_single)
                    else:
                        code_.grad.copy_(prior_grad)
                else:
                    if isinstance(code_optimizer, list):
                        for code_optimizer_single in code_optimizer:
                            code_optimizer_single.zero_grad()
                    else:
                        code_optimizer.zero_grad()

                loss.backward()

                if isinstance(code_optimizer, list):
                    for code_optimizer_single in code_optimizer:
                        code_optimizer_single.step()
                else:
                    code_optimizer.step()

                if code_scheduler is not None:
                    if isinstance(code_scheduler, list):
                        for code_scheduler_single in code_scheduler:
                            code_scheduler_single.step()
                    else:
                        code_scheduler.step()

                if show_pbar:
                    pbar.update()
            
        decoder.train(decoder_training_prev)
        return code.detach(), loss, loss_dict

    def render(self, decoder, code, h, w, intrinsics, poses, smpl_params, cfg=dict(), mask=None, return_norm=False, return_viz=False):
        decoder_training_prev = decoder.training
        decoder.train(False)
        dt_gamma = 0.0
        
        num_scenes, num_imgs, _, _ = poses.size()
        cameras = torch.cat([intrinsics, poses.reshape(num_scenes, num_imgs, -1)], dim=-1)
        outputs = decoder(
            code, self.grid_size, smpl_params, cameras,
            num_imgs, mask=mask, dt_gamma=dt_gamma, perturb=False)
        out_image = outputs['image']
        out_vizmask = outputs['viz_masks']
        
        if return_norm:
            out_norm = outputs['norm']
            out_norm = out_norm.reshape(num_scenes, num_imgs, h, w, 3)
        else:
            out_norm = None
        out_image = out_image.reshape(num_scenes, num_imgs, h, w, 3)

        decoder.train(decoder_training_prev)
        if return_viz:
            return out_image, out_norm, out_vizmask
        else:
            return out_image, out_norm

    def eval_and_viz(self, data, decoder, code, viz_dir=None, cfg=dict(), ortho=True, recon=False, return_norm=False):
        scene_name = data['scene_name']  # (num_scenes,)
        if recon:
            # test_intrinsics = data['cond_intrinsics'][:, 1::3]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['cond_poses'][:, 1::3]
            # test_smpl_param = data['cond_smpl_param']
            test_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            test_poses = data['cond_poses']
            test_smpl_param = data['cond_smpl_param']
        else:
            # test_intrinsics = data['test_intrinsics'][:,1::3]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['test_poses'][:,1::3]
            test_intrinsics = data['test_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            test_poses = data['test_poses']
            test_smpl_param = data['test_smpl_param']
        num_scenes, num_imgs, _, _ = test_poses.size()
        if 'test_imgs' in data and not cfg.get('skip_eval', False):
            test_imgs = data['test_imgs']  # (num_scenes, num_imgs, h, w, 3)
            _, _, h, w, _ = test_imgs.size()
            test_img_paths = data['test_img_paths']  # (num_scenes, (num_imgs,))
            target_imgs = test_imgs.permute(0, 1, 4, 2, 3).reshape(num_scenes * num_imgs, 4, h, w)[:, :3]
        else:
            test_imgs = test_img_paths = target_imgs = None
            h, w = cfg['img_size']
        if test_smpl_param.dim() == 3:
            image = []
            for num in range(test_smpl_param.shape[1]):
                image_single, norm_map = self.render(
                    decoder, code, h, w, test_intrinsics, test_poses, test_smpl_param[:, num], cfg=cfg, return_norm=return_norm)
                image.append(image_single)
            image = torch.cat(image, dim=1)
        elif test_smpl_param.dim() == 2:
            image, norm_map = self.render(
                decoder, code, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg, return_norm=return_norm)
        else:
            assert False
        pred_imgs = image.permute(0, 1, 4, 2, 3).reshape(
            num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
        pred_imgs = torch.round(pred_imgs * 255) / 255
        
        if return_norm:
            pred_norms = norm_map.permute(0, 1, 4, 2, 3).reshape(
            num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
            pred_norms = torch.round(pred_norms * 255) / 255

        if test_imgs is not None:
            test_psnr = eval_psnr(pred_imgs, target_imgs)
            test_ssim = eval_ssim_skimage(pred_imgs, target_imgs, data_range=1)
            log_vars = dict(test_psnr=float(test_psnr.mean()),
                            test_ssim=float(test_ssim.mean()))
            if self.lpips is not None:
                if len(self.lpips) == 0:
                    lpips_eval = lpips.LPIPS(
                        net='vgg', eval_mode=True, pnet_tune=False).to(pred_imgs.device)
                    self.lpips.append(lpips_eval)
                test_lpips = []
                for pred_imgs_batch, target_imgs_batch in zip(
                        pred_imgs.split(LPIPS_BS, dim=0), target_imgs.split(LPIPS_BS, dim=0)):
                    test_lpips.append(self.lpips[0](pred_imgs_batch * 2 - 1, target_imgs_batch * 2 - 1).flatten())
                test_lpips = torch.cat(test_lpips, dim=0)
                log_vars.update(test_lpips=float(test_lpips.mean()))
            else:
                test_lpips = [math.nan for _ in range(num_scenes * num_imgs)]
        else:
            log_vars = dict()
        
        if viz_dir is None:
            viz_dir = cfg.get('viz_dir', None)
        if viz_dir is not None:
            os.makedirs(viz_dir, exist_ok=True)
            output_viz = torch.round(pred_imgs.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
            if return_norm:
                output_norm_viz = torch.round(pred_norms.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
            if test_imgs is not None:
                real_imgs_viz = (target_imgs.permute(0, 2, 3, 1) * 255).to(
                    torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 4)[..., :3]
                output_viz = np.concatenate([real_imgs_viz, output_viz], axis=-2)
            for scene_id, scene_name_single in enumerate(scene_name):
                for img_id in range(num_imgs):
                    if test_img_paths is not None:
                        base_name = 'scene_' + scene_name_single + '_' + os.path.splitext(
                            os.path.basename(test_img_paths[scene_id][img_id]))[0]
                        name = base_name + '_psnr{:02.1f}_ssim{:.2f}_lpips{:.3f}.png'.format(
                            test_psnr[scene_id * num_imgs + img_id],
                            test_ssim[scene_id * num_imgs + img_id],
                            test_lpips[scene_id * num_imgs + img_id])
                        existing_files = glob(os.path.join(viz_dir, base_name + '*.png'))
                        for file in existing_files:
                            os.remove(file)
                    else:
                        name = 'scene_' + scene_name_single + '_{:03d}.png'.format(img_id)
                        norm_name = 'scene_' + scene_name_single + '_{:03d}_normal.png'.format(img_id)
                    plt.imsave(
                        os.path.join(viz_dir, name),
                        output_viz[scene_id][img_id][:,:,:3])
                    if return_norm:
                        plt.imsave(
                            os.path.join(viz_dir, norm_name),
                            output_norm_viz[scene_id][img_id][:,:,:3])
            if isinstance(decoder, DistributedDataParallel):
                decoder = decoder.module
            code_range = cfg.get('clip_range', [-1, 1])
            decoder.visualize(code, scene_name, viz_dir, code_range=code_range)
            if self.init_code is not None:
                decoder.visualize(self.init_code[None], ['000_mean'], viz_dir, code_range=code_range)
        return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w)

    def mean_ema_update(self, code):
        if self.init_code is None:
            return
        mean_code = reduce_mean(code.detach().mean(dim=0))
        self.init_code.mul_(1 - self.mean_ema_momentum).add_(
            mean_code.data, alpha=self.mean_ema_momentum)

    def train_step(self, data, optimizer, running_status=None):
        raise NotImplementedError

    def val_step(self, data, viz_dir=None, show_pbar=False, **kwargs):
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        if 'code' in data:
            code, density_grid, density_bitfield = self.load_scene(
                data, load_density=True)
            out_rgbs = target_rgbs = None
        else:
            cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
            cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            cond_poses = data['cond_poses']
            smpl_params = data['smpl_params']

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            # (num_scenes, num_imgs, h, w, 3)
            cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
            # (num_scenes,)
            dt_gamma = 0.0

            with torch.enable_grad():
                (code, density_grid, density_bitfield,
                 loss, loss_dict, out_rgbs, target_rgbs) = self.inverse_code(
                    decoder, cond_imgs, cond_rays_o, cond_rays_d,
                    dt_gamma=dt_gamma, smpl_params=smpl_params, cfg=self.test_cfg, show_pbar=show_pbar)

        # ==== evaluate reconstruction ====
        with torch.no_grad():
            if 'test_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz(
                    data, decoder, code, density_bitfield,
                    viz_dir=viz_dir, smpl_params=smpl_params, cfg=self.test_cfg)
            else:
                log_vars = dict()
                pred_imgs = None
            if out_rgbs is not None and target_rgbs is not None:
                train_psnr = eval_psnr(out_rgbs, target_rgbs)
                log_vars.update(train_psnr=float(train_psnr.mean()))
            code_rms = code.square().flatten(1).mean().sqrt()
            log_vars.update(code_rms=float(code_rms.mean()))

        # ==== save 3D code ====
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            self.save_scene(save_dir, code, density_grid, density_bitfield, data['scene_name'])

        # ==== outputs ====
        outputs_dict = dict(
            log_vars=log_vars,
            num_samples=len(data['scene_name']),
            pred_imgs=pred_imgs)

        return outputs_dict
