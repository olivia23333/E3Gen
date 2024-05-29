import torch
import mmcv

from copy import deepcopy
import imageio
import os
import time
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel
from mmgen.models.builder import MODELS, build_module
from mmgen.models.architectures.common import get_module_device

from ...core import eval_psnr, rgetattr, module_requires_grad, get_cam_rays
from .multiscene_nerf import MultiSceneNeRF


@MODELS.register_module()
class DiffusionNeRF(MultiSceneNeRF):

    def __init__(self,
                 *args,
                 diffusion=dict(type='GaussianDiffusion'),
                 diffusion_use_ema=True,
                 freeze_decoder=True,
                 image_cond=False,
                 code_permute=None,
                 code_reshape=None,
                 autocast_dtype=None,
                 ortho=True,
                 return_norm=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        diffusion.update(train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        # torch.autograd.set_detect_anomaly(True)
        self.diffusion = build_module(diffusion)
        self.diffusion_use_ema = diffusion_use_ema
        if self.diffusion_use_ema:
            self.diffusion_ema = deepcopy(self.diffusion)
        self.freeze_decoder = freeze_decoder
        if self.freeze_decoder:
            self.decoder.requires_grad_(False)
            if self.decoder_use_ema:
                self.decoder_ema.requires_grad_(False)
        self.image_cond = image_cond
        self.code_permute = code_permute
        self.code_reshape = code_reshape
        self.code_reshape_inv = [self.code_size[axis] for axis in self.code_permute] if code_permute is not None \
            else self.code_size
        self.code_permute_inv = [self.code_permute.index(axis) for axis in range(len(self.code_permute))] \
            if code_permute is not None else None

        self.autocast_dtype = autocast_dtype
        self.ortho = ortho
        self.return_norm = return_norm

        for key, value in self.test_cfg.get('override_cfg', dict()).items():
            self.train_cfg_backup[key] = rgetattr(self, key)

    def code_diff_pr(self, code):
        code_diff = code
        if self.code_permute is not None:
            code_diff = code_diff.permute([0] + [axis + 1 for axis in self.code_permute])  # add batch dimension
        if self.code_reshape is not None:
            code_diff = code_diff.reshape(code.size(0), *self.code_reshape)  # add batch dimension
        return code_diff

    def code_diff_pr_inv(self, code_diff):
        code = code_diff
        if self.code_reshape is not None:
            code = code.reshape(code.size(0), *self.code_reshape_inv)
        if self.code_permute_inv is not None:
            code = code.permute([0] + [axis + 1 for axis in self.code_permute_inv])
        return code

    def train_step(self, data, optimizer, running_status=None):
        iter = running_status['iteration']
        diffusion = self.diffusion
        decoder = self.decoder_ema if self.freeze_decoder and self.decoder_use_ema else self.decoder
        num_scenes = len(data['scene_id']) # 8
        extra_scene_step = self.train_cfg.get('extra_scene_step', 0) # 15
        
        if 'optimizer' in self.train_cfg:
            code_list_, code_optimizers = self.load_cache(data)
            code = self.code_activation(torch.stack(code_list_, dim=0), update_stats=True)
        else:
            assert 'code' in data
            code = self.load_scene(data, load_density='decoder' in optimizer)
            code_optimizers = []

        for key in optimizer.keys():
            if key.startswith('diffusion'):
                optimizer[key].zero_grad()
        for code_optimizer in code_optimizers:
            code_optimizer.zero_grad()
        if 'decoder' in optimizer:
            optimizer['decoder'].zero_grad()

        concat_cond = None
        if 'cond_imgs' in data:
            cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
            cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            cond_poses = data['cond_poses'] # (num_scenes, num_imgs, 4, 4)
            smpl_params = data['cond_smpl_param']
            if 'cond_norm' in data:
                cond_norm = data['cond_norm']
            else:
                cond_norm = None

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            cameras = torch.cat([cond_intrinsics, cond_poses.reshape(num_scenes, num_imgs, -1)], dim=-1)
            target_imgs = cond_imgs
            
            dt_gamma = 0.0
            
            if self.image_cond:
                cond_inds = torch.randint(num_imgs, size=(num_scenes,))  # (num_scenes,)
                concat_cond = cond_imgs[range(num_scenes), cond_inds].permute(0, 3, 1, 2)  # (num_scenes, 3, h, w)
                diff_image_size = rgetattr(diffusion, 'denoising.image_size')
                assert diff_image_size[0] % concat_cond.size(-2) == 0
                assert diff_image_size[1] % concat_cond.size(-1) == 0
                concat_cond = concat_cond.tile((diff_image_size[0] // concat_cond.size(-2),
                                                diff_image_size[1] // concat_cond.size(-1)))

        x_t_detach = self.train_cfg.get('x_t_detach', False)

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            loss_diffusion, log_vars = diffusion(
                self.code_diff_pr(code), concat_cond=concat_cond, return_loss=True,
                x_t_detach=x_t_detach, cfg=self.train_cfg)
        loss_diffusion.backward()
        for key in optimizer.keys():
            if key.startswith('diffusion'):
                optimizer[key].step()

        if extra_scene_step > 0:
            assert len(code_optimizers) > 0
            prior_grad = [code_.grad.data.clone() for code_ in code_list_]
            cfg = self.train_cfg.copy()
            cfg['n_inverse_steps'] = extra_scene_step
            code, loss_decoder, loss_dict_decoder = self.inverse_code(
                decoder, target_imgs, cameras, dt_gamma=dt_gamma, smpl_params=smpl_params, cfg=cfg,
                code_=code_list_,
                code_optimizer=code_optimizers,
                prior_grad=prior_grad,
                densify=(iter > cfg['densify_start_iter']),
                init=(iter < cfg['init_iter']),
                cond_norm=cond_norm)
            for k, v in loss_dict_decoder.items():
                log_vars.update({k: float(v)})
        else:
            cfg = self.train_cfg.copy()
            prior_grad = [code_.grad.data.clone() for code_ in code_list_]
        

        if 'decoder' in optimizer or len(code_optimizers) > 0:
            if len(code_optimizers) > 0:
                code = self.code_activation(torch.stack(code_list_, dim=0))

            loss_decoder, log_vars_decoder, out_rgbs, target_rgbs = self.loss_decoder(
                decoder, code, target_imgs, cameras, smpl_params, 
                dt_gamma, cfg=self.train_cfg, densify=(iter > cfg['densify_start_iter']), init=(iter < cfg['init_iter']), cond_norm=cond_norm)

            log_vars.update(log_vars_decoder)

            if prior_grad is not None:
                for code_, prior_grad_single in zip(code_list_, prior_grad):
                    code_.grad.copy_(prior_grad_single)
            loss_decoder.backward()

            if 'decoder' in optimizer:
                optimizer['decoder'].step()
            for code_optimizer in code_optimizers:
                code_optimizer.step()

            # ==== save cache ====
            self.save_cache(
                code_list_, code_optimizers, 
                data['scene_id'], data['scene_name'])

            # ==== evaluate reconstruction ====
            with torch.no_grad():
                if len(code_optimizers) > 0:
                    self.mean_ema_update(code)
                train_psnr = eval_psnr(out_rgbs, target_rgbs)
                code_rms = code.square().flatten(1).mean().sqrt()
                log_vars.update(train_psnr=float(train_psnr.mean()),
                                code_rms=float(code_rms.mean()))
                if 'test_imgs' in data and data['test_imgs'] is not None:
                    log_vars.update(self.eval_and_viz(
                        data, self.decoder, code, smpl_params, cfg=self.train_cfg, ortho=self.ortho, return_norm=self.return_norm)[0])

        # ==== outputs ====
        if 'decoder' in optimizer or len(code_optimizers) > 0:
            log_vars.update(loss_decoder=float(loss_decoder))
        outputs_dict = dict(
            log_vars=log_vars, num_samples=num_scenes)

        return outputs_dict

    def val_uncond(self, data, show_pbar=False, **kwargs):
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        num_batches = len(data['scene_id'])
        noise = data.get('noise', None)
        if noise is None:
            noise = torch.randn(
                (num_batches, *self.code_size), device=get_module_device(self))

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            code_out = diffusion(
                self.code_diff_pr(noise), return_loss=False,
                show_pbar=show_pbar, **kwargs)
        code_list = code_out if isinstance(code_out, list) else [code_out]
        for step_id, code in enumerate(code_list):
            code = self.code_diff_pr_inv(code)
            n_inverse_steps = self.test_cfg.get('n_inverse_steps', 0)
            if n_inverse_steps > 0 and step_id == (len(code_list) - 1):
                with module_requires_grad(diffusion, False), torch.enable_grad():
                    code_ = self.code_activation.inverse(code).requires_grad_(True)
                    code_optimizer = self.build_optimizer(code_, self.test_cfg)
                    code_scheduler = self.build_scheduler(code_optimizer, self.test_cfg)
                    if show_pbar:
                        pbar = mmcv.ProgressBar(n_inverse_steps)
                    for inverse_step_id in range(n_inverse_steps):
                        code_optimizer.zero_grad()
                        code = self.code_activation(code_)
                        loss, log_vars = diffusion(self.code_diff_pr(code), return_loss=True, cfg=self.test_cfg)
                        loss.backward()
                        code_optimizer.step()
                        if code_scheduler is not None:
                            code_scheduler.step()
                        if show_pbar:
                            pbar.update()
                code = self.code_activation(code_)
            code_list[step_id] = code
        if isinstance(code_out, list):
            return code_list
        else:
            return code_list[-1]

    def val_step(self, data, viz_dir=None, viz_dir_guide=None, **kwargs):
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder
        with torch.no_grad():
            assert 'cond_imgs' not in data
            if 'code' in data:
                code = self.load_scene(data, load_density=True)
            else:
                code = self.val_uncond(data, **kwargs)
            # ==== evaluate reconstruction ====
            if 'test_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz(
                    data, decoder, code,
                    viz_dir=viz_dir, cfg=self.test_cfg, ortho=self.ortho, return_norm=self.return_norm)
            elif 'cond_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz(
                    data, decoder, code,
                    viz_dir=viz_dir, cfg=self.test_cfg, ortho=self.ortho, recon=True, return_norm=self.return_norm)
            else:
                log_vars = dict()
                pred_imgs = None
                if viz_dir is None:
                    viz_dir = self.test_cfg.get('viz_dir', None)
                if viz_dir is not None:
                    if isinstance(decoder, DistributedDataParallel):
                        decoder = decoder.module
                    decoder.visualize(
                        code, data['scene_name'],
                        viz_dir, code_range=self.test_cfg.get('clip_range', [-1, 1]))
        # ==== save 3D code ====
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            self.save_scene(save_dir, code, data['scene_name'])
            save_mesh = self.test_cfg.get('save_mesh', False)
            if save_mesh:
                mesh_resolution = self.test_cfg.get('mesh_resolution', 256)
                mesh_threshold = self.test_cfg.get('mesh_threshold', 10)
                self.save_mesh(save_dir, decoder, code, data['scene_name'], mesh_resolution, mesh_threshold)
        # ==== outputs ====
        outputs_dict = dict(
            log_vars=log_vars,
            num_samples=len(data['scene_name']),
            pred_imgs=pred_imgs)
        return outputs_dict