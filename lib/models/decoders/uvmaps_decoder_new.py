import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from pytorch3d import ops

from mmcv.cnn import xavier_init, constant_init
from mmgen.models.builder import MODULES
import numpy as np
import time
import cv2
import math
from simple_knn._C import distCUDA2
from pytorch3d.transforms import quaternion_to_matrix

from .base_volume_renderer import VolumeRenderer
from ..deformers import SMPLXDeformer
from ..renderers import GRenderer, get_covariance, batch_rodrigues
from ..superres import SuperresolutionHybrid2X, SuperresolutionHybrid4X
from lib.ops import TruncExp


class PositionalEncoding():
    def __init__(self, input_dims=5, num_freqs=1, include_input=True):
        super(PositionalEncoding,self).__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.input_dims = input_dims
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        freq_bands = 2. ** torch.linspace(0, self.num_freqs-1, self.num_freqs)
        periodic_fns = [torch.sin, torch.cos]

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(math.pi * x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self,coords):
        '''
        use periodic positional encoding to transform cartesian positions to higher dimension
        :param coords: [N, 3]
        :return: [N, 3*2*num_freqs], where 2 comes from that for each frequency there's a sin() and cos()
        '''
        return torch.cat([fn(coords) for fn in self.embed_fns], dim=-1)

@MODULES.register_module()
class UVNDecoder(VolumeRenderer):

    activation_dict = {
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'softplus': nn.Softplus,
        'trunc_exp': TruncExp,
        'sigmoid': nn.Sigmoid}

    def __init__(self,
                 *args,
                 interp_mode='bilinear',
                 base_layers=[3 * 32, 128],
                 density_layers=[128, 1],
                 color_layers=[128, 128, 3],
                 offset_layers=[128, 3],
                 scale_layers=[128, 3],
                 radius_layers=[128, 3],
                 use_dir_enc=True,
                 dir_layers=None,
                 scene_base_size=None,
                 scene_rand_dims=(0, 1),
                 activation='silu',
                 sigma_activation='sigmoid',
                 sigmoid_saturation=0.001,
                 code_dropout=0.0,
                 flip_z=False,
                 extend_z=False,
                 gender='neutral',
                 multires=0,
                 bg_color=0,
                 image_size=1024,
                 superres=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.interp_mode = interp_mode
        self.in_chn = base_layers[0]
        self.use_dir_enc = use_dir_enc
        if scene_base_size is None:
            self.scene_base = None
        else:
            rand_size = [1 for _ in scene_base_size]
            for dim in scene_rand_dims:
                rand_size[dim] = scene_base_size[dim]
            init_base = torch.randn(rand_size).expand(scene_base_size).clone()
            self.scene_base = nn.Parameter(init_base)
        self.dir_encoder = None
        self.sigmoid_saturation = sigmoid_saturation
        self.deformer = SMPLXDeformer(gender)
        self.renderer = GRenderer(image_size=image_size, bg_color=bg_color, f=5000)
        if superres:
            self.superres = None
        else:
            self.superres = None

        select_uv = torch.as_tensor(np.load('work_dirs/cache/init_uv_smplx_thu.npy'))
        self.register_buffer('select_coord', select_uv.unsqueeze(0)*2.-1.)

        init_pcd = torch.as_tensor(np.load('work_dirs/cache/init_pcd_smplx_thu.npy'))
        self.register_buffer('init_pcd', init_pcd.unsqueeze(0), persistent=False)
        self.num_init = self.init_pcd.shape[1]

        self.multires = multires
        if multires > 0:
            uv_map = torch.as_tensor(np.load('work_dirs/cache/init_uvmap_smplx_thu.npy'))
            pcd_map = torch.as_tensor(np.load('work_dirs/cache/init_posmap_smplx_thu.npy'))
            input_coord = torch.cat([pcd_map, uv_map], dim=1)
            self.register_buffer('input_freq', input_coord, persistent=False)
            base_layers[0] += 5
            color_layers[0] += 5
        else:
            self.init_uv = None

        activation_layer = self.activation_dict[activation.lower()]


        base_net = [] # linear (in=18, out=64, bias=True)
        for i in range(len(base_layers) - 1):
            base_net.append(nn.Conv2d(base_layers[i], base_layers[i + 1], 3, padding=1))
            if i != len(base_layers) - 2:
                base_net.append(nn.BatchNorm2d(base_layers[i+1]))
                base_net.append(activation_layer())
        self.base_net = nn.Sequential(*base_net)
        self.base_bn = nn.BatchNorm2d(base_layers[-1])
        self.base_activation = activation_layer()

        density_net = [] # linear(in=64, out=1, bias=True), sigmoid
        for i in range(len(density_layers) - 1):
            density_net.append(nn.Conv2d(density_layers[i], density_layers[i + 1], 1))
            if i != len(density_layers) - 2:
                density_net.append(nn.BatchNorm2d(density_layers[i+1]))
                density_net.append(activation_layer())
        density_net.append(self.activation_dict[sigma_activation.lower()]())
        self.density_net = nn.Sequential(*density_net)

        offset_net = [] # linear(in=64, out=1, bias=True), sigmoid
        for i in range(len(offset_layers) - 1):
            offset_net.append(nn.Conv2d(offset_layers[i], offset_layers[i + 1], 1))
            if i != len(offset_layers) - 2:
                offset_net.append(nn.BatchNorm2d(offset_layers[i+1]))
                offset_net.append(activation_layer())
        self.offset_net = nn.Sequential(*offset_net)

        self.dir_net = None
        color_net = [] # linear(in=64, out=3, bias=True), sigmoid
        for i in range(len(color_layers) - 2):
            color_net.append(nn.Conv2d(color_layers[i], color_layers[i + 1], kernel_size=3, padding=1))
            color_net.append(nn.BatchNorm2d(color_layers[i+1]))
            color_net.append(activation_layer())
        color_net.append(nn.Conv2d(color_layers[-2], color_layers[-1], kernel_size=1))
        color_net.append(nn.Sigmoid())
        self.color_net = nn.Sequential(*color_net)
        self.code_dropout = nn.Dropout2d(code_dropout) if code_dropout > 0 else None

        self.flip_z = flip_z
        self.extend_z = extend_z

        init_rot = torch.as_tensor(np.load('work_dirs/cache/init_rot_smplx_thu.npy'))
        self.register_buffer('init_rot', init_rot, persistent=False)

        face_mask = torch.as_tensor(np.load('work_dirs/cache/face_mask_thu.npy'))
        self.register_buffer('face_mask', face_mask.unsqueeze(0), persistent=False)

        hands_mask = torch.as_tensor(np.load('work_dirs/cache/hands_mask_thu.npy'))
        self.register_buffer('hands_mask', hands_mask.unsqueeze(0), persistent=False)

        outside_mask = torch.as_tensor(np.load('work_dirs/cache/outside_mask_thu.npy'))
        self.register_buffer('outside_mask', outside_mask.unsqueeze(0), persistent=False)

        # mouth_in_mask = torch.as_tensor(np.load('work_dirs/cache/mouth_in_mask_thu.npy'))
        # self.register_buffer('mouth_in_mask', mouth_in_mask.unsqueeze(0), persistent=False)

        # nose_mask = torch.as_tensor(np.load('work_dirs/cache/nose_mask_thu.npy'))
        # self.register_buffer('nose_mask', nose_mask.unsqueeze(0), persistent=False)

        # lip_mask = torch.as_tensor(np.load('work_dirs/cache/nose_lip_thu.npy'))
        # self.register_buffer('lip_mask', lip_mask.unsqueeze(0), persistent=False)

        # feet_mask = torch.as_tensor(np.load('work_dirs/cache/feet_mask_thu.npy'))
        # self.register_buffer('feet_mask', feet_mask.unsqueeze(0), persistent=False)

        # ear_mask = torch.as_tensor(np.load('work_dirs/cache/ear_mask_thu.npy'))
        # self.register_buffer('ear_mask', ear_mask.unsqueeze(0), persistent=False)

        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        if self.dir_net is not None:
            constant_init(self.dir_net[-1], 0)
        if self.offset_net is not None:
            self.offset_net[-1].weight.data.uniform_(-1e-5, 1e-5)
            self.offset_net[-1].bias.data.zero_()

    def xyz_transform(self, xyz, smpl_params=None, num_scenes=1):
        assert xyz.dim() == 3
        self.deformer.prepare_deformer(smpl_params, num_scenes, device=xyz.device)
        xyz, tfs = self.deformer(xyz, mask=(self.face_mask+self.hands_mask+self.outside_mask), cano=False)
        return xyz, tfs

    def extract_pcd(self, code, smpl_params, init=False):
        if isinstance(code, list):
            num_scenes, _, h, w = code[0].size()
        else:
            num_scenes, n_channels, h, w = code.size()
        init_pcd = self.init_pcd.repeat(num_scenes, 1, 1)
        
        sigmas, rgbs, radius, rot, offset = self._decode(code, init=init)
        canon_pcd = init_pcd + offset
        defm_pcd, tfs = self.xyz_transform(canon_pcd, smpl_params, num_scenes)
        
        return defm_pcd, sigmas, rgbs, offset, radius, tfs, rot
    
    def _decode(self, point_code, init=False):
        if isinstance(point_code, list):
            num_scenes, _, h, w = point_code[0].shape
            geo_code, tex_code = point_code
            select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
            if self.multires != 0:
                input_freq = self.input_freq.repeat(num_scenes, 1, 1, 1)
        elif point_code.dim() == 4:
            num_scenes, n_channels, h, w = point_code.shape
            select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
            if self.multires != 0:
                input_freq = self.input_freq.repeat(num_scenes, 1, 1, 1)
            geo_code, tex_code = point_code.split(16, dim=1)
        else:
            assert False

        base_in = geo_code if self.multires == 0 else torch.cat([geo_code, input_freq], dim=1)
        base_x = self.base_net(base_in)
        base_x_act = self.base_activation(self.base_bn(base_x))
        sigma = self.density_net(base_x_act)
        offset = self.offset_net(base_x_act)
        color_in = tex_code if self.multires == 0 else torch.cat([tex_code, input_freq], dim=1)
        rgbs_radius_rot = self.color_net(color_in)
        
        outputs = torch.cat([sigma, offset, rgbs_radius_rot], dim=1)
        output_attr = F.grid_sample(outputs, select_coord, mode=self.interp_mode, padding_mode='border', align_corners=False).reshape(num_scenes, 13, -1).permute(0, 2, 1)
        sigma, offset, rgbs, radius, rot = output_attr.split([1, 3, 3, 3, 3], dim=2)

        if self.sigmoid_saturation > 0:
            rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        
        radius = (radius - 0.5) * 2
        rot = (rot - 0.5) * np.pi

        return sigma, rgbs, radius, rot, offset

    def gaussian_render(self, pcd, sigmas, rgbs, normals, cov3D, num_scenes, num_imgs, cameras, use_scale=False, radius=None, return_norm=False, return_viz=False, mask=None):
        #TODO: add mask or visible points to images or select ind to images
        assert num_scenes == 1
        
        pcd = pcd.reshape(-1, 3)
        if use_scale:
            dist2 = torch.clamp_min(distCUDA2(pcd), 0.0000001)
            scales = torch.sqrt(dist2)[...,None].repeat(1, 3).detach()
            scale = (radius+1)*scales
            cov3D = get_covariance(scale, cov3D).reshape(-1, 6)
       
        images_all = []
        viz_masks = [] if return_viz else None
        norm_all = [] if return_norm else None

        if mask != None:
            pcd = pcd[mask]
            rgbs = rgbs[mask]
            sigmas = sigmas[mask]
            cov3D = cov3D[mask]
            normals = normals[mask]
        if return_norm:
            for i in range(num_imgs):
                self.renderer.prepare(cameras[i])
                image = self.renderer.render_gaussian(means3D=pcd, colors_precomp=rgbs, 
                    rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D)
                images_all.append(image)
                norm = self.renderer.render_gaussian(means3D=pcd, colors_precomp=normals, 
                    rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D)
                norm_all.append(norm)
            norm_all = torch.stack(norm_all, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
        else:
            for i in range(num_imgs):
                self.renderer.prepare(cameras[i])
                image = self.renderer.render_gaussian(means3D=pcd, colors_precomp=rgbs, 
                    rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D)
                images_all.append(image)
                if return_viz:
                    viz_mask = self.renderer.render_gaussian(means3D=pcd, colors_precomp=pcd.clone(), 
                        rotations=None, opacities=sigmas*0+1, scales=None, cov3D_precomp=cov3D)
                    viz_masks.append(viz_mask)
        images_all = torch.stack(images_all, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
        if return_viz:
            viz_masks = torch.stack(viz_masks, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2).reshape(1, -1, 3)
            dist_sq, idx, neighbors = ops.knn_points(pcd.unsqueeze(0), viz_masks[:, ::10], K=1, return_nn=True)
            viz_masks = (dist_sq < 0.0001)[0]

        if use_scale:
            return images_all, norm_all, viz_masks, scale
        else:
            return images_all, norm_all, viz_masks

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        num_scenes, num_chn, h, w = code.size()
        code_viz = code.reshape(num_scenes, 4, 8, h, w).cpu().numpy()
        if not self.flip_z:
            code_viz = code_viz[..., ::-1, :]
        code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 4 * h, 8 * w)
        for code_single, code_viz_single, scene_name_single in zip(code, code_viz, scene_name):
            # torch.save(code_single, os.path.join(viz_dir, 'scene_' + scene_name_single + '.pth'))
            plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '.png'), code_viz_single,
                       vmin=code_range[0], vmax=code_range[1])

    def forward(self, code, grid_size, smpl_params, cameras, num_imgs,
                dt_gamma=0, perturb=False, T_thresh=1e-4, return_loss=False, return_norm=False, init=False, mask=None):
        """
        Args:
            rays_o: Shape (num_scenes, num_rays_per_scene, 3)
            rays_d: Shape (num_scenes, num_rays_per_scene, 3)
            code: Shape (num_scenes, *code_size)
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
        """
        if isinstance(code, list):
            num_scenes = len(code[0])
        else:
            num_scenes = len(code)
        assert num_scenes > 0

        if self.training:
            image = []
            scales = []
            norm = [] if return_norm else None
            xyzs, sigmas, rgbs, offsets, radius, tfs, rot = self.extract_pcd(code, smpl_params, init=init)
            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            normals = (R_def[:, :, -1]).reshape(num_scenes, -1, 3)
            R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
            
            if return_norm:
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, norm_single, _, _ = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single, return_norm=True)
                    image.append(image_single)
                    norm.append(norm_single)
                norm = torch.cat(norm, dim=0)
            else:
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, _, _, scale = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single)
                    image.append(image_single)
                    scales.append(scale.unsqueeze(0))
            
            image = torch.cat(image, dim=0)
            scales = torch.cat(scales, dim=0)
            offset_dist = offsets ** 2
            weighted_offset = torch.mean(offset_dist) + torch.mean(offset_dist[self.hands_mask.repeat(num_scenes, 1)]) + torch.mean(offset_dist[self.face_mask.repeat(num_scenes, 1)])

            results = dict(
                scales=scales,
                norm=norm,
                image=image,
                offset=weighted_offset)
        else:
            if isinstance(code, list):
                device = code[0].device
            else:
                device = code.device
            dtype = torch.float32

            image = []
            offsets = None
            scale = None
            xyzs, sigmas, rgbs, offsets, radius, tfs, rot = self.extract_pcd(code, smpl_params, init=False)
            # longer nose
            # z_min = xyzs[:, self.nose_mask[0], :,-1].min(1)[0]
            # z_min = z_min.unsqueeze(1)
            # xyzs[:, self.nose_mask[0], :,-1] *= 2
            # xyzs[:, self.nose_mask[0], :,-1] -= z_min
            # longer ears
            # y_max = xyzs[:, self.ear_mask[0], :,1].max(1)[0]
            # y_max = y_max.unsqueeze(1)
            # xyzs[:, self.ear_mask[0], :,1] *= 1.5 
            # xyzs[:, self.ear_mask[0], :,1] -= (y_max * 0.5)

            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            normals = (R_def[:, :, -1]).reshape(num_scenes, -1, 3)
            R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
            if mask != None:
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single, mask_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius, mask):
                    image_single, _, viz_masks, _ = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single, mask=mask_single)
                    image.append(image_single)
            else:
                # viz_masks = []
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, _, viz_mask, _ = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single, return_viz=False)
                    image.append(image_single)
                    # viz_masks.append(viz_mask)
                # viz_masks = torch.cat(viz_masks, dim=0)
                viz_masks = None
            image = torch.cat(image, dim=0)

            results = dict(
                viz_masks=viz_masks,
                scales=None,
                image=image,
                offset=offsets)

        if return_loss:
            results.update(decoder_reg_loss=self.loss())

        return results