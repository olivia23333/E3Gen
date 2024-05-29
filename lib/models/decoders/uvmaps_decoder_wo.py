import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init, constant_init
from mmgen.models.builder import MODULES
import numpy as np
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
class UVODecoder(VolumeRenderer):

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
                #  shape_layers=[3 * 32, 128],
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
                #  sigma_activation='trunc_exp',
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
        self.superres = None

        select_uv = torch.as_tensor(np.load('work_dirs/cache/init_uv_smplx_thu.npy'))
        self.register_buffer('select_coord', select_uv.unsqueeze(0)*2.-1.)

        init_pcd = torch.as_tensor(np.load('work_dirs/cache/init_pcd_smplx_thu.npy'))
        self.register_buffer('init_pcd', init_pcd.unsqueeze(0), persistent=False)
        self.num_init = self.init_pcd.shape[1]

        dist2 = torch.clamp_min(distCUDA2(init_pcd.cuda()), 0.0000001)
        scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
        self.register_buffer('scales', scales)

        init_rot = torch.as_tensor(np.load('work_dirs/cache/init_rot_smplx_thu.npy'))
        self.register_buffer('init_rot', init_rot, persistent=False)

        face_mask = torch.as_tensor(np.load('work_dirs/cache/face_mask_thu.npy'))
        self.register_buffer('face_mask', face_mask.unsqueeze(0), persistent=False)

        hands_mask = torch.as_tensor(np.load('work_dirs/cache/hands_mask_thu.npy'))
        self.register_buffer('hands_mask', hands_mask.unsqueeze(0), persistent=False)

        outside_mask = torch.as_tensor(np.load('work_dirs/cache/outside_mask_thu.npy'))
        self.register_buffer('outside_mask', outside_mask.unsqueeze(0), persistent=False)

    def normalize(self, value):
        "normalize the value into [-1, 1]"
        value_min, _ = value.min(0, keepdim=True)
        value_max, _ = value.max(0, keepdim=True)
        value_nor = ((value - value_min) / (value_max - value_min)) * 2 - 1
        return value_nor

    def xyz_transform(self, xyz, smpl_params=None, num_scenes=1):
        assert xyz.dim() == 3
        self.deformer.prepare_deformer(smpl_params, num_scenes, device=xyz.device)
        xyz, tfs = self.deformer(xyz, mask=(self.face_mask+self.hands_mask+self.outside_mask), cano=False)
        return xyz, tfs

    def extract_pcd(self, code, smpl_params, init=False):
        num_scenes, n_channels, h, w = code.size()
        init_pcd = self.init_pcd.repeat(num_scenes, 1, 1)
        
        sigmas, rgbs, radius, rot, offset = self._decode(code, init=init)
        canon_pcd = init_pcd + offset
        defm_pcd, tfs = self.xyz_transform(canon_pcd, smpl_params, num_scenes)
        
        return defm_pcd, sigmas, rgbs, offset, radius, tfs, rot
    
    def _decode(self, point_code, init=False):
        if point_code.dim() == 4:
            num_scenes, n_channels, h, w = point_code.shape
            select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
        else:
            assert False
        
        code_in = F.grid_sample(point_code, select_coord, mode=self.interp_mode, padding_mode='border', align_corners=False).reshape(num_scenes, n_channels, -1)
        code_gaussian = code_in.reshape(num_scenes, n_channels, -1).permute(0, 2, 1)
        offset, sigma, rgbs, radius, rot = code_gaussian.split([3, 1, 3, 3, 3], dim=2)


        offset = offset * 0.05
        rot = rot * np.pi/2
        sigma = (sigma + 1) * 0.5
        rgbs = (rgbs + 1) * 0.5

        return sigma, rgbs, radius, rot, offset

    def gaussian_render(self, pcd, sigmas, rgbs, normals, cov3D, num_scenes, num_imgs, cameras, use_scale=False, radius=None, return_norm=False):
        #TODO: add mask or visible points to images or select ind to images
        assert num_scenes == 1
        
        pcd = pcd.reshape(-1, 3)
        # if use_scale:
        #     scales = self.scales.repeat(num_scenes, 1)
        images_all = []
        norm_all = [] if return_norm else None

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
                image = self.renderer.render_gaussian(means3D=pcd, colors_precomp=normals, 
                    rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D)
                images_all.append(image)
           
        images_all = torch.stack(images_all, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
        
        return images_all, norm_all

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        num_scenes, num_chn, h, w = code.size()
        code_viz = code[:, 1:].reshape(num_scenes, 3, 4, h, w).cpu().numpy()
        code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 3 * h, 4 * w)
        for code_viz_single, scene_name_single in zip(code_viz, scene_name):
            plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '.png'), code_viz_single,
                       vmin=code_range[0], vmax=code_range[1])

    def forward(self, code, grid_size, smpl_params, cameras, num_imgs,
                dt_gamma=0, perturb=False, T_thresh=1e-4, return_loss=False, return_norm=False, init=False):
        """
        Args:
            rays_o: Shape (num_scenes, num_rays_per_scene, 3)
            rays_d: Shape (num_scenes, num_rays_per_scene, 3)
            code: Shape (num_scenes, *code_size)
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
        """
        num_scenes = len(code)
        assert num_scenes > 0

        if self.training:
            image = []
            norm = [] if return_norm else None
            xyzs, sigmas, rgbs, offsets, radius, tfs, rot = self.extract_pcd(code, smpl_params, init=init)
            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            normals = (R_def[:, :, -1] * 0.5 + 0.5).reshape(num_scenes, -1, 3)
            R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
            if return_norm:
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, norm_single = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single, return_norm=True)
                    image.append(image_single)
                    norm.append(norm_single)
                norm = torch.cat(norm, dim=0)
            else:
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, _ = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single)
                    image.append(image_single)
            
            image = torch.cat(image, dim=0)
            offset_dist = offsets ** 2
            weighted_offset = torch.mean(offset_dist) + torch.mean(offset_dist[self.hands_mask.repeat(num_scenes, 1)]) + torch.mean(offset_dist[self.face_mask.repeat(num_scenes, 1)])

            results = dict(
                norm=norm,
                image=image,
                offset=weighted_offset)
        else:
            device = code.device
            dtype = torch.float32

            image = []
            offsets = None
            scale = None

            xyzs, sigmas, rgbs, offsets, radius, tfs, rot = self.extract_pcd(code, smpl_params, init=False)
            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            normals = (R_def[:, :, -1] * 0.5 + 0.5).reshape(num_scenes, -1, 3)
            R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
            for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                image_single, _ = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single)
                image.append(image_single)

            image = torch.cat(image, dim=0)

            results = dict(
                image=image,
                offset=offsets)

        if return_loss:
            results.update(decoder_reg_loss=self.loss())

        return results
