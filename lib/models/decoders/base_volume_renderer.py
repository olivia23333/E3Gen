import torch
import torch.nn as nn
import numpy as np

from mmgen.models.builder import build_module


class VolumeRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 min_near=0.2,
                 bg_radius=-1,
                 max_steps=256,
                 decoder_reg_loss=None,
                 ):
        super().__init__()

        self.bound = bound
        self.min_near = min_near
        self.bg_radius = bg_radius  # radius of the background sphere.
        self.max_steps = max_steps
        self.decoder_reg_loss = build_module(decoder_reg_loss) if decoder_reg_loss is not None else None

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        self.register_buffer('aabb', aabb)

    def extract_pcd(self, num_scenes, code):
        raise NotImplementedError
        
    def point_decode(self, xyzs, dirs, code):
        raise NotImplementedError

    def point_density_decode(self, xyzs, code):
        raise NotImplementedError

    def loss(self):
        assert self.decoder_reg_loss is None
        return None
    
    def point_render(self, pcd, mask, rgbs, num_scenes, num_points, cameras):
        raise NotImplementedError
    
    def gaussian_render(self, pcd, mask, rgbs, num_scenes, num_points, cameras):
        raise NotImplementedError

    def forward(self, code, grid_size, smpl_params, cameras, num_imgs,
                points=None, masks=None, dt_gamma=0, perturb=False, T_thresh=1e-4, return_loss=False, stage2=True, init=False):
        """
        Args:
            rays_o: Shape (num_scenes, num_rays_per_scene, 3)
            rays_d: Shape (num_scenes, num_rays_per_scene, 3)
            code: Shape (num_scenes, *code_size)
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
        """
        num_scenes = len(code)
        assert num_scenes > 0
        if isinstance(grid_size, int):
            grid_size = [grid_size] * num_scenes
        if isinstance(dt_gamma, float):
            dt_gamma = [dt_gamma] * num_scenes

        if self.training:
            image = []
            offset = []
            scales = []
            
            for smpl_param_single, camera_single, code_single, point_single, mask_single in zip(smpl_params, cameras, code, points, masks):
                xyzs, sigmas, rgbs, num_points, offset_single, radius, tfs, rot = self.extract_pcd(1, code_single[None], smpl_param_single[None], point_single[None], mask_single[None], stage2=stage2, init=init)
                image_single, scale_single = self.gaussian_render(xyzs, sigmas, rgbs, radius, tfs, rot, 1, num_imgs, num_points, camera_single, mask_single, stage2=stage2)
                image.append(image_single)
                offset.append(offset_single)
                part_mask.append(self.part_mask[mask_single])
                scales.append(scale_single)
            
            image = torch.cat(image, dim=0)
            offset = torch.cat(offset, dim=0)
            scales = torch.cat(scales, dim=0)
            part_mask = torch.cat(part_mask, dim=0)

        else:
            device = code.device
            dtype = torch.float32

            image = []
            offset = None
            scales = None
            part_mask = None

            for smpl_param_single, camera_single, code_single, point_single, mask_single in zip(smpl_params, cameras, code, points, masks):
                xyzs, sigmas, rgbs, num_points, _, radius, tfs, rot = self.extract_pcd(1, code_single[None], smpl_param_single[None], point_single[None], mask_single[None], stage2=stage2)
                image_single, _ = self.gaussian_render(xyzs, sigmas, rgbs, radius, tfs, rot, 1, num_imgs, num_points, camera_single, mask_single)
                image.append(image_single)

        results = dict(
            part_mask=part_mask,
            scales=scales,
            image=image,
            offset=offset)

        if return_loss:
            results.update(decoder_reg_loss=self.loss())

        return results
