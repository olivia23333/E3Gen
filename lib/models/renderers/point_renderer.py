import cv2
import torch
import numpy as np
import math
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    RasterizationSettings,
    PointsRasterizationSettings,
    MeshRenderer,
    PointsRenderer,
    MeshRasterizer,
    PointsRasterizer,
    HardPhongShader,
    SoftPhongShader,
    AlphaCompositor,
    PointLights
)
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer.mesh import Textures

class Renderer():
    def __init__(self, image_size=256, anti_alias=False):
        super().__init__()

        self.anti_alias = anti_alias

        self.image_size = image_size

        if anti_alias: image_size = image_size*2

        self.raster_settings = PointsRasterizationSettings(image_size=image_size,
            # radius=0.3 * (0.75 ** math.log2(600000 / 100)),
            # radius=0.003,
            radius=0.005,
            points_per_pixel=10
        )
        self.compositor = AlphaCompositor(background_color=[0, 0, 0, 0])

    def prepare(self, cameras):
        self.rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings)

    def render_point_dict(self, points):
        '''
        mode: normal, phong, texture
        '''
        fragments = self.rasterizer(points) #count, dists, idx, index, zbuf
        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        render_index = fragments.idx.long().flatten()
        mask = render_index == -1
        render_index[mask] = 0
        sigmas = torch.gather(points.features_packed()[:,-1], 0, render_index)
        sigmas[mask] = 0
        sigmas = sigmas.reshape(fragments.idx.shape).permute(0, 3, 1, 2)
        alphas = (1 - dists2 / (r * r)) * sigmas
        
        rgb_sigma, _ = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            points.features_packed().permute(1, 0),
        )   
        
        image_color, weights = rgb_sigma.split([3, 1], dim=1)
            
        return  image_color, weights