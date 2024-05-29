from diff_gaussian_rasterization_depth import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
import torch
import torch.nn as nn
import cv2
# from pytorch3d.transforms import quaternion_to_matrix

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, w2c):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1, 3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ w2c
    xyz_world = xyz_world[..., :3]

    return xyz_world

def depth_pcd2normal(xyz):
    hd, wd, _ = xyz.shape 
    bottom_point = xyz[..., 2:hd,   1:wd-1, :]
    top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
    right_point  = xyz[..., 1:hd-1, 2:wd,   :]
    left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = (right_point - left_point)
    bottom_to_top = (top_point - bottom_point)
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant')
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, w2c):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(depth, intrinsic_matrix, w2c) # (HxW, 3)
    xyz_world = xyz_world.reshape(*depth.shape, 3)

    xyz_normal = depth_pcd2normal(xyz_world)

    return xyz_normal

def batch_rodrigues(rot_vecs, epsilon = 1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def build_scaling_rotation(s, r, tfs):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)
    R_ = R

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R_ @ L
    return L

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, tfs):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation, tfs)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=q.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def get_covariance(scaling, rotation, scaling_modifier = 1):
    L = torch.zeros_like(rotation)
    L[:, 0, 0] = scaling[:, 0]
    L[:, 1, 1] = scaling[:, 1]
    L[:, 2, 2] = scaling[:, 2]
    actual_covariance = rotation @ (L**2) @ rotation.permute(0, 2, 1)
    return strip_symmetric(actual_covariance)

class GDRenderer(nn.Module):
    def __init__(self, image_size=256, anti_alias=False, f=5000, near=0.01, far=40, bg_color=0):
        super().__init__()

        self.anti_alias = anti_alias
        self.image_size = image_size
        self.tanfov = self.image_size / (2 * f)

        if bg_color == 0:
            bg = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)
        else:
            bg = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)

        self.register_buffer('bg', bg)
        
        opengl_proj = torch.tensor([[2 * f / self.image_size, 0.0, 0.0, 0.0],
                                    [0.0, 2 * f / self.image_size, 0.0, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).float().unsqueeze(0).transpose(1, 2)
        self.register_buffer('opengl_proj', opengl_proj)

        intrinsic_matrix = torch.tensor([[f, 0, self.image_size / 2], [0, f, self.image_size / 2], [0, 0, 1]]).float()
        self.register_buffer('intrinsic_matrix', intrinsic_matrix)

        if anti_alias: image_size = image_size*2
        
    def prepare(self, cameras):
        cam_center = cameras[:3]
        w2c = cameras[3:].reshape(4, 4)
        w2c = w2c.unsqueeze(0).transpose(1, 2)
        self.w2c = w2c
        full_proj = w2c.bmm(self.opengl_proj)

        self.raster_settings = GaussianRasterizationSettings(
            image_height=self.image_size,
            image_width=self.image_size,
            tanfovx=self.tanfov,
            tanfovy=self.tanfov,
            bg=self.bg,
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=cam_center,
            prefiltered=False,
            debug=False
        )
        self.rasterizer = GaussianRasterizer(raster_settings=self.raster_settings)

    def render_gaussian(self, means3D, colors_precomp, rotations, opacities, scales, cov3D_precomp=None):
        '''
        mode: normal, phong, texture
        '''
        screenspace_points = (
            torch.zeros_like(
                means3D,
                dtype=means3D.dtype,
                requires_grad=True,
                device=means3D.device,
            )
            + 0
        )

        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        if cov3D_precomp != None:
            image, _, depth, alpha = self.rasterizer(means3D=means3D, colors_precomp=colors_precomp, \
                opacities=opacities, means2D=screenspace_points, cov3D_precomp=cov3D_precomp)
            normal_map = normal_from_depth_image(depth[0], self.intrinsic_matrix, self.w2c[0])
            normal_map = normal_map.permute(1, 2, 0)
        else:
            image, _, depth, alpha = self.rasterizer(means3D=means3D, colors_precomp=colors_precomp, \
                rotations=torch.nn.functional.normalize(rotations), opacities=opacities, scales=scales, \
                means2D=screenspace_points)
            normal_map = normal_from_depth_image(depth[0], self.intrinsic_matrix, self.w2c[0])
            normal_map = normal_map.permute(1, 2, 0)
            
        return  image, normal_map, alpha