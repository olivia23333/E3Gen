import torch
from torch import einsum
import torch.nn.functional as F
import os

from torch.utils.cpp_extension import load

import fuse_cuda 
import filter_cuda
import precompute_cuda
import numpy as np


class ForwardDeformer(torch.nn.Module):
    """
    Tensor shape abbreviation:
        B: batch size
        N: number of points
        J: number of bones
        I: number of init
        D: space dimension
    """

    def __init__(self,  **kwargs):
        super().__init__()

        self.soft_blend = 20

        self.init_bones = [0, 1, 2, 4, 5, 12, 15, 16, 17, 18, 19]
        
        self.init_bones_cuda = torch.tensor(self.init_bones).int()
        
        self.global_scale = 1.2

    def forward_skinning(self, xc, shape_offset, pose_offset, cond, tfs, tfs_inv, poseoff_ori, lbsw=None, mask=None):
        """Canonical point -> deformed point

        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """

        w = self.query_weights(xc, cond)
        w[:, mask[0]] = lbsw[mask]
        
        b,n,_ = xc.shape
        xc_cano, w_tf_inv = skinning(xc, w, tfs_inv.expand(b, -1, -1, -1), inverse=False)
        xc_cano_ori = xc_cano - poseoff_ori.expand(b, -1, -1)

        xc_shape = xc_cano_ori + shape_offset + pose_offset
        xd, w_tf = skinning(xc_shape, w, tfs, inverse=False)
        w_tf_all = w_tf @ w_tf_inv.expand(b, -1, -1, -1)
        
        return xd, w_tf_all

    def switch_to_explicit(self,resolution=32,smpl_verts=None, smpl_faces=None, smpl_weights=None, use_smpl=False):
        
        self.resolution = resolution
        # convert to voxel grid
    
        b, c, d, h, w = 1, 55, resolution//4, resolution, resolution
        
        self.ratio = h/d
        grid = create_voxel_grid(d, h, w)
        device = grid.device

        gt_bbox = torch.cat([smpl_verts.min(dim=1).values, smpl_verts.max(dim=1).values], dim=0).to(device)
        
        offset = (gt_bbox[0] + gt_bbox[1])[None,None,:] * 0.5
        scale = (gt_bbox[1] - gt_bbox[0]).max()/2 * self.global_scale

        self.register_buffer('scale', scale)
        self.register_buffer('offset', offset)

        self.register_buffer('offset_kernel', -self.offset)
        scale_kernel = torch.zeros_like(self.offset)
        scale_kernel[...] = 1./self.scale
        scale_kernel[:,:,-1] = scale_kernel[:,:,-1] * self.ratio
        self.register_buffer('scale_kernel', scale_kernel)
        
        def normalize(x):
            x_normalized = (x+self.offset_kernel)*self.scale_kernel
            return x_normalized

        def denormalize(x):
            x_denormalized = x.clone() #/self.global_scale
            x_denormalized[..., -1] = x_denormalized[..., -1]/self.ratio
            x_denormalized *= self.scale
            x_denormalized += self.offset

            return x_denormalized

        self.normalize = normalize
        self.denormalize = denormalize

        grid_denorm = self.denormalize(grid)

        weights = query_weights_smpl(grid_denorm, smpl_verts=smpl_verts.detach().clone(), smpl_weights=smpl_weights.detach().clone()).detach().clone()

        self.register_buffer('lbs_voxel_final', weights.detach())
        self.register_buffer('grid_denorm',grid_denorm)

        def query_weights( xc, cond=None, mask=None, mode='bilinear'):
            w = F.grid_sample(self.lbs_voxel_final.expand(xc.shape[0],-1,-1,-1,-1), self.normalize(xc).unsqueeze(2).unsqueeze(2),align_corners=True, mode=mode,padding_mode='border')
            w = w.squeeze(-1).squeeze(-1).permute(0,2,1)
            return w
    
        self.query_weights = query_weights

    def update_lbs_voxel(self):
        self.lbs_voxel_final = F.softmax( self.lbs_voxel*20,dim=1)
        def query_weights( xc, cond=None, mask=None, mode='bilinear'):
            w = F.grid_sample(self.lbs_voxel_final.expand(xc.shape[0],-1,-1,-1,-1), self.normalize(xc).unsqueeze(2).unsqueeze(2),align_corners=True, mode=mode,padding_mode='border')
            w = w.squeeze(-1).squeeze(-1).permute(0,2,1)
            return w

        self.query_weights = query_weights


    def query_sdf_smpl(self, x, smpl_verts, smpl_faces, smpl_weights):
        
        device = x.device

        resolution=128
        b, c, d, h, w = 1, 24, resolution//4, resolution, resolution
        grid = create_voxel_grid(d, h, w, device)
        grid = self.denormalize(grid)

        import trimesh
        mesh = trimesh.Trimesh(vertices=smpl_verts.data.cpu().numpy()[0], faces=smpl_faces.data.cpu().numpy())
        BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)
    
        sdf, face_id, uvw = BVH.signed_distance(grid, return_uvw=True, mode='watertight') # [N], [N], [N, 3]

        sdf = sdf.reshape(1, -1, 1)
        b, c, d, h, w = 1, 1, resolution//4, resolution, resolution

        sdf = -sdf.permute(0,2,1).reshape(b,c,d,h,w)

        return sdf.detach()

    def skinning_normal(self, xc, normal, tfs, cond=None, mask=None, inverse=False):
        ''' skinning normals
        
        Args:
            x (tensor): canonical points. shape: [B, N, D]
            normal (tensor): canonical normals. shape: [B, N, D]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            posed normal (tensor): posed normals. shape: [B, N, D]
            
        '''
        if xc.ndim == 2:
            xc = xc.unsqueeze(0)
        if normal.ndim == 2:
            normal = normal.unsqueeze(0)
        w = self.query_weights(xc, cond, mask=mask)
        p_h = F.pad(normal, (0, 1), value=0)
        p_h = torch.einsum('bpn, bnij, bpj->bpi', w, tfs, p_h)

        return p_h[:, :, :3]
    
def skinning_mask(x, w, tfs, inverse=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)
    p,n = w.shape

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1

        w_tf = einsum("bpn,bnij->bpij", w, tfs)

        x_h = x_h.view(b,p,1,4).expand(b,p,4,4)
        x_h = (fast_inverse(w_tf)*x_h).sum(-1)

    else:
        w_tf = einsum("pn,nij->pij", w, tfs.squeeze(0))

        x_h = x_h.view(p,1,4).expand(p,4,4)
        x_h = (w_tf*x_h).sum(-1)

    return x_h[:, :3], w_tf

def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)
    b,p,n = w.shape

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = einsum("bpn,bnij->bpij", w, fast_inverse(tfs))

        x_h = x_h.view(b,p,1,4).expand(b,p,4,4)
        # x_h = (fast_inverse(w_tf)*x_h).sum(-1)
        x_h = (w_tf*x_h).sum(-1)

    else:
        w_tf = einsum("bpn,bnij->bpij", w, tfs)

        x_h = x_h.view(b,p,1,4).expand(b,p,4,4)
        x_h = (w_tf*x_h).sum(-1)

    return x_h[:, :, :3], w_tf

def fast_inverse(T):

    shape = T.shape

    T = T.reshape(-1,4,4)
    R = T[:, :3,:3]
    t = T[:, :3,3].unsqueeze(-1)

    R_inv = R.transpose(1,2)
    t_inv = -bmv(R_inv,t)

    T_inv = T
    T_inv[:,:3,:3] = R_inv
    T_inv[:,:3,3] = t_inv.squeeze(-1)
    
    return T_inv.reshape(shape)

def bmv(m, v):
    return (m*v.transpose(-1,-2).expand(-1,3,-1)).sum(-1,keepdim=True)


def create_voxel_grid(d, h, w, device='cuda'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid


def query_weights_smpl(x, smpl_verts, smpl_weights):
    import pytorch3d.ops as ops

    device = smpl_weights.device
    distance_batch, index_batch, neighbor_points  = ops.knn_points(x.to(device),smpl_verts.to(device).detach(),K=10,return_nn=True)

    # neighbor_points = neighbor_points[0]
    distance_batch = distance_batch[0].sqrt().clamp_(0.00003,0.1)
    index_batch = index_batch[0]
    
    # GPU_id = index_batch.get_device()
    # print(GPU_id)
    weights = smpl_weights[0,index_batch]
   
    ws=1./distance_batch
    ws=ws/ws.sum(-1,keepdim=True)
    weights = (ws[:,:,None]*weights).sum(1)[None]

    resolution = 64

    b, c, d, h, w = 1, 55, resolution//4, resolution, resolution
    weights = weights.permute(0,2,1).reshape(b,c,d,h,w)

    return weights.detach()#, blendshapes.detach()