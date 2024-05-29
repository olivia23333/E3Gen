import numpy as np
import torch
import torch.nn.functional as F
from smplx import SMPLX
import cv2


def load_obj(path):
    """Load wavefront OBJ from file."""
    v = []
    vt = []
    vindices = []
    vtindices = []

    with open(path, "r") as f:
        while True:
            line = f.readline()

            if line == "":
                break

            if line[:2] == "v ":
                v.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "vt":
                vt.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "f ":
                vindices.append([int(entry.split('/')[0]) - 1 for entry in line.split()[1:]])
                if line.find("/") != -1:
                    vtindices.append([int(entry.split('/')[1]) - 1 for entry in line.split()[1:]])

    return v, vt, vindices, vtindices

def gentritex(v, vt, vi, vti, texsize):
    """Create 3 texture maps containing the vertex indices, texture vertex
    indices, and barycentric coordinates"""
    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)
    ntris = vi.shape[0]

    texu, texv = np.meshgrid(
            (np.arange(texsize) + 0.5) / texsize,
            (np.arange(texsize) + 0.5) / texsize)
    texuv = np.stack((texu, texv), axis=-1)

    vt = vt[vti]

    viim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    vtiim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    baryim = np.zeros((texsize, texsize, 3), dtype=np.float32)

    for i in list(range(ntris))[::-1]:
        bbox = (
            max(0, int(min(vt[i, 0, 0], min(vt[i, 1, 0], vt[i, 2, 0]))*texsize)-1),
            min(texsize, int(max(vt[i, 0, 0], max(vt[i, 1, 0], vt[i, 2, 0]))*texsize)+2),
            max(0, int(min(vt[i, 0, 1], min(vt[i, 1, 1], vt[i, 2, 1]))*texsize)-1),
            min(texsize, int(max(vt[i, 0, 1], max(vt[i, 1, 1], vt[i, 2, 1]))*texsize)+2))
        v0 = vt[None, None, i, 1, :] - vt[None, None, i, 0, :]
        v1 = vt[None, None, i, 2, :] - vt[None, None, i, 0, :]
        v2 = texuv[bbox[2]:bbox[3], bbox[0]:bbox[1], :] - vt[None, None, i, 0, :]
        d00 = np.sum(v0 * v0, axis=-1)
        d01 = np.sum(v0 * v1, axis=-1)
        d11 = np.sum(v1 * v1, axis=-1)
        d20 = np.sum(v2 * v0, axis=-1)
        d21 = np.sum(v2 * v1, axis=-1)
        denom = d00 * d11 - d01 * d01

        if denom != 0.:
            baryv = (d11 * d20 - d01 * d21) / denom
            baryw = (d00 * d21 - d01 * d20) / denom
            baryu = 1. - baryv - baryw

            baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((baryu, baryv, baryw), axis=-1),
                    baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((vi[i, 0], vi[i, 1], vi[i, 2]), axis=-1),
                    viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((vti[i, 0], vti[i, 1], vti[i, 2]), axis=-1),
                    vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])

    return viim, vtiim, baryim

@torch.jit.script
def compute_tbn(v0, v1, v2, vt0, vt1, vt2):
    v01 = v1 - v0
    v02 = v2 - v0
    vt01 = vt1 - vt0
    vt02 = vt2 - vt0

    f = 1. / ((vt01[None, :, :, 0] * vt02[None, :, :, 1] - vt01[None, :, :, 1] * vt02[None, :, :, 0]) + 1e-12)
    mask = torch.isinf(f)
    
    tangent = f[:, :, :, None] * torch.stack([
        v01[:, :, :, 0] * vt02[None, :, :, 1] - v02[:, :, :, 0] * vt01[None, :, :, 1],
        v01[:, :, :, 1] * vt02[None, :, :, 1] - v02[:, :, :, 1] * vt01[None, :, :, 1],
        v01[:, :, :, 2] * vt02[None, :, :, 1] - v02[:, :, :, 2] * vt01[None, :, :, 1]], dim=-1)

    tangent = F.normalize(tangent, dim=-1)
    normal = torch.cross(v01, v02, dim=3)
    normal = F.normalize(normal, dim=-1)
    bitangent = torch.cross(tangent, normal, dim=3) # I guess normal, tangent would be right
    bitangent = F.normalize(bitangent, dim=-1)

    # create matrix
    primrotmesh = torch.stack((tangent, bitangent, normal), dim=-1)

    return primrotmesh


if __name__ == '__main__':
    # body_model = SMPLX('/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/smplx/SMPLX', gender='male', \
    #                             use_pca=True,
    #                             num_pca_comps=12,
    #                             num_betas=10,
    #                             flat_hand_mean=True)
    body_model = SMPLX('smplx/SMPLX', gender='male', \
                                use_pca=True,
                                num_pca_comps=12,
                                num_betas=10,
                                flat_hand_mean=False)
    body_pose_t = torch.zeros((1, 63))
    # jaw_pose_t = torch.as_tensor([[0.2, 0, 0],])
    jaw_pose_t = torch.as_tensor([[0, 0, 0],])
    # transl = torch.as_tensor([[-0.0012,  0.4668, -0.0127],])
    transl = torch.as_tensor([[0., 0.35, 0.],])
    left_hand_pose_t = torch.tensor([[1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
         -0.6652, -0.7290,  0.0084, -0.4818],])
    right_hand_pose_t = torch.tensor([[1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
         -0.6652, -0.7290,  0.0084, -0.4818],])
    global_orient = torch.zeros((1, 3))
    betas = torch.zeros((1, 10))
    smpl_outputs = body_model(betas=betas, body_pose=body_pose_t, transl=transl, global_orient = global_orient, jaw_pose=jaw_pose_t, left_hand_pose=left_hand_pose_t, right_hand_pose=right_hand_pose_t)
    vs_template = smpl_outputs.vertices.detach()

    objpath = "../../../work_dirs/cache/template/dense_thuman.obj"
    v, vt, vi, vti = load_obj(objpath)
    v = np.array(v, dtype=np.float32)
    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32) # np.int_ means long
    vti = np.array(vti, dtype=np.int32)
    
    idxim, tidxim, barim = gentritex(v, vt, vi, vti, 256)
    idxim = torch.tensor(idxim).long()
    tidxim = torch.tensor(tidxim).long()
    barim = torch.tensor(barim)
    
    uvheight, uvwidth = barim.size(0), barim.size(1)
    stridey = uvheight // uvheight
    stridex = uvwidth // uvwidth

    # v = vs_template
    v = torch.as_tensor(v).unsqueeze(0)
    vt = torch.as_tensor(vt)
    vi = torch.tensor(vi).long()
    vti = torch.tensor(vti).long()
    
    v0 = v[:, idxim[stridey//2::stridey, stridex//2::stridex, 0], :]
    v1 = v[:, idxim[stridey//2::stridey, stridex//2::stridex, 1], :]
    v2 = v[:, idxim[stridey//2::stridey, stridex//2::stridex, 2], :]

    vt0 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 0], :]
    vt1 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 1], :]
    vt2 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 2], :]

    primrotmesh = compute_tbn(v0, v1, v2, vt0, vt1, vt2).view(v0.size(0), -1, 3, 3)

    primposmesh = (
                    barim[None, stridey//2::stridey, stridex//2::stridex, 0, None] * v0 +
                    barim[None, stridey//2::stridey, stridex//2::stridex, 1, None] * v1 +
                    barim[None, stridey//2::stridey, stridex//2::stridex, 2, None] * v2
                    ).view(v0.size(0), 256, 256, 3)
    
    primuv = (
                    barim[None, stridey//2::stridey, stridex//2::stridex, 0, None] * vt0 +
                    barim[None, stridey//2::stridey, stridex//2::stridex, 1, None] * vt1 +
                    barim[None, stridey//2::stridey, stridex//2::stridex, 2, None] * vt2
                    ).view(v0.size(0), 256, 256, 2)
    uv_pos = torch.as_tensor(np.load('../../../work_dirs/cache/init_uv_smplx_thu.npy')).unsqueeze(0)
    uv_pos = uv_pos * 2 - 1 
    coord_pos = torch.as_tensor(np.load('../../../work_dirs/cache/init_pcd_smplx_thu.npy'))
    primpos_map = primposmesh.permute(0, 3, 1, 2)
    primuv_map = primuv.permute(0, 3, 1, 2)
    np.save('../../../work_dirs/cache/init_posmap_smplx_thu.npy', primpos_map.numpy())
    np.save('../../../work_dirs/cache/init_uvmap_smplx_thu.npy', primuv_map.numpy())