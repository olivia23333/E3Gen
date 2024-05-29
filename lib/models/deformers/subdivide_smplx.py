import numpy as np
import json
import os
import pickle
import xatlas
import torch
import trimesh
from smplx import SMPLX


def subdivide(vertices, faces, attributes=None, face_index=None):
    """
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those faces will
    be subdivided and their neighbors won't be modified making
    the mesh no longer "watertight."

    Parameters
    ----------
    vertices : (n, 3) float
      Vertices in space
    faces : (n, 3) int
      Indexes of vertices which make up triangular faces
    attributes: (n, d) float
      vertices attributes
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces

    Returns
    ----------
    new_vertices : (n, 3) float
      Vertices in space
    new_faces : (n, 3) int
      Remeshed faces
    """
    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c,3) int set of vertex indices
    faces = faces[face_index]

    # the (c, 3, 3) float set of points in the triangles
    triangles = vertices[faces]
    
    # the 3 midpoints of each triangle edge
    # stacked to a (3 * c, 3) float
    mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]])

    # for adjacent faces we are going to be generating
    # the same midpoint twice so merge them here
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
    unique, inverse = trimesh.grouping.unique_rows(mid)
    mid = mid[unique]
    mid_idx = inverse[mid_idx] + len(vertices)

    # the new faces with correct winding
    f = np.column_stack([faces[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))
    # add the 3 new faces per old face
    new_faces = np.vstack((faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((vertices, mid))

    if attributes is not None:
        tri_att = attributes[faces]
        mid_att = np.vstack([tri_att[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]])
        mid_att = mid_att[unique]
        new_attributes = np.vstack((attributes, mid_att))
        return new_vertices, new_faces, new_attributes, unique

    return new_vertices, new_faces, unique

def write_obj(obj_name,
              vertices,
              faces,
              uvcoords=None,
              uvfaces=None,
              ):
    ''' Save 3D face model with texture. 
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    '''
    if os.path.splitext(obj_name)[-1] != '.obj':
        obj_name = obj_name + '.obj'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices
        for i in range(vertices.shape[0]):
            f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))

        # write uv coords
        for i in range(uvcoords.shape[0]):
            f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
       
        # write f: ver ind/ uv ind
        uvfaces = uvfaces + 1
        for i in range(faces.shape[0]):
            f.write('f {}/{} {}/{} {}/{}\n'.format(
                faces[i, 0], uvfaces[i, 0],
                faces[i, 1], uvfaces[i, 1],
                faces[i, 2], uvfaces[i, 2]
            )
            )


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


def get_seg_mask(smplx_face, other_ids=None):
    "following https://github.com/TingtingLiao/TADA/blob/main/lib/common/utils.py"

    smplx_segs = json.load(open(f'/home/zhangweitian/HighResAvatar/work_dirs/cache/template/smplx_vert_segmentation.json'))
    flame_segs = pickle.load(open(f'/home/zhangweitian/HighResAvatar/work_dirs/cache/template/FLAME_masks.pkl', 'rb'), encoding='latin1')
    smplx_flame_vid = np.load(f"/home/zhangweitian/HighResAvatar/work_dirs/cache/template/SMPL-X__FLAME_vertex_ids.npy", allow_pickle=True)

    eyeball_ids = smplx_segs["leftEye"] + smplx_segs["rightEye"]
    hands_ids = smplx_segs["leftHand"] + smplx_segs["rightHand"] + \
                smplx_segs["leftHandIndex1"] + smplx_segs["rightHandIndex1"]
    # neck_ids = smplx_segs["neck"]
    # head_ids = smplx_segs["head"]
    feet_ids = smplx_segs['leftFoot'] + smplx_segs['rightFoot'] + smplx_segs['leftToeBase'] + smplx_segs['rightToeBase'] 

    front_face_ids = list(smplx_flame_vid[flame_segs["face"]])
    ears_ids = list(smplx_flame_vid[flame_segs["left_ear"]]) + list(smplx_flame_vid[flame_segs["right_ear"]])
    forehead_ids = list(smplx_flame_vid[flame_segs["forehead"]])

    label_flame = []
    for key in flame_segs:
        label_flame += list(smplx_flame_vid[flame_segs[key]])
    outside_ids = list(set(smplx_flame_vid) - set(label_flame))
    # mouth_inner
    lips_ids = list(smplx_flame_vid[flame_segs["lips"]])
    nose_ids = list(smplx_flame_vid[flame_segs["nose"]])
    # scalp_ids = list(smplx_flame_vid[flame_segs["scalp"]])
    # boundary_ids = list(smplx_flame_vid[flame_segs["boundary"]])
    # neck_ids = list(smplx_flame_vid[flame_segs["neck"]])
    # eyes_ids = list(smplx_flame_vid[flame_segs["right_eye_region"]]) + list(
    # smplx_flame_vid[flame_segs["left_eye_region"]])
    if other_ids != None:
        remesh_ids = list(set(front_face_ids) - set(forehead_ids) - set(other_ids)) + ears_ids + eyeball_ids + hands_ids
    else:
        remesh_ids = list(set(front_face_ids) - set(forehead_ids)) + ears_ids + eyeball_ids + hands_ids
    remesh_mask = ~np.isin(np.arange(10475), remesh_ids) # obtain selected vertices
    remesh_mask = remesh_mask[smplx_face].all(axis=1) # obtain selected mesh

    face_ids = list(set(front_face_ids) - set(forehead_ids)) + eyeball_ids
    face_mask = ~np.isin(np.arange(10475), face_ids)
    face_mask = face_mask[smplx_face].all(axis=1)

    hands_mask = ~np.isin(np.arange(10475), hands_ids)
    hands_mask = hands_mask[smplx_face].all(axis=1)

    outside_mask = ~np.isin(np.arange(10475), outside_ids)
    outside_mask = outside_mask[smplx_face].all(axis=1)

    # feet_mask = ~np.isin(np.arange(10475), feet_ids)
    # feet_mask = feet_mask[smplx_face].all(axis=1)

    # ear_mask = ~np.isin(np.arange(10475), ears_ids)
    # ear_mask = ear_mask[smplx_face].all(axis=1)

    # mouth_in_mask = ~np.isin(np.arange(10475), other_ids)
    # mouth_in_mask = mouth_in_mask[smplx_face].all(axis=1)

    # nose_mask =  ~np.isin(np.arange(10475), nose_ids)
    # nose_mask =  nose_mask[smplx_face].all(axis=1)

    # lip_mask =  ~np.isin(np.arange(10475), lips_ids)
    # lip_mask =  lip_mask[smplx_face].all(axis=1)

    return remesh_mask, face_mask, hands_mask, outside_mask

def auto_uv(v, f):
    v_np = v
    f_np = f
    
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 4
    atlas.generate(chart_options=chart_options)
    vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

    return vt_np, ft_np


if __name__ == '__main__':
    #### Obtain subdivide regions
    smplx_mesh = load_obj('../../../work_dirs/cache/template/smplx_mouth_uv.obj')
    smplx_verts = smplx_mesh[0]
    smplx_uv = np.array(smplx_mesh[1])
    smplx_faces = np.array(smplx_mesh[2])
    mouth_inner_faces = smplx_faces[-30:]
    mouth_inner_ids = list(set(mouth_inner_faces.flatten().tolist()))
    smplx_uv_faces = np.array(smplx_mesh[3]) 
    remesh_mask, face_mask, hands_mask, outside_mask = get_seg_mask(smplx_faces, mouth_inner_ids)

    #### Obtain SMPLX canonical vertices
    # For custom
    # body_model = SMPLX('/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/smplx/SMPLX', gender='male', \
    #                             use_pca=True,
    #                             num_pca_comps=12,
    #                             num_betas=10,
    #                             flat_hand_mean=True)

    # For THuman
    body_model = SMPLX('smplx/SMPLX', gender='male', \
                                use_pca=True,
                                num_pca_comps=12,
                                num_betas=10,
                                flat_hand_mean=False)
    body_pose_t = torch.zeros((1, 63))

    # For custom
    # jaw_pose_t = torch.as_tensor([[0.2, 0, 0],])
    # transl = torch.as_tensor([[-0.0012,  0.4668, -0.0127],])

    # For THuman
    jaw_pose_t = torch.as_tensor([[0., 0, 0],])
    transl = torch.as_tensor([[0., 0.35, 0.],])
    left_hand_pose_t = torch.tensor([[1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
         -0.6652, -0.7290,  0.0084, -0.4818],])
    right_hand_pose_t = torch.tensor([[1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
         -0.6652, -0.7290,  0.0084, -0.4818],])
    
    global_orient = torch.zeros((1, 3))
    betas = torch.zeros((1, 10))

    # For custom
    # smpl_outputs = body_model(betas=betas, body_pose=body_pose_t, transl=transl, global_orient = global_orient, jaw_pose=jaw_pose_t)

    # For THuman
    smpl_outputs = body_model(betas=betas, body_pose=body_pose_t, transl=transl, global_orient = global_orient, jaw_pose=jaw_pose_t, left_hand_pose=left_hand_pose_t, right_hand_pose=right_hand_pose_t)
    vs_template = smpl_outputs.vertices.detach()[0]

    #### subdivide process
    num_remeshing = 1
    shapedirs = torch.cat([body_model.shapedirs, body_model.expr_dirs], dim=-1).reshape(10475, -1)
    posedirs = body_model.posedirs.reshape(-1, 10475, 3).permute(1, 2, 0)

    blending_weights = body_model.lbs_weights
    dense_v_cano, dense_faces, dense_blendweights, _ = subdivide(vs_template.clone(), smplx_faces[remesh_mask].copy(), blending_weights.clone())
    _, _, dense_shapedirs, _ = subdivide(vs_template.clone(), smplx_faces[remesh_mask].copy(), shapedirs.clone())
    _, _, dense_posedirs, _ = subdivide(vs_template.clone(), smplx_faces[remesh_mask].copy(), posedirs.clone())
    dense_uv, dense_uv_faces, _ = subdivide(smplx_uv, smplx_uv_faces[remesh_mask])
    np.save('../../../work_dirs/cache/init_podir_smplx_thu.npy', dense_posedirs.transpose(2, 0, 1).reshape(486, -1))
    np.save('../../../work_dirs/cache/init_spdir_smplx_thu.npy', dense_shapedirs.reshape(-1, 3, 20))


    mask_part1 = outside_mask[remesh_mask]
    dense_outside_mask = mask_part1.repeat(4)

    mask_face_part = face_mask[remesh_mask]
    dense_face_mask = mask_face_part.repeat(4)

    # mask_mouth_in_part = mouth_in_mask[remesh_mask]
    # dense_mouth_in_mask = mask_mouth_in_part.repeat(4)

    # nose_mask_part = nose_mask[remesh_mask]
    # dense_nose_mask = nose_mask_part.repeat(4)

    # lip_mask_part = lip_mask[remesh_mask]
    # dense_lip_mask = lip_mask_part.repeat(4)

    # feet_mask_part = feet_mask[remesh_mask]
    # dense_feet_mask = feet_mask_part.repeat(4)

    # ear_mask_part = ear_mask[remesh_mask]
    # dense_ear_mask = ear_mask_part.repeat(4)
    
    # mask_part1 = np.ones(dense_faces.shape[0]//4)
    # mask_part1[-30:] = 0
    # dense_mask_mouth_in = mask_part1.repeat(4)

    dense_face_masks = np.concatenate([dense_face_mask, face_mask[~remesh_mask]])
    dense_hands_masks = np.concatenate([np.ones(dense_faces.shape[0]), hands_mask[~remesh_mask]])
    dense_outside_masks = np.concatenate([dense_outside_mask, outside_mask[~remesh_mask]])
    # dense_lips_masks = np.concatenate([np.ones(dense_faces.shape[0]), lips_mask[~remesh_mask]])
    # dense_mouth_in_masks = np.concatenate([dense_mouth_in_mask, mouth_in_mask[~remesh_mask]])
    # dense_nose_masks = np.concatenate([dense_nose_mask, nose_mask[~remesh_mask]])
    # dense_lip_masks = np.concatenate([dense_lip_mask, lip_mask[~remesh_mask]])
    # dense_feet_masks = np.concatenate([dense_feet_mask, feet_mask[~remesh_mask]])
    # dense_ear_masks = np.concatenate([dense_ear_mask, ear_mask[~remesh_mask]])

    dense_face_masks = torch.as_tensor(dense_face_masks).bool()
    dense_hands_masks = torch.as_tensor(dense_hands_masks).bool()
    dense_outside_masks = torch.as_tensor(dense_outside_masks).bool()
    # dense_mouth_in_masks = torch.as_tensor(dense_mouth_in_masks).bool()
    # dense_nose_masks = torch.as_tensor(dense_nose_masks).bool()
    # dense_lip_masks = torch.as_tensor(dense_lip_masks).bool()
    # dense_feet_masks = torch.as_tensor(dense_feet_masks).bool()
    # dense_ear_masks = torch.as_tensor(dense_ear_masks).bool()

    np.save('../../../work_dirs/cache/outside_mask_thu.npy', np.array(~dense_outside_masks))
    np.save('../../../work_dirs/cache/face_mask_thu.npy', np.array(~dense_face_masks))
    np.save('../../../work_dirs/cache/hands_mask_thu.npy', np.array(~dense_hands_masks))
    
    dense_faces = np.concatenate([dense_faces, smplx_faces[~remesh_mask]])
    dense_uv_faces = np.concatenate([dense_uv_faces, smplx_uv_faces[~remesh_mask]])
    final_blendweights = np.stack([dense_blendweights[dense_faces[..., i]] for i in range(3)], axis=1).mean(1)
    np.save('../../../work_dirs/cache/init_lbsw_smplx_thu.npy', final_blendweights)

    for _ in range(1, num_remeshing):
        dense_v_cano, dense_faces, _ = subdivide(dense_v_cano, dense_faces)
        dense_uv, dense_uv_faces, _ = subdivide(dense_uv, dense_uv_faces)

    #### export densified mesh
    write_obj('../../../work_dirs/cache/template/dense_thuman.obj', dense_v_cano, dense_faces, dense_uv, dense_uv_faces)