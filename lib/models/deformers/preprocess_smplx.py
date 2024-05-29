import trimesh
import numpy as np
import os
from smplx import SMPLX
import torch


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


if __name__ == '__main__':
    # v, vt, vi, vti
    flame_mouth_mesh = load_obj('../../../work_dirs/cache/template/head_template_mesh_mouth.obj')
    flame_mouth_faces = flame_mouth_mesh[2]
    flame_mouth_uv = flame_mouth_mesh[1]
    flame_mouth_uv_faces = flame_mouth_mesh[3]
    flame_mesh = load_obj('../../../work_dirs/cache/template/head_template.obj')
    flame_faces = flame_mesh[2]
    flame_uv = flame_mesh[1]
    flame_uv_faces = flame_mesh[3]
    smplx_mesh = load_obj('../../../work_dirs/cache/template/smplx_uv.obj')
    smplx_verts = smplx_mesh[0]
    smplx_uv = smplx_mesh[1]
    smplx_faces = smplx_mesh[2]
    smplx_uv_faces = smplx_mesh[3]

    extra_uvs = [uv for uv in flame_mouth_uv if uv not in flame_uv]
    extra_uvs = np.array(extra_uvs)
    extra_uvs *= np.array([[0.1, 0.06], ])
    extra_uvs += np.array([[0.52, 0.55], ])

    extra_faces = [face for face in flame_mouth_faces if face not in flame_faces]
    extra_uv_faces = [face for face in flame_mouth_uv_faces if face not in flame_uv_faces]
    extra_uv_faces_smplx = np.array(extra_uv_faces) - len(flame_uv) + len(smplx_uv)

    smplx_flame_correspond = np.load('../../../work_dirs/cache/template/SMPL-X__FLAME_vertex_ids.npy')

    extra_faces_smplx = []
    for face in extra_faces:
        new_faces = [smplx_flame_correspond[idx] for idx in face]
        extra_faces_smplx.append(new_faces)

    # write obj
    vertices = np.array(smplx_verts)
    uvcoords = np.array(smplx_uv + extra_uvs.tolist())
    faces = np.array(smplx_faces + extra_faces_smplx)
    uvfaces = np.array(smplx_uv_faces + extra_uv_faces_smplx.tolist())
    output_obj_path = '../../../work_dirs/cache/template/smplx_mouth_uv.obj'
    write_obj(output_obj_path, vertices, faces, uvcoords=uvcoords, uvfaces=uvfaces)
    
    

