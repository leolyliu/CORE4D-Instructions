import os
from os.path import dirname
import trimesh
import open3d as o3d


def save_mesh(trimesh_mesh, save_path):
    os.makedirs(dirname(save_path), exist_ok=True)
    mesh_txt = trimesh.exchange.obj.export_obj(trimesh_mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
    with open(save_path, "w") as fp:
        fp.write(mesh_txt)
