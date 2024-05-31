import numpy as np
import trimesh
import open3d as o3d


def simplify_mesh(o_trimesh, scale=20):
    a_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(o_trimesh.vertices), triangles=o3d.utility.Vector3iVector(o_trimesh.faces))
    if o_trimesh.faces.shape[0] > 2000:
        simple_a_o3d = a_o3d.simplify_quadric_decimation(target_number_of_triangles=o_trimesh.faces.shape[0] // scale)
    else:
        simple_a_o3d = a_o3d
    
    simple_trimesh = trimesh.Trimesh(vertices=np.float32(simple_a_o3d.vertices), faces=np.int32(simple_a_o3d.triangles))
    return simple_trimesh
