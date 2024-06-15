import os
from os.path import join, isdir, isfile
import trimesh


def load_hho_object_meshes():
    obj_model_root = "/data2/datasets/HHO_object_dataset_final_simplified"

    mesh_dict = {}
    for category in os.listdir(obj_model_root):
        category_dir = join(obj_model_root, category)
        if not isdir(category_dir):
            continue
        for obj_name in os.listdir(category_dir):
            obj_dir = join(category_dir, obj_name)
            if not isdir(obj_dir):
                continue
            for file_name in os.listdir(obj_dir):
                if file_name.endswith("_m.obj"):
                    fp = join(obj_dir, file_name)
                    mesh_dict[fp] = trimesh.load(fp)

    return mesh_dict


if __name__ == "__main__":
    mesh_dict = load_hho_object_meshes()
    for k in mesh_dict:
        print(k, mesh_dict[k])
