import os
from os.path import join, isfile, dirname
import argparse
import numpy as np
import pickle
import torch
import trimesh
from smplx import smplx
import cv2
import imageio
from pytorch3d.renderer import PointLights
from pyt3d_wrapper import Pyt3DWrapper
from mesh_utils import simplify_mesh
from tqdm import tqdm


IMAGE_SIZE = (1920, 1080)
INTRINSIC = np.float32([
    [913.15277099609375, 0.0,  960.768798828125],
	[0.0, 912.6463623046875, 551.0286865234375],
	[0.0, 0.0, 1.0],
])
CAMERA_TO_WORLD = np.float32([
    [-3.362828398520266715e-01, 4.835672948475828736e-01, -8.081315010410416999e-01, 2.000948008927065480e+00],
    [-6.864405426693195866e-02, -8.684100987259859572e-01, -4.910721884250031088e-01, 1.749267542201616843e+00],
    [-9.392560063341914844e-01, -1.096657274836312368e-01, 3.252254337849846966e-01, -9.957471075610673239e-01],
    [0.0, 0.0, 0.0, 1.0],
])
EXTRINSIC = np.linalg.inv(CAMERA_TO_WORLD)


def visualize(obj_mesh, obj_poses, person1_smplx_data, person2_smplx_data, save_path=None, sampling_rate=1, device="cuda:0"):
    N = obj_poses.shape[0]
    
    obj_pts = obj_mesh.vertices
    obj_faces = obj_mesh.faces
    
    pyt3d_wrapper = Pyt3DWrapper(image_size=IMAGE_SIZE, use_fixed_cameras=True, intrin=INTRINSIC, extrin=EXTRINSIC, device=device, lights=PointLights(device=device, location=[[0.0, 4.0, 0.0]]))
    
    print("###### start visualization ... ######")
    
    assert not save_path is None
    os.makedirs(dirname(save_path), exist_ok=True)
    rgb_imgs = []
    
    for frame_idx in tqdm(range(0, N, sampling_rate)):
        # get object poses in this frame
        obj_pose = obj_poses[frame_idx]
        
        # construct object meshes in this frame
        obj_posed_mesh = trimesh.Trimesh(vertices=(obj_pts.copy() @ obj_pose[:3, :3].T) + obj_pose[:3, 3], faces=obj_faces.copy())
        person1_mesh = trimesh.Trimesh(vertices=person1_smplx_data["vertices"][frame_idx], faces=person1_smplx_data["faces"])
        person2_mesh = trimesh.Trimesh(vertices=person2_smplx_data["vertices"][frame_idx], faces=person2_smplx_data["faces"])
        meshes = [person1_mesh, person2_mesh, obj_posed_mesh]
        
        # render
        render_result = pyt3d_wrapper.render_meshes(meshes)
        img = (render_result[0]*255).astype(np.uint8)
        rgb_imgs.append(img)
    
    # save
    imageio.mimsave(save_path, rgb_imgs, duration=(1000/(15//sampling_rate)), loop=0)

    print("###### finish visualization !!! ######")


def process(args):
    dataset_root = args.dataset_root
    object_model_root = args.object_model_root
    sequence_name = args.sequence_name
    save_path = args.save_path
    device = args.device
    
    motion_dir = join(dataset_root, "human_object_motions", sequence_name)
    
    # get object name and category
    obj_name = None
    for fn in os.listdir(motion_dir):
        if fn.startswith("object_"):
            obj_name = fn.split(".")[0].split("_")[2]
    assert not obj_name is None
    category = obj_name[:-3]
    
    # load object model and poses
    obj_mesh = trimesh.load(join(object_model_root, category, obj_name + "_m.obj"))  # unit: m
    obj_mesh = simplify_mesh(obj_mesh, scale=20)
    obj_poses = np.load(join(motion_dir, "object_poses_{}.npy".format(obj_name)))
    
    # load human SMPLX data
    person1_smplx_data = np.load(join(motion_dir, "person1_poses.npz"), allow_pickle=True)["arr_0"].item()
    person2_smplx_data = np.load(join(motion_dir, "person2_poses.npz"), allow_pickle=True)["arr_0"].item()
    
    # visualize object poses as an RGB video
    visualize(obj_mesh, obj_poses, person1_smplx_data, person2_smplx_data, save_path=save_path, sampling_rate=1, device=device)


def get_args():
    parser = argparse.ArgumentParser()
    ###################################################################
    parser.add_argument('--dataset_root', type=str, default="/share/datasets/hhodataset/CORE4D_release/CORE4D_Real")
    parser.add_argument('--object_model_root', type=str, default="/share/datasets/hhodataset/CORE4D_release/CORE4D_Real/object_models")
    parser.add_argument('--sequence_name', type=str, default="20231002/004")
    parser.add_argument('--save_path', type=str, default="./example.gif")
    parser.add_argument('--device', type=str, default="cuda:0")
    ###################################################################

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    process(args)
