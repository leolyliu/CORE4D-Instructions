import os
from os.path import join, isfile, isdir, dirname, abspath
import sys
import argparse
import numpy as np
from copy import deepcopy
import trimesh
import igl
from os.path import isfile
from psbody.mesh import Mesh
import yaml
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json
sys.path.insert(0, join(dirname(abspath(__file__)), "../../../.."))  # benchmarks
from motion_forecasting.MDM_InterDiff.interdiff.data.prepare_behave import ContactLabelGenerator
sys.path.insert(0, join(dirname(abspath(__file__)), "../../../.."))  # benchmarks
from data_processing.utils.VTS_object import get_obj_info
from data_processing.utils.load_smplx_params import load_multiperson_smplx_params


def prepare_a_sequence(seq_data, num_samples):
    N_frame = seq_data["N_frame"]
    obj_model_path = seq_data["obj_model_path"]
    save_dir = seq_data["save_dir"]
    obj_poses = seq_data["obj_poses"]
    human_params = seq_data["human_params"]
    os.makedirs(save_dir, exist_ok=True)
    
    # obj mesh and downsampled point cloud
    # mesh_obj = Mesh()
    # mesh_obj.load_from_obj(obj_model_path)
    # obj_verts = mesh_obj.v
    # obj_faces = mesh_obj.f
    # obj = trimesh.Trimesh(obj_verts, obj_faces, process=False)
    obj = trimesh.load(obj_model_path)
    object_points, ids = obj.sample(num_samples, return_index=True)
    object_normals = obj.face_normals[ids]
    object_all = np.concatenate([object_points, object_normals], axis=1)
    
    # tensor to np
    for person in human_params:
        for key in human_params[person]:
            if isinstance(human_params[person][key], torch.Tensor):
                human_params[person][key] = human_params[person][key].detach().cpu().numpy()
    
    generator = ContactLabelGenerator()
    data = {
        "object_points": object_all,  # shape = (num_samples, 6)
        "N_frame": N_frame,  # int
        "obj_model_path": obj_model_path,  # str
        "obj_poses": obj_poses,  # shape = (N_frame, 4, 4)
        "human_params": human_params,  # {"person1": {...}, "person2": {...}}
        "contact_object_to_person1": [],
        "contact_person1_to_object": [],
        "contact_object_to_person2": [],
        "contact_person2_to_object": [],
        "foot_contact_label_person1": [],
        "foot_contact_label_person2": [],
    }
    for i in tqdm(range(N_frame)):
        person1_mesh = trimesh.Trimesh(human_params["person1"]["vertices"][i], human_params["person1"]["faces"], process=False)
        person2_mesh = trimesh.Trimesh(human_params["person2"]["vertices"][i], human_params["person2"]["faces"], process=False)
        person1_joints = human_params["person1"]["joints"][i]
        person2_joints = human_params["person2"]["joints"][i]
        foot_contact_label_person1 = 10 if person1_joints[10, 1] < person1_joints[11, 1] else 11
        foot_contact_label_person2 = 10 if person2_joints[10, 1] < person2_joints[11, 1] else 11
        obj_v = object_points.copy()  # points in the object canonical space
        obj_pose = obj_poses[i]
        obj_v = (obj_v @ obj_pose[:3, :3].T) + obj_pose[:3, 3]  # points in the world space
        contact_object_to_person1, contact_person1_to_object = generator.get_contact_labels(person1_mesh, obj_v)
        contact_object_to_person2, contact_person2_to_object = generator.get_contact_labels(person2_mesh, obj_v)
        
        # save
        data["contact_object_to_person1"].append(contact_object_to_person1)
        data["contact_person1_to_object"].append(contact_person1_to_object)
        data["contact_object_to_person2"].append(contact_object_to_person2)
        data["contact_person2_to_object"].append(contact_person2_to_object)
        data["foot_contact_label_person1"].append(foot_contact_label_person1)
        data["foot_contact_label_person2"].append(foot_contact_label_person2)
    
    np.savez(join(save_dir, "data.npz"), data=data)


def main(dataset_root, obj_dataset_dir, save_root, smplx_model_dir, clip_names, num_samples, device):
    for clip_name in clip_names:
        clip_dir = join(dataset_root, clip_name)
        for seq_name in os.listdir(clip_dir):
            seq_dir = join(clip_dir, seq_name)
            if not isdir(seq_dir):
                continue
            print("processing {} ...".format(seq_dir))
            
            Opose_fp = join(seq_dir, "smooth_objposes.npy")
            HHpose_dir = seq_dir
            if (not isfile(Opose_fp)) or (not isfile(join(HHpose_dir, "person1_poses.npz"))) or (not isfile(join(HHpose_dir, "person2_poses.npz"))):
                print("[error] object_pose or human_pose does not exist !!!")
                continue
            
            obj_name, obj_model_path = get_obj_info(seq_dir, obj_dataset_dir)
            if not isfile(obj_model_path):
                print("[error] no object mesh for object {} at {} !!!".format(obj_name, obj_model_path))
                continue
            
            obj_poses = np.load(Opose_fp)  # 世界系objpose, shape = (N_frame, 4, 4)
            N_frame = obj_poses.shape[0]
            human_params = load_multiperson_smplx_params(HHpose_dir, smplx_model_dir, start_frame=0, end_frame=N_frame, device=device)
            if human_params["person1"]["betas"].shape[0] != N_frame:
                print("[error] incorrect frame number of person1 !!!")
                continue
            if human_params["person2"]["betas"].shape[0] != N_frame:
                print("[error] incorrect frame number of person2 !!!")
                continue
            seq_data = {
                "N_frame": N_frame,  # int
                "obj_model_path": obj_model_path,  # obj file
                "obj_poses": obj_poses,  # shape = (N_frame, 4, 4)
                "human_params": human_params,  # {"person1": {...}, "person2": {}}
                "save_dir": join(save_root, clip_name, seq_name),
            }
            prepare_a_sequence(seq_data, num_samples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_name', type=str, default="all")  # e.g.: 20231018.20231020
    parser.add_argument('--num_samples', type=int, default="2048")  # downsampled object point number
    parser.add_argument('--dataset_root', type=str, default="/share/datasets/hhodataset/CORE4D_release/CORE4D_Real/human_object_motions")
    parser.add_argument('--obj_dataset_dir', type=str, default="/share/datasets/hhodataset/CORE4D_release/CORE4D_Real/object_models")
    parser.add_argument('--smplx_model_dir', type=str, default="/share/human_model/models")
    parser.add_argument('--save_root', type=str, default="/share/datasets/hhodataset/prepared_motion_forecasting_data")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.clip_name == "all":
        clip_names = ["20231002", "20231003_1", "20231003_2", "20231008", "20231011", "20231018", "20231020", "20231023", "20231030", "20231108"]
    else:
        clip_names = args.clip_name.split(".")

    main(args.dataset_root, args.obj_dataset_dir, args.save_root, args.smplx_model_dir, clip_names, args.num_samples, args.device)
