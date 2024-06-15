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
sys.path.insert(0, join(dirname(abspath(__file__)), "../../../.."))  # HHO-dataset
from motion_forecasting.InterDiff.interdiff.data.prepare_behave import ContactLabelGenerator
sys.path.insert(0, join(dirname(abspath(__file__)), "../../../.."))  # HHO-dataset
from data_processing.utils.VTS_object import get_obj_info
from data_processing.utils.load_smplx_params import load_multiperson_smplx_params, create_SMPLX_model
from data_processing.utils.mesh import save_mesh
from dataset_statistics.train_test_split import TRAIN_OBJECTS, TEST_OBJECTS, load_train_test_split
from data_processing.smplx.smplx.lbs import batch_rodrigues


def prepare_a_sequence(seq_data, num_samples):
    N_frame = seq_data["N_frame"]
    obj_model_path = seq_data["obj_model_path"]
    save_dir = seq_data["save_dir"]
    obj_poses = seq_data["obj_poses"]
    human_params = seq_data["human_params"]
    os.makedirs(save_dir, exist_ok=True)
    
    # obj mesh and downsampled point cloud
    mesh_obj = Mesh()
    mesh_obj.load_from_obj(obj_model_path)
    obj_verts = mesh_obj.v
    obj_faces = mesh_obj.f
    obj = trimesh.Trimesh(obj_verts, obj_faces, process=False)
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
        obj_v = object_points.copy()  # canonical space下的坐标
        obj_pose = obj_poses[i]
        obj_v = (obj_v @ obj_pose[:3, :3].T) + obj_pose[:3, 3]  # 世界系下的坐标
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


def main(dataset_root, obj_dataset_dir, save_root, clip_names, num_samples, device):
    for clip_name in clip_names:
        clip_dir = join(dataset_root, clip_name)
        for seq_name in os.listdir(clip_dir):
            seq_dir = join(clip_dir, seq_name)
            if not isdir(seq_dir):
                continue
            print("processing {} ...".format(seq_dir))
            
            Opose_fp = join(seq_dir, "aligned_objposes.npy")
            HHpose_dir = join(seq_dir, "SMPLX_fitting")
            if (not isfile(Opose_fp)) or (not isdir(HHpose_dir)):
                print("[error] object_pose or human_pose does not exist !!!")
                continue
            
            obj_name, obj_model_path = get_obj_info(seq_dir, obj_dataset_dir)
            if not isfile(obj_model_path):
                print("[error] no object mesh for object {} at {} !!!".format(obj_name, obj_model_path))
                continue
            
            obj_poses = np.load(Opose_fp)  # 世界系objpose, shape = (N_frame, 4, 4)
            N_frame = obj_poses.shape[0]
            try:
                human_params = load_multiperson_smplx_params(HHpose_dir, start_frame=0, end_frame=N_frame, device=device)
            except:
                print("[error] error in load_multiperson_smplx_params for {}".format(seq_dir))
                continue
            seq_data = {
                "N_frame": N_frame,  # int
                "obj_model_path": obj_model_path,  # obj file
                "obj_poses": obj_poses,  # shape = (N_frame, 4, 4)
                "human_params": human_params,  # {"person1": {...}, "person2": {}}
                "save_dir": join(save_root, clip_name, seq_name),
            }
            prepare_a_sequence(seq_data, num_samples)


def read_txt(txt_path):
    lines = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            lines.append(line)
    return lines


def main_retargeted_data_old(save_root, num_samples, device):
    
    gt_obj_name_fp = "/home/liuyun/HHO-dataset/dataset_statistics/retargeting_metadata/gt_obj_name_list.txt"
    # human_pose_fp = "/home/liuyun/HHO-dataset/dataset_statistics/retargeting_metadata/human_pose_内层_list.txt"
    human_pose_fp = "/home/liuyun/HHO-dataset/dataset_statistics/retargeting_metadata/human_pose_打分器_list.txt"
    obj_pose_fp = "/home/liuyun/HHO-dataset/dataset_statistics/retargeting_metadata/obj_pose_list.txt"
    obj_mesh_fp = "/home/liuyun/HHO-dataset/dataset_statistics/retargeting_metadata/obj_mesh_list.txt"
    
    gt_obj_names = read_txt(gt_obj_name_fp)
    N_seq = len(gt_obj_names)
    train_seq_ids, test_seq_ids = [], []
    for i, obj_name in enumerate(gt_obj_names):
        if obj_name in TRAIN_OBJECTS:
            train_seq_ids.append(i)
        elif obj_name in TEST_OBJECTS:
            test_seq_ids.append(i)
        else:
            assert False
    
    print("training seq number =", len(train_seq_ids))
    print("test seq number =", len(test_seq_ids))
    
    human_pose_list = read_txt(human_pose_fp)
    obj_pose_list = read_txt(obj_pose_fp)
    obj_mesh_list = read_txt(obj_mesh_fp)
    assert len(human_pose_list) == N_seq
    assert len(obj_pose_list) == N_seq
    assert len(obj_mesh_list) == N_seq
    
    train_seq_list, test_seq_list = [], []
    
    for seq_idx, (human_pose_path, obj_pose_path, obj_model_path) in enumerate(zip(human_pose_list, obj_pose_list, obj_mesh_list)):
        
        print("preparing {} ...".format(obj_pose_path))
        
        raw_human_poses = np.load(human_pose_path, allow_pickle=True)
        raw_obj_poses = np.load(obj_pose_path, allow_pickle=True)
        
        clip_name, seq_name = obj_pose_path.split("/")[5:7]
        seq_name += "_" + obj_pose_path.split("/")[-1].split(".")[0] + "_outdis"
        
        if seq_idx in train_seq_ids:
            train_seq_list.append(clip_name + "." + seq_name)
        else:
            test_seq_list.append(clip_name + "." + seq_name)
        
        N_frame = len(raw_human_poses)
        
        # prepare obj poses
        obj_poses = []
        for i in range(N_frame):
            r = raw_obj_poses[i]["rotation"].reshape(1, 3)
            R = batch_rodrigues(r).reshape(3, 3).detach().cpu().numpy()
            t = raw_obj_poses[i]["translation"].detach().cpu().numpy().reshape(3)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            obj_poses.append(T)
        obj_poses = np.float32(obj_poses)
        
        # prepare human params
        N_betas = raw_human_poses[0]["person1"]["betas"].shape[1]
        N_expression = raw_human_poses[0]["person1"]["expression"].shape[1]
        N_hand_pca = raw_human_poses[0]["person1"]["left_hand_pose"].shape[1]
        human_params = {}
        smplx_model = create_SMPLX_model(batch_size=N_frame, device=device)
        
        for person in ["person1", "person2"]:
            human_params[person] = {
                "betas": torch.zeros((N_frame, N_betas), dtype=torch.float32).to(device),
                "expression": torch.zeros((N_frame, N_expression), dtype=torch.float32).to(device),
                "global_orient": torch.zeros((N_frame, 3), dtype=torch.float32).to(device),
                "transl": torch.zeros((N_frame, 3), dtype=torch.float32).to(device),
                "body_pose": torch.zeros((N_frame, 21, 3)).to(device),
                "left_hand_pose": torch.zeros((N_frame, N_hand_pca)).to(device),
                "right_hand_pose": torch.zeros((N_frame, N_hand_pca)).to(device),
                "joints": torch.zeros((N_frame, 127, 3)).to(device),
                "vertices": torch.zeros((N_frame, 10475, 3)).to(device),
                "faces": smplx_model.faces_tensor.reshape(-1, 3),
            }
            for i in range(N_frame):
                human_params[person]["betas"][i] = raw_human_poses[i][person]["betas"][0].to(device)
                human_params[person]["expression"][i] = raw_human_poses[i][person]["expression"][0].to(device)
                human_params[person]["global_orient"][i] = raw_human_poses[i][person]["global_orient"][0].to(device)
                human_params[person]["transl"][i] = raw_human_poses[i][person]["transl"][0].to(device)
                human_params[person]["body_pose"][i] = raw_human_poses[i][person]["body_pose"][0].to(device)
                human_params[person]["left_hand_pose"][i] = raw_human_poses[i][person]["left_hand_pose"][0].to(device)
                human_params[person]["right_hand_pose"][i] = raw_human_poses[i][person]["right_hand_pose"][0].to(device)
            
            model = smplx_model(betas=human_params[person]["betas"], body_pose=human_params[person]["body_pose"], global_orient=human_params[person]["global_orient"], transl=human_params[person]["transl"], left_hand_pose=human_params[person]["left_hand_pose"], right_hand_pose=human_params[person]["right_hand_pose"])
            assert model.joints.shape == (N_frame, 127, 3)
            assert model.vertices.shape == (N_frame, 10475, 3)
            human_params[person]["joints"] = model.joints
            human_params[person]["vertices"] = model.vertices
        
        seq_data = {
            "N_frame": N_frame,  # int
            "obj_model_path": obj_model_path,  # obj file
            "obj_poses": obj_poses,  # shape = (N_frame, 4, 4)
            "human_params": human_params,  # {"person1": {...}, "person2": {}}
            "save_dir": join(save_root, clip_name, seq_name),
        }
        
        prepare_a_sequence(seq_data, num_samples)
    
    json.dump(train_seq_list, open("/home/liuyun/HHO-dataset/dataset_statistics/retargeting_metadata/train_sequence_names_retargeted_outdis.json", "w"))
    json.dump(test_seq_list, open("/home/liuyun/HHO-dataset/dataset_statistics/retargeting_metadata/test_sequence_names_retargeted_outdis.json", "w"))


def main_retargeted_data_0303(save_root, num_samples, device):
    
    human_pose_fps = [
        "/home/liuyun/HHO-dataset/dataset_visualization/0302_1/retargeting_video_paths_.txt",  # chair,desk
        "/home/liuyun/HHO-dataset/dataset_visualization/0302_1/retargeting_video_paths_内层+外层+打分器.txt",  # chair,desk
    ]
    
    human_pose_list = []
    for fp in human_pose_fps:
        human_pose_list += read_txt(fp)
    obj_pose_list, obj_mesh_list = [], []
    for human_pose_fp in human_pose_list:
        obj_retargeting_dir = join(dirname(dirname(dirname(human_pose_fp))), "obj_retargeting_final")
        retarget_obj_name = human_pose_fp.split("/")[-2].split("_")[0]
        obj_pose_list.append(join(obj_retargeting_dir, retarget_obj_name + ".npy"))
        obj_mesh_list.append(join(obj_retargeting_dir, retarget_obj_name + ".obj"))
    
    train_sequence_names, _, _ = load_train_test_split()
    train_seq_ids = []
    for i, human_pose_fp in enumerate(human_pose_list):
        clip, seq = human_pose_fp.split("/")[-5:-3]
        seq_name = clip + "." + seq
        if seq_name in train_sequence_names:
            train_seq_ids.append(i)
    
    print("training seq number =", len(train_seq_ids))
    
    train_seq_list = []
    
    for seq_idx, (human_pose_path, obj_pose_path, obj_model_path) in enumerate(zip(human_pose_list, obj_pose_list, obj_mesh_list)):
        
        if not seq_idx in train_seq_ids:
            continue
        
        raw_human_poses = np.load(human_pose_path, allow_pickle=True)
        raw_obj_poses = np.load(obj_pose_path, allow_pickle=True)
        
        clip_name, seq_name = human_pose_path.split("/")[-5:-3]
        seq_name += "_" + human_pose_path.split("/")[-2].split("_")[0] + "_seq" + human_pose_path.split("/")[-1].split(".")[0] + "_0303"
        
        train_seq_list.append(clip_name + "." + seq_name)
        
        print("preparing {}, sequence name = {} ...".format(human_pose_path, clip_name + "." + seq_name))
        
        N_frame = len(raw_human_poses)
        
        # prepare obj poses
        obj_poses = []
        for i in range(N_frame):
            r = raw_obj_poses[i]["rotation"].reshape(1, 3)
            R = batch_rodrigues(r).reshape(3, 3).detach().cpu().numpy()
            t = raw_obj_poses[i]["translation"].detach().cpu().numpy().reshape(3)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            obj_poses.append(T)
        obj_poses = np.float32(obj_poses)
        
        # prepare human params
        N_betas = raw_human_poses[0]["person1"]["betas"].shape[1]
        N_expression = raw_human_poses[0]["person1"]["expression"].shape[1]
        N_hand_pca = raw_human_poses[0]["person1"]["left_hand_pose"].shape[1]
        human_params = {}
        smplx_model = create_SMPLX_model(batch_size=N_frame, device=device)
        
        for person in ["person1", "person2"]:
            human_params[person] = {
                "betas": torch.zeros((N_frame, N_betas), dtype=torch.float32).to(device),
                "expression": torch.zeros((N_frame, N_expression), dtype=torch.float32).to(device),
                "global_orient": torch.zeros((N_frame, 3), dtype=torch.float32).to(device),
                "transl": torch.zeros((N_frame, 3), dtype=torch.float32).to(device),
                "body_pose": torch.zeros((N_frame, 21, 3)).to(device),
                "left_hand_pose": torch.zeros((N_frame, N_hand_pca)).to(device),
                "right_hand_pose": torch.zeros((N_frame, N_hand_pca)).to(device),
                "joints": torch.zeros((N_frame, 127, 3)).to(device),
                "vertices": torch.zeros((N_frame, 10475, 3)).to(device),
                "faces": smplx_model.faces_tensor.reshape(-1, 3),
            }
            for i in range(N_frame):
                human_params[person]["betas"][i] = raw_human_poses[i][person]["betas"][0].to(device)
                human_params[person]["expression"][i] = raw_human_poses[i][person]["expression"][0].to(device)
                human_params[person]["global_orient"][i] = raw_human_poses[i][person]["global_orient"][0].to(device)
                human_params[person]["transl"][i] = raw_human_poses[i][person]["transl"][0].to(device)
                human_params[person]["body_pose"][i] = raw_human_poses[i][person]["body_pose"][0].to(device)
                human_params[person]["left_hand_pose"][i] = raw_human_poses[i][person]["left_hand_pose"][0].to(device)
                human_params[person]["right_hand_pose"][i] = raw_human_poses[i][person]["right_hand_pose"][0].to(device)
            
            model = smplx_model(betas=human_params[person]["betas"], body_pose=human_params[person]["body_pose"], global_orient=human_params[person]["global_orient"], transl=human_params[person]["transl"], left_hand_pose=human_params[person]["left_hand_pose"], right_hand_pose=human_params[person]["right_hand_pose"])
            assert model.joints.shape == (N_frame, 127, 3)
            assert model.vertices.shape == (N_frame, 10475, 3)
            human_params[person]["joints"] = model.joints
            human_params[person]["vertices"] = model.vertices
        
        seq_data = {
            "N_frame": N_frame,  # int
            "obj_model_path": obj_model_path,  # obj file
            "obj_poses": obj_poses,  # shape = (N_frame, 4, 4)
            "human_params": human_params,  # {"person1": {...}, "person2": {}}
            "save_dir": join(save_root, clip_name, seq_name),
        }
        
        prepare_a_sequence(seq_data, num_samples)
    
    json.dump(train_seq_list, open("/home/liuyun/HHO-dataset/dataset_statistics/retargeting_metadata/train_sequence_names_retargeted_0303.json", "w"))


def main_retargeted_data_0305(save_root, num_samples, device):
    
    human_pose_fps = [
        "/home/liuyun/HHO-dataset/dataset_visualization/0302_2/retargeting_video_paths_simple_scale.txt",  # board, box, barrel, stick
    ]
    
    human_pose_list = []
    for fp in human_pose_fps:
        human_pose_list += read_txt(fp)
    obj_pose_list, obj_mesh_list = [], []
    for human_pose_fp in human_pose_list:
        obj_retargeting_dir = join(dirname(dirname(human_pose_fp)), "obj_retargeting_simple_scale")
        retarget_obj_name = "_".join(human_pose_fp.split("/")[-1].split("_")[1:6])
        obj_pose_list.append(join(obj_retargeting_dir, retarget_obj_name + ".npy"))
        obj_mesh_list.append(join(obj_retargeting_dir, retarget_obj_name + ".obj"))
    
    train_sequence_names, _, _ = load_train_test_split()
    train_seq_ids = []
    for i, human_pose_fp in enumerate(human_pose_list):
        try:
            clip, seq = human_pose_fp.split("/")[-4:-2]
        except:
            print("[error]", i, human_pose_fp)
            continue
        seq_name = clip + "." + seq
        if seq_name in train_sequence_names:
            train_seq_ids.append(i)
    
    print("training seq number =", len(train_seq_ids))
    
    train_seq_list = []
    
    for seq_idx, (human_pose_path, obj_pose_path, obj_model_path) in enumerate(zip(human_pose_list, obj_pose_list, obj_mesh_list)):
        
        if not seq_idx in train_seq_ids:
            continue
        
        clip_name, seq_name = human_pose_path.split("/")[-5:-3]
        seq_name += "_" + human_pose_path.split("/")[-2].split("_")[0] + "_seq" + human_pose_path.split("/")[-1].split(".")[0] + "_0303"
        
        train_seq_list.append(clip_name + "." + seq_name)
        
        print("preparing {}, {}, sequence name = {} ...".format(seq_idx, human_pose_path, clip_name + "." + seq_name))
        
        raw_human_poses = np.load(human_pose_path, allow_pickle=True)
        raw_obj_poses = np.load(obj_pose_path, allow_pickle=True)
        N_frame = len(raw_human_poses)
        
        # prepare obj poses
        obj_poses = []
        for i in range(N_frame):
            r = raw_obj_poses[i]["rotation"].reshape(1, 3)
            R = batch_rodrigues(r).reshape(3, 3).detach().cpu().numpy()
            t = raw_obj_poses[i]["translation"].detach().cpu().numpy().reshape(3)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            obj_poses.append(T)
        obj_poses = np.float32(obj_poses)
        
        # prepare human params
        N_betas = raw_human_poses[0]["person1"]["betas"].shape[1]
        N_expression = raw_human_poses[0]["person1"]["expression"].shape[1]
        N_hand_pca = raw_human_poses[0]["person1"]["left_hand_pose"].shape[1]
        human_params = {}
        smplx_model = create_SMPLX_model(batch_size=N_frame, device=device)
        
        for person in ["person1", "person2"]:
            human_params[person] = {
                "betas": torch.zeros((N_frame, N_betas), dtype=torch.float32).to(device),
                "expression": torch.zeros((N_frame, N_expression), dtype=torch.float32).to(device),
                "global_orient": torch.zeros((N_frame, 3), dtype=torch.float32).to(device),
                "transl": torch.zeros((N_frame, 3), dtype=torch.float32).to(device),
                "body_pose": torch.zeros((N_frame, 21, 3)).to(device),
                "left_hand_pose": torch.zeros((N_frame, N_hand_pca)).to(device),
                "right_hand_pose": torch.zeros((N_frame, N_hand_pca)).to(device),
                "joints": torch.zeros((N_frame, 127, 3)).to(device),
                "vertices": torch.zeros((N_frame, 10475, 3)).to(device),
                "faces": smplx_model.faces_tensor.reshape(-1, 3),
            }
            for i in range(N_frame):
                human_params[person]["betas"][i] = raw_human_poses[i][person]["betas"][0].to(device)
                human_params[person]["expression"][i] = raw_human_poses[i][person]["expression"][0].to(device)
                human_params[person]["global_orient"][i] = raw_human_poses[i][person]["global_orient"][0].to(device)
                human_params[person]["transl"][i] = raw_human_poses[i][person]["transl"][0].to(device)
                human_params[person]["body_pose"][i] = raw_human_poses[i][person]["body_pose"][0].to(device)
                human_params[person]["left_hand_pose"][i] = raw_human_poses[i][person]["left_hand_pose"][0].to(device)
                human_params[person]["right_hand_pose"][i] = raw_human_poses[i][person]["right_hand_pose"][0].to(device)
            
            model = smplx_model(betas=human_params[person]["betas"], body_pose=human_params[person]["body_pose"], global_orient=human_params[person]["global_orient"], transl=human_params[person]["transl"], left_hand_pose=human_params[person]["left_hand_pose"], right_hand_pose=human_params[person]["right_hand_pose"])
            assert model.joints.shape == (N_frame, 127, 3)
            assert model.vertices.shape == (N_frame, 10475, 3)
            human_params[person]["joints"] = model.joints
            human_params[person]["vertices"] = model.vertices
        
        seq_data = {
            "N_frame": N_frame,  # int
            "obj_model_path": obj_model_path,  # obj file
            "obj_poses": obj_poses,  # shape = (N_frame, 4, 4)
            "human_params": human_params,  # {"person1": {...}, "person2": {}}
            "save_dir": join(save_root, clip_name, seq_name),
        }
        
        prepare_a_sequence(seq_data, num_samples)
    
    json.dump(train_seq_list, open("/home/liuyun/HHO-dataset/dataset_statistics/retargeting_metadata/train_sequence_names_retargeted_0305.json", "w"))


def main_retargeted_data_0307(save_root, num_samples, device):
    
    human_pose_fps = [
        "/home/liuyun/HHO-dataset/dataset_visualization/0302_3/retargeting_video_paths_best.txt",  # board, box, barrel, stick
    ]
    
    human_pose_list = []
    for fp in human_pose_fps:
        human_pose_list += read_txt(fp)
        
    hpl_copy = human_pose_list.copy()
    human_pose_list = []
    for x in hpl_copy:
        d = dirname(x)
        mp4_fn = x.split("/")[-1]
        try:
            idx = int(mp4_fn[:4])
        except:
            continue
        hp = None
        for fn in os.listdir(d):
            if (fn[:4] == mp4_fn[:4]) and (not fn.endswith(".mp4")):
                hp = join(d, fn)
                break
        if not hp is None:
            human_pose_list.append(hp)

    obj_pose_list, obj_mesh_list = [], []
    for human_pose_fp in human_pose_list:
        obj_retargeting_dir = join(dirname(dirname(human_pose_fp)), "obj_retargeting_scale")
        retarget_obj_name = "_".join(human_pose_fp.split("/")[-2].split("_")[0:5])
        obj_pose_list.append(join(obj_retargeting_dir, retarget_obj_name + ".npy"))
        obj_mesh_list.append(join(obj_retargeting_dir, retarget_obj_name + ".obj"))
    
    train_sequence_names, _, _ = load_train_test_split()
    train_seq_ids = []
    for i, human_pose_fp in enumerate(human_pose_list):
        try:
            clip, seq = human_pose_fp.split("/")[-4:-2]
        except:
            print("[error]", i, human_pose_fp)
            continue
        seq_name = clip + "." + seq
        if seq_name in train_sequence_names:
            train_seq_ids.append(i)
    
    print("training seq number =", len(train_seq_ids))
    
    train_seq_list = []
    
    cnt = -1
    for seq_idx, (human_pose_path, obj_pose_path, obj_model_path) in enumerate(zip(human_pose_list, obj_pose_list, obj_mesh_list)):
        
        if not seq_idx in train_seq_ids:
            continue
        
        cnt += 1
        if (cnt < 900) or (10000 <= cnt):
            continue
        
        clip_name, seq_name = human_pose_path.split("/")[-4:-2]
        seq_name += "_" + "_".join(human_pose_path.split("/")[-2].split("_")[0:5]) + "_seq" + human_pose_path.split("/")[-1][:4] + "_0307"
        
        train_seq_list.append(clip_name + "." + seq_name)
        
        print("preparing {}, {}, sequence name = {} ...".format(seq_idx, human_pose_path, clip_name + "." + seq_name))
        
        raw_human_poses = np.load(human_pose_path, allow_pickle=True)
        raw_obj_poses = np.load(obj_pose_path, allow_pickle=True)
        N_frame = len(raw_human_poses)
        
        # prepare obj poses
        obj_poses = []
        for i in range(N_frame):
            r = raw_obj_poses[i]["rotation"].reshape(1, 3)
            R = batch_rodrigues(r).reshape(3, 3).detach().cpu().numpy()
            t = raw_obj_poses[i]["translation"].detach().cpu().numpy().reshape(3)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            obj_poses.append(T)
        obj_poses = np.float32(obj_poses)
        
        # prepare human params
        N_betas = raw_human_poses[0]["person1"]["betas"].shape[1]
        N_expression = raw_human_poses[0]["person1"]["expression"].shape[1]
        N_hand_pca = raw_human_poses[0]["person1"]["left_hand_pose"].shape[1]
        human_params = {}
        smplx_model = create_SMPLX_model(batch_size=N_frame, device=device)
        
        for person in ["person1", "person2"]:
            human_params[person] = {
                "betas": torch.zeros((N_frame, N_betas), dtype=torch.float32).to(device),
                "expression": torch.zeros((N_frame, N_expression), dtype=torch.float32).to(device),
                "global_orient": torch.zeros((N_frame, 3), dtype=torch.float32).to(device),
                "transl": torch.zeros((N_frame, 3), dtype=torch.float32).to(device),
                "body_pose": torch.zeros((N_frame, 21, 3)).to(device),
                "left_hand_pose": torch.zeros((N_frame, N_hand_pca)).to(device),
                "right_hand_pose": torch.zeros((N_frame, N_hand_pca)).to(device),
                "joints": torch.zeros((N_frame, 127, 3)).to(device),
                "vertices": torch.zeros((N_frame, 10475, 3)).to(device),
                "faces": smplx_model.faces_tensor.reshape(-1, 3),
            }
            for i in range(N_frame):
                human_params[person]["betas"][i] = raw_human_poses[i][person]["betas"][0].to(device)
                human_params[person]["expression"][i] = raw_human_poses[i][person]["expression"][0].to(device)
                human_params[person]["global_orient"][i] = raw_human_poses[i][person]["global_orient"][0].to(device)
                human_params[person]["transl"][i] = raw_human_poses[i][person]["transl"][0].to(device)
                human_params[person]["body_pose"][i] = raw_human_poses[i][person]["body_pose"][0].to(device)
                human_params[person]["left_hand_pose"][i] = raw_human_poses[i][person]["left_hand_pose"][0].to(device)
                human_params[person]["right_hand_pose"][i] = raw_human_poses[i][person]["right_hand_pose"][0].to(device)
            
            model = smplx_model(betas=human_params[person]["betas"], body_pose=human_params[person]["body_pose"], global_orient=human_params[person]["global_orient"], transl=human_params[person]["transl"], left_hand_pose=human_params[person]["left_hand_pose"], right_hand_pose=human_params[person]["right_hand_pose"])
            assert model.joints.shape == (N_frame, 127, 3)
            assert model.vertices.shape == (N_frame, 10475, 3)
            human_params[person]["joints"] = model.joints
            human_params[person]["vertices"] = model.vertices
        
        seq_data = {
            "N_frame": N_frame,  # int
            "obj_model_path": obj_model_path,  # obj file
            "obj_poses": obj_poses,  # shape = (N_frame, 4, 4)
            "human_params": human_params,  # {"person1": {...}, "person2": {}}
            "save_dir": join(save_root, clip_name, seq_name),
        }
        
        prepare_a_sequence(seq_data, num_samples)
    
    # json.dump(train_seq_list, open("/home/liuyun/HHO-dataset/dataset_statistics/retargeting_metadata/train_sequence_names_retargeted_0307.json", "w"))


def read_paired_data(fp):
    file_paths = []
    with open(fp, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            file_paths.append(line)
    return file_paths


def get_retargeting_meshes():
    
    file_paths = read_paired_data("/home/liuyun/HHO-dataset/data_processing/utils/Untitled-1.sh")
    print(len(file_paths))
    N = len(file_paths) // 4
    
    for seq_i in range(N):
        print("############# processing {} ... ############".format(seq_i))
        clip, seq = file_paths[seq_i*4+2].split("/")[-3:-1]
        seq_name = clip + "." + seq
        
        obj_model_path = file_paths[seq_i*4+3]
        raw_human_poses = np.load(file_paths[seq_i*4+0], allow_pickle=True)
        raw_obj_poses = np.load(file_paths[seq_i*4+1], allow_pickle=True)
        N_frame = len(raw_human_poses)
        
        # prepare obj poses
        obj_poses = []
        for i in range(N_frame):
            r = raw_obj_poses[i]["rotation"].reshape(1, 3)
            R = batch_rodrigues(r).reshape(3, 3).detach().cpu().numpy()
            t = raw_obj_poses[i]["translation"].detach().cpu().numpy().reshape(3)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            obj_poses.append(T)
        obj_poses = np.float32(obj_poses)
        
        # prepare human params
        N_betas = raw_human_poses[0]["person1"]["betas"].shape[1]
        N_expression = raw_human_poses[0]["person1"]["expression"].shape[1]
        N_hand_pca = raw_human_poses[0]["person1"]["left_hand_pose"].shape[1]
        human_params = {}
        smplx_model = create_SMPLX_model(batch_size=N_frame, device=device)
        
        for person in ["person1", "person2"]:
            human_params[person] = {
                "betas": torch.zeros((N_frame, N_betas), dtype=torch.float32).to(device),
                "expression": torch.zeros((N_frame, N_expression), dtype=torch.float32).to(device),
                "global_orient": torch.zeros((N_frame, 3), dtype=torch.float32).to(device),
                "transl": torch.zeros((N_frame, 3), dtype=torch.float32).to(device),
                "body_pose": torch.zeros((N_frame, 21, 3)).to(device),
                "left_hand_pose": torch.zeros((N_frame, N_hand_pca)).to(device),
                "right_hand_pose": torch.zeros((N_frame, N_hand_pca)).to(device),
                "joints": torch.zeros((N_frame, 127, 3)).to(device),
                "vertices": torch.zeros((N_frame, 10475, 3)).to(device),
                "faces": smplx_model.faces_tensor.reshape(-1, 3),
            }
            for i in range(N_frame):
                human_params[person]["betas"][i] = raw_human_poses[i][person]["betas"][0].to(device)
                human_params[person]["expression"][i] = raw_human_poses[i][person]["expression"][0].to(device)
                human_params[person]["global_orient"][i] = raw_human_poses[i][person]["global_orient"][0].to(device)
                human_params[person]["transl"][i] = raw_human_poses[i][person]["transl"][0].to(device)
                human_params[person]["body_pose"][i] = raw_human_poses[i][person]["body_pose"][0].to(device)
                human_params[person]["left_hand_pose"][i] = raw_human_poses[i][person]["left_hand_pose"][0].to(device)
                human_params[person]["right_hand_pose"][i] = raw_human_poses[i][person]["right_hand_pose"][0].to(device)
            
            model = smplx_model(betas=human_params[person]["betas"], body_pose=human_params[person]["body_pose"], global_orient=human_params[person]["global_orient"], transl=human_params[person]["transl"], left_hand_pose=human_params[person]["left_hand_pose"], right_hand_pose=human_params[person]["right_hand_pose"])
            assert model.joints.shape == (N_frame, 127, 3)
            assert model.vertices.shape == (N_frame, 10475, 3)
            human_params[person]["joints"] = model.joints
            human_params[person]["vertices"] = model.vertices
        
        # object mesh
        mesh_obj = Mesh()
        mesh_obj.load_from_obj(obj_model_path)
        obj_verts = mesh_obj.v
        obj_faces = mesh_obj.f
        obj = trimesh.Trimesh(obj_verts, obj_faces, process=False)
        
        save_dir = join("./to_chengwen_0315", seq_name)
        os.makedirs(save_dir, exist_ok=True)
        for i in range(N_frame):
            p1_verts = human_params["person1"]["vertices"][i].detach().cpu().numpy()  # (10475, 3)
            p1_faces = human_params["person1"]["faces"].detach().cpu().numpy()  # (?, 3)
            p1_mesh = trimesh.Trimesh(vertices=p1_verts, faces=p1_faces)
            save_mesh(p1_mesh, join(save_dir, str(i).zfill(5) + "_person1.obj"))
            p2_verts = human_params["person2"]["vertices"][i].detach().cpu().numpy()  # (10475, 3)
            p2_faces = human_params["person2"]["faces"].detach().cpu().numpy()  # (?, 3)
            p2_mesh = trimesh.Trimesh(vertices=p2_verts, faces=p2_faces)
            save_mesh(p2_mesh, join(save_dir, str(i).zfill(5) + "_person2.obj"))
            new_obj_verts = obj_verts.copy()  # (?, 3)
            obj_pose = obj_poses[i]
            new_obj_verts = new_obj_verts @ obj_pose[:3, :3].T + obj_pose[:3, 3]
            new_obj_mesh = trimesh.Trimesh(new_obj_verts, obj_faces, process=False)
            save_mesh(new_obj_mesh, join(save_dir, str(i).zfill(5) + "_object.obj"))
            


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_name', type=str, default="all")  # e.g.: 20231018.20231020
    parser.add_argument('--num_samples', type=int, default="2048")  # downsampled object point number
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #####################################################################
    dataset_root = "/share/datasets/hhodataset/VTS"
    obj_dataset_dir = "/data2/datasets/HHO_object_dataset_final_simplified"
    save_root = "/data2/datasets/hhodataset/prepared_motion_forecasting_data"
    device = "cuda:0"
    
    args = parse_args()
    if args.clip_name == "all":
        clip_names = ["20231002", "20231003_1", "20231003_2", "20231008", "20231011", "20231018", "20231020", "20231023", "20231030", "20231108"]
    else:
        clip_names = args.clip_name.split(".")
    #####################################################################

    # main(dataset_root, obj_dataset_dir, save_root, clip_names, args.num_samples, device)
    
    # main_retargeted_data_0307(save_root, args.num_samples, device)
    
    get_retargeting_meshes()
