import os
from os.path import join, isdir, dirname, abspath
import sys
sys.path.insert(0, join(dirname(abspath(__file__)), ".."))
import re
import numpy as np
import torch
# from smplx import smplx
from optimization.bvh2smplx import Simple_SMPLX, create_SMPLX_model
from utils.get_joints import get_joints_no_datatype_change
from smplx.smplx.utils import Struct, to_tensor, to_np

model_path = "/share/human_model/models/smplx/SMPLX_NEUTRAL.npz"
model_data = np.load(model_path, allow_pickle=True)
data_struct = Struct(**model_data)
g_shapedirs = data_struct.shapedirs
g_v_template = data_struct.v_template
g_J_regressor = data_struct.J_regressor
g_parents = data_struct.kintree_table[0]
g_left_hand_components = data_struct.hands_componentsl[:12]
g_right_hand_components = data_struct.hands_componentsr[:12]

def load_single_person_SMPLX_params(person_SMPLX_fitting_dir, start_frame, end_frame, N_betas=10, N_expression=10, N_hand_pca=12, device="cuda:0"):
    """
    frame range: [start_frame, end_frame)
    """
    smplx_model = create_SMPLX_model(device=device)
    N = end_frame - start_frame

    # shapedirs = to_tensor(to_np(g_shapedirs[:, :, :10]), dtype=torch.float32).to(device)
    # v_template = to_tensor(to_np(g_v_template), dtype=torch.float32).to(device)
    # J_regressor = to_tensor(to_np(g_J_regressor), dtype=torch.float32).to(device)
    # parents = to_tensor(to_np(g_parents)).long().to(device)
    # parents[0] = -1
    # extra_joint_parents = to_tensor(to_np([10, 10, 10, 11, 11, 11, 39, 27, 30, 36, 33, 54, 42, 45, 51, 48])).long().to(device)
    # parents = torch.cat([parents, extra_joint_parents], dim=0)
    # left_pca, right_pca = to_tensor(to_np(g_left_hand_components), dtype=torch.float32).to(device), to_tensor(to_np(g_right_hand_components), dtype=torch.float32).to(device)
        

    SMPLX_params = {
        "betas": torch.zeros((N, N_betas), dtype=torch.float32).to(device),
        "expression": torch.zeros((N, N_expression), dtype=torch.float32).to(device),
        "global_orient": torch.zeros((N, 3), dtype=torch.float32).to(device),
        "transl": torch.zeros((N, 3), dtype=torch.float32).to(device),
        "body_pose": torch.zeros((N, 21, 3)).to(device),
        "left_hand_pose": torch.zeros((N, N_hand_pca)).to(device),
        "right_hand_pose": torch.zeros((N, N_hand_pca)).to(device),
        "joints": torch.zeros((N, 127, 3)).to(device),
        # "joints": torch.zeros((N, 71, 3)).to(device),
        "vertices": torch.zeros((N, 10475, 3)).to(device),
        "faces": smplx_model.faces_tensor.reshape(-1, 3),
    }
    
    flag = np.zeros(N)  # existing data
    
    target_filename = re.compile("^[0-9]+to[0-9]+.npz$")  # [start]to[end].npz
    for fn in os.listdir(person_SMPLX_fitting_dir):
        if len(target_filename.findall(fn)) == 0:
            continue
        
        batch_L, batch_R = int(fn.split(".")[0].split("to")[0]), int(fn.split(".")[0].split("to")[1])
        L, R = max(batch_L, start_frame), min(batch_R, end_frame - 1)
        if L > R:
            continue
        
        batch_params = np.load(join(person_SMPLX_fitting_dir, fn), allow_pickle=True)["results"].item()  # read data
        
        flag[L - start_frame : R + 1 - start_frame] = 1
        SMPLX_params["betas"][L - start_frame : R + 1 - start_frame] = batch_params["betas"][L - batch_L : R + 1 - batch_L].to(device)
        SMPLX_params["body_pose"][L - start_frame : R + 1 - start_frame] = batch_params["body_pose"][L - batch_L : R + 1 - batch_L].to(device)
        SMPLX_params["transl"][L - start_frame : R + 1 - start_frame] = batch_params["transl"][L - batch_L : R + 1 - batch_L].to(device)
        SMPLX_params["global_orient"][L - start_frame : R + 1 - start_frame] = batch_params["global_orient"][L - batch_L : R + 1 - batch_L].to(device)
        SMPLX_params["left_hand_pose"][L - start_frame : R + 1 - start_frame] = batch_params["left_hand_pose"][L - batch_L : R + 1 - batch_L].to(device)
        SMPLX_params["right_hand_pose"][L - start_frame : R + 1 - start_frame] = batch_params["right_hand_pose"][L - batch_L : R + 1 - batch_L].to(device)
    
    assert flag.min() > 0, "[load_single_person_SMPLX_params] error: incomplete data!, dir = {}".format(person_SMPLX_fitting_dir)
    batch_joints = torch.zeros((N, 127, 3)).to(device)
    # batch_joints = torch.zeros((N, 71, 3)).to(device)
    batch_vertices = torch.zeros((N, 10475, 3)).to(device)
    for i in range(N):
        betas = SMPLX_params["betas"].unsqueeze(1)
        body_pose = SMPLX_params["body_pose"].unsqueeze(1)
        transl = SMPLX_params["transl"].unsqueeze(1)
        global_orient = SMPLX_params["global_orient"].unsqueeze(1)
        left_hand_pose = SMPLX_params["left_hand_pose"].unsqueeze(1)
        right_hand_pose = SMPLX_params["right_hand_pose"].unsqueeze(1)
        model = smplx_model(betas=betas[i, :], body_pose=body_pose[i, :, :], global_orient=global_orient[i, :], transl=transl[i, :], left_hand_pose=left_hand_pose[i, :], right_hand_pose=right_hand_pose[i, :])
        # batch_joints[i, :, :] = get_joints_no_datatype_change(global_orient[i, :, :], betas[i, :, :], body_pose[i, :, :], transl[i, :, :], left_hand_pose[i, :, :], right_hand_pose[i, :, :], left_pca, right_pca, shapedirs, v_template, J_regressor, parents, device).detach().clone()
        batch_joints[i, :, :] = model.joints.detach().clone()
        batch_vertices[i, :, :] = model.vertices.detach().clone()
    SMPLX_params["joints"] = batch_joints.squeeze(1)
    SMPLX_params["vertices"] = batch_vertices.squeeze(1)
    return SMPLX_params


def load_multiperson_smplx_params(SMPLX_fitting_dir, start_frame, end_frame, device="cuda:0") -> dict:
    """
    frame range: [start_frame, end_frame)
    """
    
    multiperson_SMPLX_params = {
        "person1": None,
        "person2": None,
    }
    
    person_SMPLX_fitting_dir = join(SMPLX_fitting_dir, "person_1")
    assert isdir(person_SMPLX_fitting_dir)
    multiperson_SMPLX_params["person1"] = load_single_person_SMPLX_params(person_SMPLX_fitting_dir, start_frame, end_frame, device=device)
    
    person_SMPLX_fitting_dir = join(SMPLX_fitting_dir, "person_2")
    assert isdir(person_SMPLX_fitting_dir)
    multiperson_SMPLX_params["person2"] = load_single_person_SMPLX_params(person_SMPLX_fitting_dir, start_frame, end_frame, device=device)
    
    return multiperson_SMPLX_params
