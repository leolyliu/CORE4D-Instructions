import os
from os.path import join, dirname, abspath
import sys
sys.path.insert(0, join(dirname(abspath(__file__)), ".."))
import numpy as np
import torch
from optimization.bvh2smplx import create_SMPLX_model


def load_single_person_SMPLX_params(person_SMPLX_fitting_path, smplx_model_dir, start_frame, end_frame, device="cuda:0"):
    """
    frame range: [start_frame, end_frame)
    """
    smplx_model = create_SMPLX_model(smplx_model_dir=smplx_model_dir, device=device)
    N = end_frame - start_frame
    
    person_data = np.load(person_SMPLX_fitting_path, allow_pickle=True)["arr_0"].item()
    SMPLX_params = {
        "betas": torch.from_numpy(person_data["betas"][start_frame : end_frame]).to(dtype=torch.float32).to(device),  # (N, 10)
        "expression": torch.from_numpy(person_data["expression"][start_frame : end_frame]).to(dtype=torch.float32).to(device),  # (N, 10)
        "global_orient": torch.from_numpy(person_data["global_orient"][start_frame : end_frame]).to(dtype=torch.float32).to(device),  # (N, 3)
        "transl": torch.from_numpy(person_data["transl"][start_frame : end_frame]).to(dtype=torch.float32).to(device),  # (N, 3)
        "body_pose": torch.from_numpy(person_data["body_pose"][start_frame : end_frame]).to(dtype=torch.float32).to(device),  # (N, 21, 3)
        "left_hand_pose": torch.from_numpy(person_data["left_hand_pose"][start_frame : end_frame]).to(dtype=torch.float32).to(device),  # (N, 12)
        "right_hand_pose": torch.from_numpy(person_data["right_hand_pose"][start_frame : end_frame]).to(dtype=torch.float32).to(device),  # (N, 12)
        "joints": torch.from_numpy(person_data["joints"][start_frame : end_frame]).to(dtype=torch.float32).to(device),  # (N, 127, 3)
        "vertices": torch.from_numpy(person_data["vertices"][start_frame : end_frame]).to(dtype=torch.float32).to(device),  # (N, 10475, 3)
        "faces": smplx_model.faces_tensor.reshape(-1, 3),  # (N_face, 3)
    }
    
    return SMPLX_params


def load_multiperson_smplx_params(SMPLX_fitting_dir, smplx_model_dir, start_frame, end_frame, device="cuda:0") -> dict:
    multiperson_SMPLX_params = {
        "person1": load_single_person_SMPLX_params(join(SMPLX_fitting_dir, "person1_poses.npz"), smplx_model_dir, start_frame, end_frame, device=device),
        "person2": load_single_person_SMPLX_params(join(SMPLX_fitting_dir, "person2_poses.npz"), smplx_model_dir, start_frame, end_frame, device=device),
    }
    return multiperson_SMPLX_params
