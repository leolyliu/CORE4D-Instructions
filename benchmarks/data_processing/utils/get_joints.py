import os
from os.path import join, isfile
import sys
sys.path.append("..")
import argparse
import numpy as np
import pickle
import torch
from torch import nn
import pytorch3d
import pytorch3d.io as IO
import trimesh
from smplx import smplx
from smplx.smplx.utils import Struct, to_tensor, to_np
from smplx.smplx.lbs import batch_rodrigues, vertices2joints, blend_shapes, batch_rigid_transform


# EXTRA_JOINTS_TPOSE_POS = [[ 0.0971, -1.3538,  0.1476],
#         [ 0.1679, -1.3602,  0.0984],
#         [ 0.1060, -1.3345, -0.1217],
#         [-0.0886, -1.3553,  0.1455],
#         [-0.1679, -1.3606,  0.0984],
#         [-0.1079, -1.3518, -0.1196],
#         [ 0.8220,  0.0142,  0.0299],
#         [ 0.9067,  0.0314, -0.0415],
#         [ 0.9161,  0.0359, -0.0833],
#         [ 0.8977,  0.0338, -0.1137],
#         [ 0.8562,  0.0220, -0.1453],
#         [-0.8185,  0.0165,  0.0320],
#         [-0.9067,  0.0314, -0.0415],
#         [-0.9160,  0.0358, -0.0833],
#         [-0.8975,  0.0338, -0.1136],
#         [-0.8562,  0.0220, -0.1453]]

EXTRA_JOINTS_TPOSE_POS = [[ 0.0879, -1.2864,  0.1382],
        [ 0.1552, -1.2937,  0.0906],
        [ 0.0963, -1.2660, -0.1178],
        [-0.0792, -1.2880,  0.1362],
        [-0.1552, -1.2941,  0.0907],
        [-0.0981, -1.2824, -0.1157],
        [ 0.7719,  0.0133,  0.0295],
        [ 0.8522,  0.0285, -0.0387],
        [ 0.8614,  0.0326, -0.0786],
        [ 0.8435,  0.0310, -0.1072],
        [ 0.8038,  0.0208, -0.1384],
        [-0.7685,  0.0155,  0.0315],
        [-0.8522,  0.0285, -0.0387],
        [-0.8613,  0.0326, -0.0786],
        [-0.8433,  0.0310, -0.1072],
        [-0.8038,  0.0208, -0.1384]]


def get_joints(global_orient, betas, pose, transl, left_hand_pose, right_hand_pose, left_pca, right_pca, shapedirs, v_template, J_regressor, parents, device="cuda:0"):
    # model_path = "/share/human_model/models/smplx/SMPLX_NEUTRAL.npz""

    # ------------------------------------------------------------
    batch_size = betas.shape[0]
    shapedirs = to_tensor(to_np(shapedirs[:, :, :betas.shape[1]]), dtype=torch.float32).to(device)
    if not torch.is_tensor(v_template):
        v_template = to_tensor(to_np(v_template), dtype=torch.float32).to(device)
    
    J_regressor = to_tensor(to_np(J_regressor), dtype=torch.float32).to(device)
    parents = to_tensor(to_np(parents)).long().to(device)
    parents[0] = -1
    extra_joint_parents = to_tensor(to_np([10, 10, 10, 11, 11, 11, 39, 27, 30, 36, 33, 54, 42, 45, 51, 48])).long().to(device)
    parents = torch.cat([parents, extra_joint_parents], dim=0)
    
    global_orient = global_orient.unsqueeze(1)
    extra_pose = torch.zeros([batch_size, 3, 3]).to(device)
    left_pca, right_pca = to_tensor(to_np(left_pca), dtype=torch.float32).to(device), to_tensor(to_np(right_pca), dtype=torch.float32).to(device)
    left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, left_pca]).reshape(-1, 15, 3)
    right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, right_pca]).reshape(-1, 15, 3)
    extra_pose_2 = torch.zeros([batch_size, 16, 3]).to(device)

    full_pose = torch.cat([global_orient, pose, extra_pose, left_hand_pose, right_hand_pose, extra_pose_2], dim=1) # B x 71 x 3
    
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])


    # ------------------------------------------------------------

    v_shape = v_template + blend_shapes(betas, shapedirs) # B x 10475 x 3
    J = vertices2joints(J_regressor, v_shape) # B x 55 x 3
    # 需要将J的坐标系转换到全局坐标系下 并且添加16个新的坐标 分别是10个手指尖和6个脚趾
    # 56-65: 10个手指尖, 66-71: 6个脚趾
    extra_J = torch.tensor(EXTRA_JOINTS_TPOSE_POS).to(device)
    extra_J = extra_J.unsqueeze(0).repeat(batch_size, 1, 1)
    J = torch.cat([J, extra_J], dim=1) # B x 71 x 3
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents) # B x 71 x 3, B x 71 x 3 x 3
    J_transformed += transl.unsqueeze(1) # B x 71 x 3
    return J_transformed


def get_joints_no_datatype_change(global_orient, betas, pose, transl, left_hand_pose, right_hand_pose, left_pca, right_pca, shapedirs, v_template, J_regressor, parents, device="cuda:0"):
    """
    copy from get_joints
    """
    
    batch_size = betas.shape[0]
    
    global_orient = global_orient.unsqueeze(1)
    extra_pose = torch.zeros([batch_size, 3, 3]).to(device)
    left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, left_pca]).reshape(-1, 15, 3)
    right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, right_pca]).reshape(-1, 15, 3)
    extra_pose_2 = torch.zeros([batch_size, 16, 3]).to(device)

    full_pose = torch.cat([global_orient, pose, extra_pose, left_hand_pose, right_hand_pose, extra_pose_2], dim=1) # B x 71 x 3
    
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])

    # ------------------------------------------------------------

    v_shape = v_template + blend_shapes(betas, shapedirs) # B x 10475 x 3
    J = vertices2joints(J_regressor, v_shape) # B x 55 x 3
    # 需要将J的坐标系转换到全局坐标系下 并且添加16个新的坐标 分别是10个手指尖和6个脚趾
    # 56-65: 10个手指尖, 66-71: 6个脚趾
    extra_J = torch.tensor(EXTRA_JOINTS_TPOSE_POS).to(device)
    extra_J = extra_J.unsqueeze(0).repeat(batch_size, 1, 1)
    J = torch.cat([J, extra_J], dim=1) # B x 71 x 3
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents) # B x 71 x 3, B x 71 x 3 x 3
    J_transformed += transl.unsqueeze(1) # B x 71 x 3
    return J_transformed
