import json
import os
from os.path import join, dirname, abspath, isfile, isdir
import sys
sys.path.insert(0, join(dirname(abspath(__file__)), ".."))
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import yaml
from knn_cuda import KNN
from data.tools import vertex_normals
from data.utils import markerset_ssm67_smplh, markerset_wfinger
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
sys.path.insert(0, join(dirname(abspath(__file__)), "../../../.."))
from data_processing.smplx import smplx
import open3d as o3d


def compute_contact_and_closest_point(source_vertices, target_vertices, threshould=0.05):
    """
    source_vertices: torch.tensor, shape = (N, 3)
    target_vertices: torch.tensor, shape = (M, 3)
    
    return:
    contact: torch.tensor, shape = (N), bool
    dist: torch.tensor, shape = (N)
    closest_point: torch.tensor, shape = (N)
    """
    # knn = nn.DataParallel(KNN(k=1, transpose_mode=True), device_ids=[1, 2])
    knn = KNN(k=1, transpose_mode=True)
    # source_to_target contact
    # print(target_vertices.type, source_vertices.type())
    dist, closest_point = knn(target_vertices.unsqueeze(0), source_vertices.unsqueeze(0))
    # print(source_vertices.shape, target_vertices.shape)
    # dist, closest_point = knn(target_vertices, source_vertices)
    dist = dist.squeeze(dim=0).squeeze(dim=-1)
    contact = dist < threshould
    
    closest_point = closest_point[0, :, 0]
    
    return contact, dist, closest_point


def ICP(A, B):
    A_pcd = o3d.t.geometry.PointCloud()
    A_pcd.point.positions = o3d.core.Tensor(A, o3d.core.float32, o3d.core.Device("CPU:0"))
    B_pcd = o3d.t.geometry.PointCloud()
    B_pcd.point.positions = o3d.core.Tensor(B, o3d.core.float32, o3d.core.Device("CPU:0"))
    result = o3d.t.pipelines.registration.icp(A_pcd, B_pcd, 0.05)
    print(result)
    T = result.transformation
    return T.numpy()  # A作用T之后和B对应


if __name__ == "__main__":
    smplh_male = SMPL_Layer(center_idx=0, gender='male', num_betas=10, model_root=str("/share/human_model/smplh"), hands=True)
    smplx_male = smplx.create("/share/human_model/models", model_type="smplx", gender="male", batch_size=1, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=12, flat_hand_mean=True)

    smplh_Tpose_verts, smplh_Tpose_jts, _, _ = smplh_male(torch.zeros(1, 66+45*2), th_betas=torch.zeros(1, 10), th_trans=torch.zeros(1, 3))  # shape = (1, 6890, 3)
    smplh_Tpose_root = smplh_Tpose_jts[:, 0]  # shape = (1, 3)
    smplx_Tpose = smplx_male(body_pose=torch.zeros(1, 21, 3), betas=torch.zeros(1, 10), global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3))
    smplx_Tpose_verts = smplx_Tpose.vertices  # shape = (1, 10475, 3)
    smplx_Tpose_root = smplx_Tpose.joints[:, 0]  # shape = (1, 3)
    
    smplh_Tpose_verts = smplh_Tpose_verts[0].detach().cpu().numpy()
    smplh_Tpose_verts -= smplh_Tpose_root.detach().cpu().numpy()
    smplx_Tpose_verts = smplx_Tpose_verts[0].detach().cpu().numpy()
    smplx_Tpose_verts -= smplx_Tpose_root.detach().cpu().numpy()
    
    N_smplh = smplh_Tpose_verts.shape[0]
    N_smplx = smplx_Tpose_verts.shape[0]
    
    smplh_left_arm_flag = (smplh_Tpose_verts[:, 0] > 0.2) & (smplh_Tpose_verts[:, 1] > 0.2)
    smplh_left_arm = smplh_Tpose_verts[smplh_left_arm_flag]
    smplx_left_arm_flag = (smplx_Tpose_verts[:, 0] > 0.2) & (smplx_Tpose_verts[:, 1] > 0.2)
    smplx_left_arm = smplx_Tpose_verts[smplx_left_arm_flag]
    smplh_right_arm_flag = (smplh_Tpose_verts[:, 0] < -0.2) & (smplh_Tpose_verts[:, 1] > 0.2)
    smplh_right_arm = smplh_Tpose_verts[smplh_right_arm_flag]
    smplx_right_arm_flag = (smplx_Tpose_verts[:, 0] < -0.2) & (smplx_Tpose_verts[:, 1] > 0.2)
    smplx_right_arm = smplx_Tpose_verts[smplx_right_arm_flag]
    print(smplh_left_arm.shape, smplx_left_arm.shape, smplh_right_arm.shape, smplx_right_arm.shape)
    
    # 1: body, 2: left_arm, 3: right_arm
    smplh_info = [[]] * N_smplh
    smplh_left_arm_ids = list(np.where(smplh_left_arm_flag)[0])
    smplh_left_arm_dict = {}
    for i in range(len(smplh_left_arm_ids)):
        smplh_left_arm_dict[smplh_left_arm_ids[i]] = i
    smplh_right_arm_ids = list(np.where(smplh_right_arm_flag)[0])
    smplh_right_arm_dict = {}
    for i in range(len(smplh_right_arm_ids)):
        smplh_right_arm_dict[smplh_right_arm_ids[i]] = i
    for i in range(N_smplh):
        if smplh_left_arm_flag[i]:
            smplh_info[i] = [2, smplh_left_arm_dict[i]]
        elif smplh_right_arm_flag[i]:
            smplh_info[i] = [3, smplh_right_arm_dict[i]]
        else:
            smplh_info[i] = [1, i]
    smplx_info = {1: list(np.arange(0, N_smplx)), 2: list(np.where(smplx_left_arm_flag)[0]), 3: list(np.where(smplx_right_arm_flag)[0])}
    
    T = ICP(smplh_left_arm, smplx_left_arm)
    aligned_smplh_left_arm = smplh_left_arm @ T[:3, :3].T + T[:3, 3]
    T = ICP(smplh_right_arm, smplx_right_arm)
    aligned_smplh_right_arm = smplh_right_arm @ T[:3, :3].T + T[:3, 3]
    
    _, _, left_arm_closest_ids = compute_contact_and_closest_point(torch.from_numpy(aligned_smplh_left_arm).to("cuda:0"), torch.from_numpy(smplx_left_arm).to("cuda:0"))
    _, _, right_arm_closest_ids = compute_contact_and_closest_point(torch.from_numpy(aligned_smplh_right_arm).to("cuda:0"), torch.from_numpy(smplx_right_arm).to("cuda:0"))
    _, _, body_closest_ids = compute_contact_and_closest_point(torch.from_numpy(smplh_Tpose_verts).to("cuda:0"), torch.from_numpy(smplx_Tpose_verts).to("cuda:0"))
    
    markerset_ssm67_smplx = []
    for idx in markerset_ssm67_smplh:
        flag, local_idx = smplh_info[idx]
        if flag == 1:
            x = smplx_info[flag][body_closest_ids[local_idx]]
        elif flag == 2:
            x = smplx_info[flag][left_arm_closest_ids[local_idx]]
        else:
            x = smplx_info[flag][right_arm_closest_ids[local_idx]]
        markerset_ssm67_smplx.append(x)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(smplh_Tpose_verts)
    o3d.io.write_point_cloud("./smplh_male_Tpose.ply", pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(smplx_Tpose_verts)
    o3d.io.write_point_cloud("./smplx_male_Tpose.ply", pcd)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(smplh_Tpose_verts[markerset_ssm67_smplh])
    o3d.io.write_point_cloud("./smplh_male_Tpose_markers67.ply", pcd)
    
    print(markerset_ssm67_smplx)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(smplx_Tpose_verts[markerset_ssm67_smplx])
    o3d.io.write_point_cloud("./smplx_male_Tpose_markers67.ply", pcd)
