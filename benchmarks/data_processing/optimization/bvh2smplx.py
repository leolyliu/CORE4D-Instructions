import os
from os.path import join, dirname, abspath
import sys
sys.path.insert(0, join(dirname(abspath(__file__)), "../.."))
import copy
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from data_processing.smplx import smplx
from data_processing.smplx.smplx.lbs import batch_rodrigues
from transforms3d.axangles import mat2axangle
from data_processing.utils.bvh2joint import bvh2joint, get_joint_data
from data_processing.utils.scale import get_scale_from_data, get_constraints_for_optimizing_shape
from data_processing.utils.pyt3d_wrapper import Pyt3DWrapper
from data_processing.utils.visualization import read_data_from_SMPLX, render_SMPLX, save_pcd
from data_processing.utils.index_corr import smplx2bvh_for_optimizing_smplx
from data_processing.utils.pose_range import get_pose_range
from data_processing.utils.process_transformation import np_mat2axangle
import trimesh
import open3d as o3d


class Simple_SMPLX(nn.Module):
    def __init__(self, smplx_model, init_smplx_params, cfg):
        super(Simple_SMPLX, self).__init__()
        self.smplx_model = smplx_model
        self.smplx_betas = nn.Parameter(init_smplx_params["betas"].clone().detach(), requires_grad=False)
        self.smplx_expression = nn.Parameter(init_smplx_params["expression"].clone().detach(), requires_grad=False)
        self.smplx_global_orient = nn.Parameter(init_smplx_params["global_orient"].clone().detach(), requires_grad=False)
        self.smplx_body_pose = nn.Parameter(init_smplx_params["body_pose"].clone().detach(), requires_grad=False)
        self.smplx_left_hand_pose = nn.Parameter(init_smplx_params["left_hand_pose"].clone().detach(), requires_grad=False)
        self.smplx_right_hand_pose = nn.Parameter(init_smplx_params["right_hand_pose"].clone().detach(), requires_grad=False)
        self.smplx_transl = nn.Parameter(init_smplx_params["transl"].clone().detach(), requires_grad=False)
        self.smplx_jaw_pose = nn.Parameter(torch.zeros([self.smplx_betas.shape[0], 3], dtype=torch.float32), requires_grad=False)
        self.smplx_leye_pose = nn.Parameter(torch.zeros([self.smplx_betas.shape[0], 3], dtype=torch.float32), requires_grad=False)
        self.smplx_reye_pose = nn.Parameter(torch.zeros([self.smplx_betas.shape[0], 3], dtype=torch.float32), requires_grad=False)

        if cfg["OPT_BETAS"]:
            self.smplx_betas.requires_grad = True
        if cfg["OPT_EXPRESSION"]:
            self.smplx_expression.requires_grad = True
        if cfg["OPT_POSE"]:
            self.smplx_global_orient.requires_grad = True
            self.smplx_transl.requires_grad = True
            self.smplx_body_pose.requires_grad = True
            self.smplx_left_hand_pose.requires_grad = True
            self.smplx_right_hand_pose.requires_grad = True
        self.cfg = cfg
    
    def forward(self):
        result_model = self.smplx_model(betas=self.smplx_betas, expression=self.smplx_expression, global_orient=self.smplx_global_orient, transl=self.smplx_transl, body_pose=self.smplx_body_pose, left_hand_pose=self.smplx_left_hand_pose, right_hand_pose=self.smplx_right_hand_pose, jaw_pose=self.smplx_jaw_pose, leye_pose=self.smplx_leye_pose, reye_pose=self.smplx_reye_pose, return_verts=True)
        result_vertices = result_model.vertices
        result_joints = result_model.joints
        results = {
            "vertices": result_vertices,
            "joints": result_joints,
            "betas": self.smplx_betas,
            "expression": self.smplx_expression,
            "global_orient": self.smplx_global_orient,
            "transl": self.smplx_transl,
            "body_pose": self.smplx_body_pose,
            "left_hand_pose": self.smplx_left_hand_pose,
            "right_hand_pose": self.smplx_right_hand_pose,
        }
        return results


def test_smplx():
    num_pca_comps = 12  # hand PCA dimension
    smplx_model = smplx.create("/share/human_model/models", model_type="smplx", gender="neutral", use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=num_pca_comps, flat_hand_mean=True)
    smplx_model.to("cuda:0")
    init_smplx_params = {
        "betas": torch.zeros([1, smplx_model.num_betas], dtype=torch.float32).to("cuda:0"),
        "expression": torch.zeros([1, smplx_model.num_expression_coeffs], dtype=torch.float32).to("cuda:0"),
        "global_orient": torch.zeros([1, 3], dtype=torch.float32).to("cuda:0"),
        "transl": torch.zeros([1, 3], dtype=torch.float32).to("cuda:0"),
        "body_pose": torch.zeros([1, smplx_model.NUM_BODY_JOINTS, 3]).to("cuda:0"),
        "left_hand_pose": torch.zeros([1, num_pca_comps]).to("cuda:0"),
        "right_hand_pose": torch.zeros([1, num_pca_comps]).to("cuda:0"),
    }
    cfg = {
        "OPT_BETAS": True,
        "OPT_POSE": False,
        "OPT_EXPRESSION": False,
    }

    optim_model = Simple_SMPLX(smplx_model=smplx_model, init_smplx_params=init_smplx_params, cfg=cfg)
    optim_model.to("cuda:0")
    results = optim_model()
    j = results["joints"]
    for i in range(127):
        print(i, j[0][i].detach().cpu().numpy())

    save_pcd("./ex.ply", j[0].detach().cpu().numpy())


def create_SMPLX_model(use_pca=True, num_pca_comps=12, batch_size=1, device="cuda:0"):
    """
    create an SMPLX model
    num_pca_comps: hand PCA dimension
    """
    smplx_model = smplx.create("/share/human_model/models", model_type="smplx", gender="neutral", batch_size=batch_size, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=use_pca, num_pca_comps=num_pca_comps, flat_hand_mean=True)
    smplx_model.to(device)
    return smplx_model


def init_SMPLX_pose_from_bvh_pelvis(smplx_model, betas, bvh_pelvis_rot, use_pca=True, num_pca_comps=12, device="cuda:0"):
    # init, 初始化极其重要, 至少初始化一个大概的global 6D pose! (EasyMocap: 先优化出大致的global 6D pose再优化theta)
    if not use_pca:
        raise NotImplementedError
    betas = torch.from_numpy(betas).reshape([1, smplx_model.num_betas]).to(device)
    init_smplx_params = {
        "betas": betas,
        "expression": torch.zeros([1, smplx_model.num_expression_coeffs], dtype=torch.float32).to(device),
        "global_orient": torch.zeros([1, 3], dtype=torch.float32).to(device),
        "transl": torch.zeros([1, 3], dtype=torch.float32).to(device),
        "body_pose": torch.zeros([1, smplx_model.NUM_BODY_JOINTS, 3]).to(device),
        "left_hand_pose": torch.zeros([1, num_pca_comps]).to(device),
        "right_hand_pose": torch.zeros([1, num_pca_comps]).to(device),
    }
    T = np.eye(4)
    T[1, 3] = 0.40  # SMPLX TPOSE下pelvis到(0,0,0)的平移
    pelvis2world = bvh_pelvis_rot @ T
    init_smplx_params["transl"] = torch.from_numpy(pelvis2world[:3, 3].reshape(1, 3).astype(np.float32)).to(device)
    d, a = mat2axangle(pelvis2world[:3, :3])
    init_smplx_params["global_orient"] = torch.from_numpy((d * a).reshape(1, 3).astype(np.float32)).to(device)

    return init_smplx_params


def optimize_shape(human_scale, from_bvh=False):
    constraints, human_scale = get_constraints_for_optimizing_shape(human_scale, from_bvh=from_bvh)
    
    num_pca_comps = 12  # hand PCA dimension
    smplx_model = smplx.create("/share/human_model/models", model_type="smplx", gender="neutral", use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=num_pca_comps, flat_hand_mean=True)
    smplx_model.to("cuda:0")
    init_smplx_params = {
        "betas": torch.zeros([1, smplx_model.num_betas], dtype=torch.float32).to("cuda:0"),
        "expression": torch.zeros([1, smplx_model.num_expression_coeffs], dtype=torch.float32).to("cuda:0"),
        "global_orient": torch.zeros([1, 3], dtype=torch.float32).to("cuda:0"),
        "transl": torch.zeros([1, 3], dtype=torch.float32).to("cuda:0"),
        "body_pose": torch.zeros([1, smplx_model.NUM_BODY_JOINTS, 3]).to("cuda:0"),
        "left_hand_pose": torch.zeros([1, num_pca_comps]).to("cuda:0"),
        "right_hand_pose": torch.zeros([1, num_pca_comps]).to("cuda:0"),
    }
    cfg = {
        "OPT_BETAS": True,
        "OPT_POSE": False,
        "OPT_EXPRESSION": False,
    }

    constraints = np.int32(constraints).transpose(1, 0)
    start = torch.LongTensor(constraints[0]).to("cuda:0")
    end = torch.LongTensor(constraints[1]).to("cuda:0")
    human_scale = torch.FloatTensor(human_scale).to("cuda:0")

    optim_model = Simple_SMPLX(smplx_model=smplx_model, init_smplx_params=init_smplx_params, cfg=cfg)
    optim_model.to("cuda:0")
    optimizer = torch.optim.Adam(optim_model.parameters(), lr=0.001)
    optim_model.train()
    for epoch in range(5001):
        optimizer.zero_grad()
        results = optim_model()
        j, betas = results["joints"], results["betas"]
        if epoch % 100 == 0:
            print((j[0, start] - j[0, end]).norm(dim=1), human_scale)
            print("betas =", results["betas"])
        loss = torch.sum(((j[0, start] - j[0, end]).norm(dim=1) - human_scale)**2) * 1.0 + torch.sum(betas**2) * 5e-4
        print(epoch, loss.item())
        loss.backward()
        optimizer.step()
    
    optim_model.eval()
    results = optim_model()
    return results["betas"]


def optimize_pose_singleframe(joint_data, bvh_rot, betas, frame_idx=0, N_epoch=10000, save_dir=None, device="cuda:0"):
    """
    return: single frame SMPLX params
    """
    
    # get pre-defined constraints
    assert joint_data.shape[1] == 74
    constraints_nohand, constraints_hand, comp_constraints, direction_constraints = smplx2bvh_for_optimizing_smplx(N=joint_data.shape[1], split_hand=True)

    # create an SMPLX model
    use_pca, num_pca_comps = True, 12
    smplx_model = create_SMPLX_model(use_pca=use_pca, num_pca_comps=num_pca_comps, device=device)

    # init SMPLX pose from bvh pelvis pose
    init_smplx_params = init_SMPLX_pose_from_bvh_pelvis(smplx_model, betas, bvh_rot[frame_idx][0], use_pca=use_pca, num_pca_comps=12, device=device)

    cfg = {
        "OPT_BETAS": False,
        "OPT_POSE": True,
        "OPT_EXPRESSION": False,
    }

    # prepare constraints
    # point-to-point constraints
    constraints_nohand = np.int32(constraints_nohand).transpose(1, 0)
    idx_nohand = torch.LongTensor(constraints_nohand[0]).to(device)
    idx_bvh_skeleton = constraints_nohand[1]
    gt_pos_nohand = torch.from_numpy(joint_data[frame_idx][idx_bvh_skeleton]).to(device)
    constraints_hand = np.int32(constraints_hand).transpose(1, 0)
    idx_hand = torch.LongTensor(constraints_hand[0]).to(device)
    idx_bvh_skeleton = constraints_hand[1]
    gt_pos_hand = torch.from_numpy(joint_data[frame_idx][idx_bvh_skeleton]).to(device)
    # multi_point-to-point constraints
    comp_constraint_weights = np.zeros((len(comp_constraints), 127)).astype(np.float32)  # number of SMPLX joint = 127
    idx_bvh_skeleton = []
    for i, cc in enumerate(comp_constraints):
        for item in cc[0]:
            comp_constraint_weights[i, item[1]] = item[0]
        assert comp_constraint_weights[i].sum() == 1.0
        idx_bvh_skeleton.append(cc[1])
    comp_constraint_weights = torch.from_numpy(comp_constraint_weights).to(device)
    gt_pos_for_comp = torch.from_numpy(joint_data[frame_idx][idx_bvh_skeleton]).to(device)
    # direction-to-direction constraints
    direction_constraint_ids = [[], []]
    gt_direction = []
    for dc in direction_constraints:
        direction_constraint_ids[0].append(dc[0][0])
        direction_constraint_ids[1].append(dc[0][1])
        gt_d = joint_data[frame_idx][dc[1][0]] - joint_data[frame_idx][dc[1][1]]
        gt_d /= max(np.linalg.norm(gt_d), 1e-8)
        gt_direction.append(gt_d)
    gt_direction = torch.from_numpy(np.float32(gt_direction)).to(device)

    pose_range = get_pose_range()
    pose_range = torch.from_numpy(pose_range).to(device)

    # optimize a frame
    optim_model = Simple_SMPLX(smplx_model=smplx_model, init_smplx_params=init_smplx_params, cfg=cfg)
    optim_model.to(device)
    optimizer = torch.optim.Adam(optim_model.parameters(), lr=0.01)
    optim_model.train()

    print("------------ optimizing single frame {} -----------------------".format(str(frame_idx)))
    for epoch in range(N_epoch):
        optimizer.zero_grad()
        results = optim_model()
        j, body_pose, left_hand_pose, right_hand_pose = results["joints"], results["body_pose"], results["left_hand_pose"], results["right_hand_pose"]
        # print(body_pose.shape, left_hand_pose.shape, right_hand_pose.shape)  # (1, 21, 3), (1, 12), (1, 12)
        
        # loss
        # joint3D_loss = point-to-point loss + multi_point-to-point loss + direction-to-direction loss
        joint3D_loss_nohand = torch.sum((j[0, idx_nohand] - gt_pos_nohand)**2) * 1.0
        joint3D_loss_nohand += torch.sum((torch.mm(comp_constraint_weights, j[0]) - gt_pos_for_comp)**2) * 1.0
        joint3D_loss_hand = torch.sum((j[0, idx_hand] - gt_pos_hand)**2) * 1.0
        # direction = j[0, direction_constraint_ids[0]] - j[0, direction_constraint_ids[1]]
        # direction = direction / torch.clamp(torch.norm(direction, dim=-1).unsqueeze(-1), 1e-8, None)
        # joint3D_loss += torch.sum(1 - torch.sum(direction * gt_direction, dim=-1)) * 0.01

        regularizer = torch.sum(body_pose**2) * 2e-4 + torch.sum(left_hand_pose**2) * 1e-4 + torch.sum(right_hand_pose**2) * 1e-4
        # twist loss包含太多关节或range_loss过大都会让训练容易陷入local minima
        twist_loss = torch.sum((body_pose[:, [9,10,12,13,15,16,17,18,19,20], 0]**2) + torch.sum(body_pose[:, [0,1,2,3,4,5,8,9,10,11,14], 1]**2) + torch.sum(body_pose[:, [9, 10], 2]**2)) * 1e-1  # 惩罚几个关节绕轴的自转, 权重>1e-2
        spine_loss = -torch.sum(body_pose[:, [2,5,8,11], 0].clamp(None, 0)) * 1e-1 + torch.sum((body_pose[:, [2,5,8,11]] - body_pose[:, [2,5,8,11]].mean(dim=1).unsqueeze(1).expand(body_pose[:, [2,5,8,11]].shape))**2) * 1e-1  # 由于bvh对脊柱约束较少, 这里额外约束脊柱不能后仰, 以及转动要平滑
        range_loss = torch.sum((body_pose**2 - pose_range**2).clamp(0, None)) * 1e-1  # 软约束: 要求各个关节rotation不能出界

        loss = joint3D_loss_nohand + regularizer + twist_loss + spine_loss + joint3D_loss_hand + range_loss
        # if epoch > N_epoch * 0.5:
        #     loss += range_loss
        # if epoch > N_epoch * 0.75:
        #     loss += joint3D_loss_hand
        if epoch % 100 == 0:
            print(epoch, loss.item(), joint3D_loss_nohand.item(), regularizer.item(), twist_loss.item(), spine_loss.item(), range_loss.item(), joint3D_loss_hand.item())
            # faces = smplx_model.faces_tensor.detach().cpu().numpy()
            # mesh = trimesh.Trimesh(vertices=results["vertices"][0].detach().cpu().numpy(), faces=faces)
            # mesh_txt = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
            # with open(join(save_dir, str(frame_idx) + "_SMPLX_mesh_epoch_{}.obj".format(str(epoch))), "w") as fp:
            #     fp.write(mesh_txt)
        loss.backward()
        optimizer.step()
    
    # get SMPLX result
    optim_model.eval()
    results = optim_model()
    smplx_params = {
        "betas": results["betas"].detach(),
        "expression": results["expression"].detach(),
        "global_orient": results["global_orient"].detach(),
        "transl": results["transl"].detach(),
        "body_pose": results["body_pose"].detach(),
        "left_hand_pose": results["left_hand_pose"].detach(),
        "right_hand_pose": results["right_hand_pose"].detach(),
    }

    # save and visualize the result
    if not save_dir is None:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(join(save_dir, str(frame_idx) + ".npz"), results=results)  # save params
        save_pcd(join(save_dir, str(frame_idx) + "_bvh_joints.ply"), joint_data[frame_idx])
        save_pcd(join(save_dir, str(frame_idx) + "_SMPLX_joints.ply"), results["joints"][0].detach().cpu().numpy())
        faces = smplx_model.faces_tensor.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=results["vertices"][0].detach().cpu().numpy(), faces=faces)
        mesh_txt = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
        with open(join(save_dir, str(frame_idx) + "_SMPLX_mesh.obj"), "w") as fp:
            fp.write(mesh_txt)

    return smplx_params


def optimize_pose_sequence(data_path, person_id, betas, start_frame_idx=0, end_frame_idx=1000, end_link_trans=None, save_dir=None, cfg={}, device="cuda:0", selected_frames=None):
    """
    cfg: 各个loss项的权重

    return SMPLX params for each frame in the sequence, range: [start_frame_idx, end_frame_idx)
    """

    print("start: [optimize_pose_sequence]")
    
    # read gt bvh data
    if selected_frames is None:  # fake case
        sampling_rate = 3
        joint_data, bvh_rot = get_joint_data(data_path, person_id=person_id, frame_ids=None, end_link_trans=end_link_trans, return_local_rot=True, selected_frames=None, sampling_rate=sampling_rate)
        print("frame number = ", joint_data.shape[0], ", sampling rate = 1 /", sampling_rate)
    else:
        joint_data, bvh_rot = get_joint_data(data_path, person_id=person_id, frame_ids=None, end_link_trans=end_link_trans, return_local_rot=True, selected_frames=selected_frames)
        print("frame number = ", joint_data.shape[0], "(timestamps is aligned)")

    # get pre-defined constraints
    assert joint_data.shape[1] == 74
    constraints_nohand, constraints_hand, comp_constraints, direction_constraints = smplx2bvh_for_optimizing_smplx(N=joint_data.shape[1], split_hand=True)

    if start_frame_idx is None:
        start_frame_idx = 0
    if end_frame_idx is None:
        end_frame_idx = joint_data.shape[0]

    # init SMPLX pose from an optimization
    if os.path.isfile(join(save_dir, str(start_frame_idx) + ".npz")):
        init_results = np.load(join(save_dir, str(start_frame_idx) + ".npz"), allow_pickle=True)["results"].item()
        init_smplx_params = {
            "betas": init_results["betas"].detach().to(device),
            "expression": init_results["expression"].detach().to(device),
            "global_orient": init_results["global_orient"].detach().to(device),
            "transl": init_results["transl"].detach().to(device),
            "body_pose": init_results["body_pose"].detach().to(device),
            "left_hand_pose": init_results["left_hand_pose"].detach().to(device),
            "right_hand_pose": init_results["right_hand_pose"].detach().to(device),
        }
    else:
        init_smplx_params = optimize_pose_singleframe(joint_data, bvh_rot, betas, frame_idx=start_frame_idx, N_epoch=5000, save_dir=save_dir, device=device)

    cfg = {
        "OPT_BETAS": False,
        "OPT_POSE": True,
        "OPT_EXPRESSION": False,
    }

    # prepare constraints
    # point-to-point constraints
    constraints_nohand = np.int32(constraints_nohand).transpose(1, 0)
    idx_nohand = torch.LongTensor(constraints_nohand[0]).to(device)
    idx_nohand_bvh_skeleton = constraints_nohand[1]
    constraints_hand = np.int32(constraints_hand).transpose(1, 0)
    idx_hand = torch.LongTensor(constraints_hand[0]).to(device)
    idx_hand_bvh_skeleton = constraints_hand[1]

    ############ tune: optimization parameters ############
    M = 50  # 每次优化连续的M帧, M>=3
    N_epoch = 5000  # 每连续M帧的优化轮数
    ############ finish ############

    result_smplx_params = []

    # create an SMPLX model
    use_pca, num_pca_comps = True, 12

    for i in range(start_frame_idx, end_frame_idx, M):
        ids = np.arange(i, min(i+M, end_frame_idx))
        L = ids.shape[0]
        print("------------ optimizing frame {} - {} -----------------------".format(str(i), str(i+L-1)))
        # get gt data for this subsequence
        gt_pos_nohand = torch.from_numpy(joint_data[ids][:, idx_nohand_bvh_skeleton]).to(device)  # (N, 34, 3)
        gt_pos_hand = torch.from_numpy(joint_data[ids][:, idx_hand_bvh_skeleton]).to(device)  # (N, 40, 3)
        # gt_localrot = torch.from_numpy(bvh_rot[ids]).to(device)
        # gt_rot_neck_to_spine2_rotmat = bvh_rot[ids][:, 10, :3, :3]  # (N, 3, 3)
        # gt_rot_neck_to_spine2 = torch.from_numpy(np_mat2axangle(gt_rot_neck_to_spine2_rotmat)).to(device)  # (N, 3)
        # gt_rot_head_to_neck_rotmat = bvh_rot[ids][:, 11, :3, :3] @ bvh_rot[ids][:, 12, :3, :3]  # (N, 3, 3)
        # gt_rot_head_to_neck = torch.from_numpy(np_mat2axangle(gt_rot_head_to_neck_rotmat)).to(device)  # (N, 3)
        gt_head_rot = torch.from_numpy(bvh_rot[ids][:, 0, :3, :3] @ bvh_rot[ids][:, 7, :3, :3] @ bvh_rot[ids][:, 8, :3, :3] @ bvh_rot[ids][:, 9, :3, :3] @ bvh_rot[ids][:, 10, :3, :3] @ bvh_rot[ids][:, 11, :3, :3] @ bvh_rot[ids][:, 12, :3, :3]).to(device)  # (N, 3, 3)
        gt_spine_rot = torch.from_numpy(bvh_rot[ids][:, 0, :3, :3] @ bvh_rot[ids][:, 7, :3, :3]).to(device)  # (N, 3, 3)
        gt_spine2_rot = torch.from_numpy(bvh_rot[ids][:, 0, :3, :3] @ bvh_rot[ids][:, 7, :3, :3] @ bvh_rot[ids][:, 8, :3, :3]@bvh_rot[ids][:,9,:3,:3]).to(device)  # (N, 3, 3)
        gt_rightUpLeg_rot = torch.from_numpy(bvh_rot[ids][:, 0, :3, :3] @ bvh_rot[ids][:, 1, :3, :3]).to(device)  # (N, 3, 3)
        gt_leftUpLeg_rot = torch.from_numpy(bvh_rot[ids][:, 0, :3, :3] @ bvh_rot[ids][:, 4, :3, :3]).to(device)  # (N, 3, 3)
        gt_rightArm_rot = torch.from_numpy(bvh_rot[ids][:, 0, :3, :3] @ bvh_rot[ids][:, 7, :3, :3] @ bvh_rot[ids][:, 8, :3, :3] @ bvh_rot[ids][:, 9, :3, :3] @ bvh_rot[ids][:, 13, :3, :3] @ bvh_rot[ids][:, 14, :3, :3]).to(device)  # (N, 3, 3)
        gt_leftArm_rot = torch.from_numpy(bvh_rot[ids][:, 0, :3, :3] @ bvh_rot[ids][:, 7, :3, :3] @ bvh_rot[ids][:, 8, :3, :3] @ bvh_rot[ids][:, 9, :3, :3] @ bvh_rot[ids][:, 36, :3, :3]@ bvh_rot[ids][:, 37, :3, :3]).to(device)  # (N, 3, 3)
        gt_rightShoulder_rot = torch.from_numpy(bvh_rot[ids][:, 0, :3, :3] @ bvh_rot[ids][:, 7, :3, :3] @ bvh_rot[ids][:, 8, :3, :3] @ bvh_rot[ids][:, 9, :3, :3] @ bvh_rot[ids][:, 13, :3, :3]).to(device)  # (N, 3, 3)
        gt_leftShoulder_rot = torch.from_numpy(bvh_rot[ids][:, 0, :3, :3] @ bvh_rot[ids][:, 7, :3, :3] @ bvh_rot[ids][:, 8, :3, :3] @ bvh_rot[ids][:, 9, :3, :3] @ bvh_rot[ids][:, 36, :3, :3]).to(device)  # (N, 3, 3)
        gt_neck_rot = torch.from_numpy(bvh_rot[ids][:, 0, :3, :3] @ bvh_rot[ids][:, 7, :3, :3] @ bvh_rot[ids][:, 8, :3, :3]@bvh_rot[ids][:,9,:3,:3] @ bvh_rot[ids][:,10,:3,:3]).to(device)
        
        sequence_init_smplx_params = {}
        for key in init_smplx_params:
            sequence_init_smplx_params[key] = init_smplx_params[key].expand((L,) + init_smplx_params[key].shape[1:]).clone()

        smplx_model = create_SMPLX_model(use_pca=use_pca, num_pca_comps=num_pca_comps, batch_size=L, device=device)
        optim_model = Simple_SMPLX(smplx_model=smplx_model, init_smplx_params=sequence_init_smplx_params, cfg=cfg)
        optim_model.to(device)
        optimizer = torch.optim.Adam(optim_model.parameters(), lr=0.003)
        optim_model.train()

        for epoch in range(N_epoch):
            optimizer.zero_grad()
            results = optim_model()
            j, body_pose, left_hand_pose, right_hand_pose = results["joints"], results["body_pose"], results["left_hand_pose"], results["right_hand_pose"]

            
            pelvis_global_rot = batch_rodrigues(results["global_orient"])  # (N, 3, 3)
            local_rot = batch_rodrigues(results["body_pose"].view(L * 21, 3)).view(L, 21, 3, 3)  # (N, 21, 3, 3)
            pred_head_rot = torch.matmul(pelvis_global_rot, torch.matmul(local_rot[:, 2, :, :], torch.matmul(local_rot[:, 5, :, :], torch.matmul(local_rot[:, 8, :, :], torch.matmul(local_rot[:, 11, :, :], local_rot[:, 14, :, :])))))  # (N, 3, 3)

            # mark: 约束脊柱旋转
            # pred_spine_rot = torch.matmul(pelvis_global_rot, local_rot[:, 0, :, :])  # (N, 3, 3)            
            pred_rightHip_rot = torch.matmul(pelvis_global_rot, local_rot[:, 1, :, :])  # (N, 3, 3)
            pred_leftHip_rot = torch.matmul(pelvis_global_rot, local_rot[:, 0, :, :])
            pred_spine3_rot = torch.matmul(pelvis_global_rot, torch.matmul(local_rot[:, 2, :, :], torch.matmul(local_rot[:, 5, :, :], local_rot[:, 8, :, :])))
            pred_rightCollar_rot = torch.matmul(pelvis_global_rot, torch.matmul(local_rot[:, 2, :, :], torch.matmul(local_rot[:, 5, :, :], torch.matmul(local_rot[:, 8, :, :], local_rot[:, 13, :, :]))))
            pred_rightShoulder_rot = torch.matmul(pelvis_global_rot, torch.matmul(local_rot[:, 2, :, :], torch.matmul(local_rot[:, 5, :, :], torch.matmul(local_rot[:, 8, :, :], torch.matmul(local_rot[:, 13, :, :], local_rot[:, 16, :, :])))))
            pred_leftCollar_rot = torch.matmul(pelvis_global_rot, torch.matmul(local_rot[:, 2, :, :], torch.matmul(local_rot[:, 5, :, :], torch.matmul(local_rot[:, 8, :, :], local_rot[:, 12, :, :]))))
            pred_leftShoulder_rot = torch.matmul(pelvis_global_rot, torch.matmul(local_rot[:, 2, :, :], torch.matmul(local_rot[:, 5, :, :], torch.matmul(local_rot[:, 8, :, :], torch.matmul(local_rot[:, 12, :, :], local_rot[:, 15, :, :])))))
            pred_neck_rot = torch.matmul(pelvis_global_rot, torch.matmul(local_rot[:, 2, :, :], torch.matmul(local_rot[:, 5, :, :], torch.matmul(local_rot[:, 8, :, :], local_rot[:, 11, :, :]))))
            
            # a naive version
            # TODO: use cfg
            joint3D_loss_nohand = torch.sum((j[:, idx_nohand, :] - gt_pos_nohand)**2) * 1.0
            joint3D_loss_hand = torch.sum((j[:, idx_hand, :] - gt_pos_hand)**2) * 2.0
            regularizer = torch.sum(body_pose**2) * 1e-3 + torch.sum(left_hand_pose**2) * 1e-4 + torch.sum(right_hand_pose**2) * 1e-4
            smoothness = (torch.sum((2 * body_pose[1:-1] - body_pose[:-2] - body_pose[2:])**2) + torch.sum((2 * left_hand_pose[1:-1] - left_hand_pose[:-2] - left_hand_pose[2:])**2) + torch.sum((2 * right_hand_pose[1:-1] - right_hand_pose[:-2] - right_hand_pose[2:])**2)) * 20.0  # acceleration
            if (i > start_frame_idx) and (L >= 2):  # 不是第一个batch
                smoothness_to_init = (torch.sum((2 * body_pose[0] - init_smplx_params["body_pose"][0].detach() - body_pose[1])**2) + torch.sum((2 * left_hand_pose[0] - init_smplx_params["left_hand_pose"][0].detach() - left_hand_pose[1])**2) + torch.sum((2 * right_hand_pose[0] - init_smplx_params["right_hand_pose"][0].detach() - right_hand_pose[1])**2)) * 1e3  # acceleration
            else:
                smoothness_to_init = 0.0
            # head_orientation_loss = (torch.sum(torch.abs((body_pose[:, 11] - gt_rot_neck_to_spine2))) + torch.sum(torch.abs(body_pose[:, 14] - gt_rot_head_to_neck))) * 2e-1  # local constraint
            head_orientation_loss = torch.sum(torch.abs(pred_head_rot - gt_head_rot)) * 2e-1  # global constraint
            # mark: 约束脊柱旋转，命名为bvh
            spine_orientation_loss = torch.sum(torch.abs(pelvis_global_rot - gt_spine_rot)) * 2e-1
            spine2_orientation_loss = torch.sum(torch.abs(pred_spine3_rot - gt_spine2_rot)) * 2e-1
            leftUpLeg_orientation_loss = torch.sum(torch.abs(pred_leftHip_rot - gt_leftUpLeg_rot)) * 2e-1
            rightUpLeg_orientation_loss = torch.sum(torch.abs(pred_rightHip_rot - gt_rightUpLeg_rot)) * 2e-1
            rightShoulder_orientation_loss = torch.sum(torch.abs(pred_rightCollar_rot - gt_rightShoulder_rot)) * 2e-1
            leftShoulder_orientation_loss = torch.sum(torch.abs(pred_leftCollar_rot - gt_leftShoulder_rot)) * 2e-1
            rightArm_orientation_loss = torch.sum(torch.abs(pred_rightShoulder_rot - gt_rightArm_rot)) * 2e-1
            leftArm_orientation_loss = torch.sum(torch.abs(pred_leftShoulder_rot - gt_leftArm_rot)) * 2e-1
            neck_orientation_loss = torch.sum(torch.abs(pred_neck_rot - gt_neck_rot)) * 2e-1
            # regularizer_to_init = (torch.sum((body_pose - sequence_init_smplx_params["body_pose"].detach())**2) + torch.sum((left_hand_pose - sequence_init_smplx_params["left_hand_pose"].detach())**2) + torch.sum((right_hand_pose - sequence_init_smplx_params["right_hand_pose"].detach())**2)) * 0e-3
            loss = joint3D_loss_nohand + regularizer + joint3D_loss_hand + smoothness + smoothness_to_init + head_orientation_loss + spine_orientation_loss + leftUpLeg_orientation_loss + rightUpLeg_orientation_loss + rightShoulder_orientation_loss + leftShoulder_orientation_loss + rightArm_orientation_loss + leftArm_orientation_loss + neck_orientation_loss + spine2_orientation_loss
            # if (epoch % 100 == 0) or (epoch == N_epoch - 1):
            #     print(epoch, loss.item(), joint3D_loss_nohand.item(), joint3D_loss_hand.item(), regularizer.item(), smoothness.item(), smoothness_to_init.item(), 
            #         #   head_orientation_loss.item()
            #           ) 
            loss.backward()
            optimizer.step()
        
        optim_model.eval()
        results = optim_model()
        
        # save result
        for j in range(L):
            result_smplx_params.append({
                "betas": results["betas"][j:j+1].clone().detach(),
                "expression": results["expression"][j:j+1].clone().detach(),
                "global_orient": results["global_orient"][j:j+1].clone().detach(),
                "transl": results["transl"][j:j+1].clone().detach(),
                "body_pose": results["body_pose"][j:j+1].clone().detach(),
                "left_hand_pose": results["left_hand_pose"][j:j+1].clone().detach(),
                "right_hand_pose": results["right_hand_pose"][j:j+1].clone().detach(),
            })
        
        # update init pose
        init_smplx_params = {
            "betas": results["betas"][-1:].clone().detach(),
            "expression": results["expression"][-1:].clone().detach(),
            "global_orient": results["global_orient"][-1:].clone().detach(),
            "transl": results["transl"][-1:].clone().detach(),
            "body_pose": results["body_pose"][-1:].clone().detach(),
            "left_hand_pose": results["left_hand_pose"][-1:].clone().detach(),
            "right_hand_pose": results["right_hand_pose"][-1:].clone().detach(),
        }

        # save and visualize the result
        if not save_dir is None:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(join(save_dir, str(ids[0]) + "to" + str(ids[-1]) + ".npz"), results=results)  # save params
            save_pcd(join(save_dir, str(ids[-1]) + "_SMPLX_joints.ply"), results["joints"][-1].detach().cpu().numpy())
            save_pcd(join(save_dir, str(ids[-1]) + "_bvh_joints.ply"), joint_data[ids[-1]])
            faces = smplx_model.faces_tensor.detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=results["vertices"][-1].detach().cpu().numpy(), faces=faces)
            mesh_txt = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
            with open(join(save_dir, str(ids[-1]) + "_SMPLX_mesh.obj"), "w") as fp:
                fp.write(mesh_txt)
    
    return result_smplx_params
