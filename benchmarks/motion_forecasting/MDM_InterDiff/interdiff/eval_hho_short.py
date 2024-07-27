import os
from os.path import join, dirname, abspath
import sys
from datetime import datetime
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from data.dataset_hho import Dataset, MODEL_PATH
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation
from render.mesh_viz import visualize_body_obj_hho
from train_correction_hho import LitInteraction as LitObj
from train_diffusion_hho import LitInteraction
from data.utils import markerset_ssm67_smplx
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix, axis_angle_to_quaternion
from copy import deepcopy
from tools import point2point_signed
from data.tools import vertex_normals
sys.path.insert(0, join(dirname(abspath(__file__)), "../../../.."))
from data_processing.smplx import smplx


def compute_contact_accuracy(obj_points, pred_joints, gt_joints, contact_threshould=0.05):
    """
    obj_points: (T, B, N_point, 3)
    pred_joints: (T, B, 127, 3)
    gt_joints: (T, B, 127, 3)
    contact_threshould: 5cm
    
    return: (B,)
    """
    T, B, _, _ = obj_points.shape
    assert pred_joints.shape == (T, B, 127, 3)  # joints from SMPLX forward
    assert gt_joints.shape == (T, B, 127, 3)  # joints from SMPLX forward
    
    pred_left_wrist = (pred_joints[:, :, 20] + pred_joints[:, :, 28]) / 2  # approximate palm center
    pred_right_wrist = (pred_joints[:, :, 21] + pred_joints[:, :, 43]) / 2  # approximate palm center
    gt_left_wrist = (gt_joints[:, :, 20] + gt_joints[:, :, 28]) / 2  # approximate palm center
    gt_right_wrist = (gt_joints[:, :, 21] + gt_joints[:, :, 43]) / 2  # approximate palm center
    
    pred_left_contact_flag = (((obj_points - pred_left_wrist.reshape(T, B, 1, 3))**2).sum(dim=-1)**0.5).min(dim=-1)[0] < contact_threshould
    pred_right_contact_flag = (((obj_points - pred_right_wrist.reshape(T, B, 1, 3))**2).sum(dim=-1)**0.5).min(dim=-1)[0] < contact_threshould
    gt_left_contact_flag = (((obj_points - gt_left_wrist.reshape(T, B, 1, 3))**2).sum(dim=-1)**0.5).min(dim=-1)[0] < contact_threshould
    gt_right_contact_flag = (((obj_points - gt_right_wrist.reshape(T, B, 1, 3))**2).sum(dim=-1)**0.5).min(dim=-1)[0] < contact_threshould
    
    left_contact_accuracy = (pred_left_contact_flag == gt_left_contact_flag).sum(dim=0) / T
    right_contact_accuracy = (pred_right_contact_flag == gt_right_contact_flag).sum(dim=0) / T
    contact_accuracy = (left_contact_accuracy + right_contact_accuracy) / 2  # (B,)
    return contact_accuracy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def metrics(obj_pred, body1_jtr, body1, body2_jtr, body2, obj_gt, body1_jtr_gt, body1_gt, body2_jtr_gt, body2_gt, verts1, verts2, faces, obj_points):
    # NOTE: could be modified for more efficient implementation
    # body_jtr, body_trans, obj_pred, body_jtr_gt, body_trans_gt, obj_gt
    # body_jtr: T, B, N_jtr, 3
    # body_trans: T, B, 3
    # obj_pred: T, B, 6  rot in first three items as axis-angle 
    # gender: from dataset, each example has a different gender
    T, B, N_jtr,_ = body1_jtr_gt.shape# N_jtr 52?

    obj_rot_matrix = axis_angle_to_matrix(obj_pred[:, :, :-3].view(T, B, 3))
    obj_points_pred = torch.matmul(obj_points.unsqueeze(0), obj_rot_matrix.permute(0, 1, 3, 2)) + obj_pred[:, :, -3:].unsqueeze(2)

    # liuyun: contact accuracy
    contact_accuracy_p12o = compute_contact_accuracy(obj_points_pred.view(T, B, -1, 3), body1_jtr.view(T, B, -1, 3), body1_jtr_gt.view(T, B, -1, 3))
    contact_accuracy_p22o = compute_contact_accuracy(obj_points_pred.view(T, B, -1, 3), body2_jtr.view(T, B, -1, 3), body2_jtr_gt.view(T, B, -1, 3))

    # penetration
    normals1 = vertex_normals(verts1.view(T * B, -1, 3), faces.unsqueeze(0).repeat(T * B, 1, 1))
    o2p1_signed, p12o_signed, o2p1_idx, p12o_idx, o2p1, p12o = point2point_signed(verts1.view(T * B, -1, 3), obj_points_pred.view(T * B, -1, 3), x_normals=normals1, return_vector=True)
    w_dist_neg_o2p1 = (o2p1_signed < 0).view(T, B, -1).float()
    penetrate_o2p1 = w_dist_neg_o2p1.mean(dim=2).mean(dim=0)
    normals2 = vertex_normals(verts2.view(T * B, -1, 3), faces.unsqueeze(0).repeat(T * B, 1, 1))
    o2p2_signed, p22o_signed, o2p2_idx, p22o_idx, o2p2, p22o = point2point_signed(verts2.view(T * B, -1, 3), obj_points_pred.view(T * B, -1, 3), x_normals=normals2, return_vector=True)
    w_dist_neg_o2p2 = (o2p2_signed < 0).view(T, B, -1).float()
    penetrate_o2p2 = w_dist_neg_o2p2.mean(dim=2).mean(dim=0)

    body1_trans = body1[:, :, -3:]
    body1_trans_gt = body1_gt[:, :, -3:]
    body2_trans = body2[:, :, -3:]
    body2_trans_gt = body2_gt[:, :, -3:]
    # global mpjpe
    p1_global_mpjpe = (body1_jtr - body1_jtr_gt).norm(dim=3).mean(dim=2).mean(dim=0)
    p2_global_mpjpe = (body2_jtr - body2_jtr_gt).norm(dim=3).mean(dim=2).mean(dim=0)

    # align pelvis
    pelvis1 = body1_jtr[:,:,0:1,:]
    pelvis1_gt = body1_jtr_gt[:,:,0:1,:]
    pelvis2 = body2_jtr[:,:,0:1,:]
    pelvis2_gt = body2_jtr_gt[:,:,0:1,:]
    
    body1_jtr = body1_jtr - pelvis1
    body1_jtr_gt = body1_jtr_gt - pelvis1_gt
    body2_jtr = body2_jtr - pelvis2
    body2_jtr_gt = body2_jtr_gt - pelvis2_gt
    # local mpjpe
    p1_local_mpjpe = (body1_jtr - body1_jtr_gt).norm(dim=3).mean(dim=2).mean(dim=0)
    p2_local_mpjpe = (body2_jtr - body2_jtr_gt).norm(dim=3).mean(dim=2).mean(dim=0)
    
    # hand mpjpe
    assert body1_jtr.shape == (T, B, 127, 3)
    p1_hand_mpjpe = (body1_jtr[:, :, 20:22, :] - body1_jtr_gt[:, :, 20:22, :]).norm(dim=3).mean(dim=2).mean(dim=0)
    p2_hand_mpjpe = (body2_jtr[:, :, 20:22, :] - body2_jtr_gt[:, :, 20:22, :]).norm(dim=3).mean(dim=2).mean(dim=0)

    # body_translation
    p1_translation = (body1_trans - body1_trans_gt).norm(dim=2).mean(dim=0)
    p2_translation = (body2_trans - body2_trans_gt).norm(dim=2).mean(dim=0)

    # translation
    obj_translation = (obj_pred[:,:,-3:] - obj_gt[:,:,-3:]).norm(dim=2).mean(dim=0)
    
    # rotation error (unit: degree)
    obj_rot_mat = axis_angle_to_matrix(obj_pred[:,:,:3])  # (T, B, 3, 3)
    obj_rot_mat_gt = axis_angle_to_matrix(obj_gt[:,:,:3])  # (T, B, 3, 3)
    rotation_error = []
    for frame_idx in range(T):
        for batch_idx in range(B):
            R_pred = obj_rot_mat[frame_idx, batch_idx].detach().cpu().numpy()
            R_gt = obj_rot_mat_gt[frame_idx, batch_idx].detach().cpu().numpy()
            R_diff = np.arccos(((np.trace(R_pred @ R_gt.T) - 1) / 2).clip(-1, 1)) / np.pi * 180  # unit: deg
            rotation_error.append(R_diff)
    rotation_error = torch.tensor(rotation_error, dtype=obj_translation.dtype).to(obj_translation.device).reshape(T, B).mean(dim=0)

    metric_dict = dict(
        p1_global_mpjpe = p1_global_mpjpe,
        p1_local_mpjpe = p1_local_mpjpe,
        p1_translation = p1_translation,
        p2_global_mpjpe = p2_global_mpjpe,
        p2_local_mpjpe = p2_local_mpjpe,
        p2_translation = p2_translation,
        obj_translation = obj_translation,
        obj_rot_error = rotation_error,
        p1_contact_accuracy = contact_accuracy_p12o,
        p2_contact_accuracy = contact_accuracy_p22o,
        penetrate_o2p1 = penetrate_o2p1,
        penetrate_o2p2 = penetrate_o2p2,
        p1_hand_mpjpe = p1_hand_mpjpe,
        p2_hand_mpjpe = p2_hand_mpjpe,
    )
    return metric_dict


def denoised_fn(x, t, model_kwargs):
    # x: (B, C, 1, T)
    
    if t[0] > 500 or t[0] % 50 != 0:
        return x
    body1, body2, obj = torch.split(x.squeeze(1).permute(2, 0, 1).contiguous(), [args.smpl_dim+3, args.smpl_dim+3, 9], dim=2)
    body1_gt, body2_gt, obj_gt = torch.split(model_kwargs['y']['inpainted_motion'].squeeze(1).permute(2, 0, 1).contiguous(), [args.smpl_dim+3, args.smpl_dim+3, 9], dim=2)
    T, B, _ = body1[:, :, :-3].shape
    obj_rot_matrix = rotation_6d_to_matrix(obj[:, :, :-3].view(T, B, 6))
    body1_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body1[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
    body2_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body2[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
    # hand_pose = model_kwargs['y']['hand_pose']
    body1_pred = torch.cat([body1_rot, body1[:, :, -3:]], dim=2)
    body2_pred = torch.cat([body2_rot, body2[:, :, -3:]], dim=2)
    
    smplx_model = model_kwargs['y']['smplx']

    beta1_batch = model_kwargs['y']['beta1'].view(T * B, -1)
    body1_pred = body1_pred.detach().clone()
    body1_pred_batch = body1_pred.view(T * B, -1)
    body1_pts = smplx_model(beta1_batch.detach(), body_pose=body1_pred_batch[:, 3:-3].detach(), global_orient=body1_pred_batch[:, :3].detach(), transl=body1_pred_batch[:, -3:].detach())
    verts1, jtr1 = body1_pts.vertices, body1_pts.joints
    beta2_batch = model_kwargs['y']['beta2'].view(T * B, -1)
    body2_pred = body2_pred.detach().clone()
    body2_pred_batch = body2_pred.view(T * B, -1)
    body2_pts = smplx_model(beta2_batch.detach(), body_pose=body2_pred_batch[:, 3:-3].detach(), global_orient=body2_pred_batch[:, :3].detach(), transl=body2_pred_batch[:, -3:].detach())
    verts2, jtr2 = body2_pts.vertices, body2_pts.joints
        
    markers1 = verts1[:, markerset_ssm67_smplx]
    markers1 = markers1.view(T, B, -1, 3)  # (T, B, 67, 3)
    markers2 = verts2[:, markerset_ssm67_smplx]
    markers2 = markers2.view(T, B, -1, 3)  # (T, B, 67, 3)

    obj_model = model_kwargs['y']['obj_model']
    obj_points_pred = torch.matmul(model_kwargs['y']['obj_points'].unsqueeze(0), obj_rot_matrix.permute(0, 1, 3, 2)) + obj[:, :, -3:].unsqueeze(2)  # (T, B, 2048, 3)
    # print(torch.where((torch.norm((markers.unsqueeze(2) - obj_points_pred.unsqueeze(3)), dim=4) < 0.03).any(dim=3)))

    smplx_faces = smplx_model.faces_tensor.detach()
    p1_normals = vertex_normals(verts1, smplx_faces.unsqueeze(0).repeat(T * B, 1, 1))
    p2_normals = vertex_normals(verts2, smplx_faces.unsqueeze(0).repeat(T * B, 1, 1))
    o2p1_signed, p12o_signed, o2p1_idx, p12o_idx, o2p1, p12o = point2point_signed(verts1.view(T * B, -1, 3), obj_points_pred.view(T * B, -1, 3), x_normals=p1_normals, return_vector=True)
    o2p2_signed, p22o_signed, o2p2_idx, p22o_idx, o2p2, p22o = point2point_signed(verts2.view(T * B, -1, 3), obj_points_pred.view(T * B, -1, 3), x_normals=p2_normals, return_vector=True)

    w1 = torch.zeros([T * B, o2p1_signed.size(1)]).to(o2p1_signed.device)
    w1_dist = (o2p1_signed < 0.01) * (o2p1_signed > 0)
    w1_dist_neg = o2p1_signed < 0
    w1[w1_dist] = 0  # small weight for far away vertices
    w1[w1_dist_neg] = 20  # large weight for penetration
    w2 = torch.zeros([T * B, o2p2_signed.size(1)]).to(o2p2_signed.device)
    w2_dist = (o2p2_signed < 0.01) * (o2p2_signed > 0)
    w2_dist_neg = o2p2_signed < 0
    w2[w2_dist] = 0  # small weight for far away vertices
    w2[w2_dist_neg] = 20  # large weight for penetration

    loss_dist_o2p1 = torch.einsum('ij,ij->ij', torch.abs(o2p1_signed), w1).view(T, B, -1)
    distance1 = (torch.norm((markers1.unsqueeze(2) - obj_points_pred.unsqueeze(3)), dim=4)).min(dim=3)[0].min(dim=2)[0].mean(dim=0)
    # condition1 = torch.logical_not(torch.logical_and(loss_dist_o2p1[args.past_len:].mean(dim=2).mean(dim=0) < 0.002, distance1 < 0.02))  # (B,)
    condition1 = torch.logical_and(loss_dist_o2p1[args.past_len:].mean(dim=2).mean(dim=0) < 0.002, distance1 < 0.02)  # (B,)
    contact_person1_label = (torch.norm((markers1.unsqueeze(2) - obj_points_pred.unsqueeze(3)), dim=4) < 0.02).any(dim=2)
    contact_person1 = torch.zeros_like(contact_person1_label, device=contact_person1_label.device)
    contact_person1[contact_person1_label] = 1
    contact_person1 = contact_person1[args.past_len:].sum(dim=0)
    loss_dist_o2p2 = torch.einsum('ij,ij->ij', torch.abs(o2p2_signed), w2).view(T, B, -1)
    distance2 = (torch.norm((markers2.unsqueeze(2) - obj_points_pred.unsqueeze(3)), dim=4)).min(dim=3)[0].min(dim=2)[0].mean(dim=0)
    # condition2 = torch.logical_not(torch.logical_and(loss_dist_o2p2[args.past_len:].mean(dim=2).mean(dim=0) < 0.002, distance2 < 0.02))  # (B,)
    condition2 = torch.logical_and(loss_dist_o2p2[args.past_len:].mean(dim=2).mean(dim=0) < 0.002, distance2 < 0.02)  # (B,)
    contact_person2_label = (torch.norm((markers2.unsqueeze(2) - obj_points_pred.unsqueeze(3)), dim=4) < 0.02).any(dim=2)
    contact_person2 = torch.zeros_like(contact_person2_label, device=contact_person2_label.device)
    contact_person2[contact_person2_label] = 1
    contact_person2 = contact_person2[args.past_len:].sum(dim=0)
    
    obj_proj = obj_model.model.sample(obj_gt[:, :, :-3], obj_gt[:, :, -3:], markers1, contact_person1, markers2, contact_person2)
    x_ = torch.cat([body1, body2, obj_proj], dim=2).permute(1, 2, 0).unsqueeze(1).contiguous()
    x_ = t[0] / 1000 * x + (1 - t[0] / 1000) * x_
    x[condition2] = x_[condition2]
    x[condition1] = x_[condition1]
    
    return x


def sample_once_proj(batch):
    with torch.no_grad():
        embedding, gt = model.model._get_embeddings(batch, device)
        T, B, _ = gt.shape
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()
        model_kwargs = {'y': {'cond': embedding}}
        model_kwargs['y']['inpainted_motion'] = gt
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                device=device)  # True means use gt motion
        model_kwargs['y']['inpainting_mask'][:, :, :, args.past_len:] = False  # do inpainting in those frames

        sample_fn = model.diffusion.p_sample_loop
        # hand_pose = torch.cat([frame['smplfit_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float().to(device)
        # model_kwargs['y']['hand_pose'] = hand_pose[idx_pad]
        smplx_model = model_kwargs['y']['smplx'] = smplx.create(MODEL_PATH, model_type="smplx", gender="neutral", batch_size=T*B, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=12, flat_hand_mean=True).to(device)
        model_kwargs['y']['beta1'] = torch.stack([record['person1_params']['betas'] for record in batch['frames']], dim=0).to(device)
        model_kwargs['y']['beta2'] = torch.stack([record['person2_params']['betas'] for record in batch['frames']], dim=0).to(device)
        model_kwargs['y']['obj_model'] = obj_model
        model_kwargs['y']['obj_points'] = batch['obj_points'][:, :, :3].float().to(device)

        noise = torch.randn(*gt.shape, device=device)
        sample = sample_fn(model.model, gt.shape, clip_denoised=False, noise=noise, model_kwargs=model_kwargs, denoised_fn=denoised_fn)
        body1_pred, body2_pred, obj_pred = torch.split(sample.squeeze(1).permute(2, 0, 1).contiguous(), [args.smpl_dim+3, args.smpl_dim+3, 9], dim=2)
        body1_gt, body2_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), [args.smpl_dim+3, args.smpl_dim+3, 9], dim=2)
        T, B, _ = body1_pred[:, :, :-3].shape
        body1_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body1_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        body1_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body1_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        body2_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body2_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        body2_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body2_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        obj_rot_matrix = rotation_6d_to_matrix(obj_pred[:, :, :-3].view(T, B, 6))
        obj_rot = matrix_to_axis_angle(obj_rot_matrix).view(T, B, -1)
        obj_rot_gt_matrix = rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, 6))
        obj_rot_gt = matrix_to_axis_angle(obj_rot_gt_matrix).view(T, B, -1)
        body1_pred = torch.cat([body1_rot, body1_pred[:, :, -3:]], dim=2)
        body1_gt = torch.cat([body1_rot_gt, body1_gt[:, :, -3:]], dim=2)
        body2_pred = torch.cat([body2_rot, body2_pred[:, :, -3:]], dim=2)
        body2_gt = torch.cat([body2_rot_gt, body2_gt[:, :, -3:]], dim=2)
        
        body1_betas = torch.cat([record['person1_params']['betas'] for record in batch['frames']], dim=0).to(device)
        body1_betas_batch = body1_betas.view(T * B, -1)
        body1 = body1_pred.detach().clone()
        body1_batch = body1.view(T * B, -1)
        body1_pts = smplx_model(body1_betas_batch.detach(), body_pose=body1_batch[:, 3:-3].detach(), global_orient=body1_batch[:, :3].detach(), transl=body1_batch[:, -3:].detach())
        verts1, jtr1 = body1_pts.vertices, body1_pts.joints
        body2_betas = torch.cat([record['person2_params']['betas'] for record in batch['frames']], dim=0).to(device)
        body2_betas_batch = body2_betas.view(T * B, -1)
        body2 = body2_pred.detach().clone()
        body2_batch = body2.view(T * B, -1)
        body2_pts = smplx_model(body2_betas_batch.detach(), body_pose=body2_batch[:, 3:-3].detach(), global_orient=body2_batch[:, :3].detach(), transl=body2_batch[:, -3:].detach())
        verts2, jtr2 = body2_pts.vertices, body2_pts.joints

        obj_pred = torch.cat([obj_rot, obj_pred[:, :, -3:]], dim=2)

    return obj_pred, body1_pred, verts1.view(T, B, -1, 3), jtr1.view(T, B, -1, 3), jtr1.view(T, B, -1, 3)[:, :, 0, :], body2_pred, verts2.view(T, B, -1, 3), jtr2.view(T, B, -1, 3), jtr2.view(T, B, -1, 3)[:, :, 0, :]

def sample_once(batch):
    with torch.no_grad():
        embedding, gt = model.model._get_embeddings(batch, device)
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()
        model_kwargs = {'y': {'cond': embedding}}
        model_kwargs['y']['inpainted_motion'] = gt
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                device=device)  # True means use gt motion
        model_kwargs['y']['inpainting_mask'][:, :, :, args.past_len:] = False  # do inpainting in those frames

        sample_fn = model.diffusion.p_sample_loop

        noise = torch.randn(*gt.shape, device=device)
        sample = sample_fn(model.model, gt.shape, clip_denoised=False, noise=noise, model_kwargs=model_kwargs)
        body1_pred, body2_pred, obj_pred = torch.split(sample.squeeze(1).permute(2, 0, 1).contiguous(), [args.smpl_dim+3, args.smpl_dim+3, 9], dim=2)
        body1_gt, body2_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), [args.smpl_dim+3, args.smpl_dim+3, 9], dim=2)
        T, B, _ = body1_pred[:, :, :-3].shape
        body1_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body1_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        body1_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body1_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        body2_rot = matrix_to_axis_angle(rotation_6d_to_matrix(body2_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        body2_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body2_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        obj_rot = matrix_to_axis_angle(rotation_6d_to_matrix(obj_pred[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        obj_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        # hand_pose = torch.cat([frame['smplfit_params']['pose'][:, 66:].unsqueeze(0) for frame in batch['frames']], dim=0).float().to(device)
        body1_pred = torch.cat([body1_rot, body1_pred[:, :, -3:]], dim=2)
        body1_gt = torch.cat([body1_rot_gt, body1_gt[:, :, -3:]], dim=2)
        body2_pred = torch.cat([body2_rot, body2_pred[:, :, -3:]], dim=2)
        body2_gt = torch.cat([body2_rot_gt, body2_gt[:, :, -3:]], dim=2)
        
        smplx_model = smplx.create(MODEL_PATH, model_type="smplx", gender="neutral", batch_size=T*B, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=12, flat_hand_mean=True)
        
        body1_betas = torch.cat([record['person1_params']['betas'] for record in batch['frames']], dim=0)
        body1_betas_batch = body1_betas.view(T * B, -1)
        body1 = body1_pred.detach().clone()
        body1_batch = body1.view(T * B, -1)
        body1_pts = smplx_model(body1_betas_batch.detach().cpu(), body_pose=body1_batch[:, 3:-3].detach().cpu(), global_orient=body1_batch[:, :3].detach().cpu(), transl=body1_batch[:, -3:].detach().cpu())
        verts1, jtr1 = body1_pts.vertices, body1_pts.joints
        body2_betas = torch.cat([record['person2_params']['betas'] for record in batch['frames']], dim=0)
        body2_betas_batch = body2_betas.view(T * B, -1)
        body2 = body2_pred.detach().clone()
        body2_batch = body2.view(T * B, -1)
        body2_pts = smplx_model(body2_betas_batch.detach().cpu(), body_pose=body2_batch[:, 3:-3].detach().cpu(), global_orient=body2_batch[:, :3].detach().cpu(), transl=body2_batch[:, -3:].detach().cpu())
        verts2, jtr2 = body2_pts.vertices, body2_pts.joints
        
        obj_pred = torch.cat([obj_rot, obj_pred[:, :, -3:]], dim=2)

    return obj_pred, body1_pred, verts1.view(T, B, -1, 3).to(device), jtr1.view(T, B, -1, 3).to(device), jtr1.view(T, B, -1, 3)[:, :, 0, :].to(device), body2_pred, verts2.view(T, B, -1, 3).to(device), jtr2.view(T, B, -1, 3).to(device), jtr2.view(T, B, -1, 3)[:, :, 0, :].to(device)

def smooth(obj, body1, verts1, jtrs1, pelvis1, body2, verts2, jtrs2, pelvis2):
    obj[-args.future_len:] = obj[-args.future_len:] + (2 * obj[-args.future_len-1] - obj[-args.future_len-2] - obj[-args.future_len])
    body1[-args.future_len:] = body1[-args.future_len:] + (2 * body1[-args.future_len-1] - body1[-args.future_len-2] - body1[-args.future_len])
    verts1[-args.future_len:] = verts1[-args.future_len:] + (2 * verts1[-args.future_len-1] - verts1[-args.future_len-2] - verts1[-args.future_len])
    jtrs1[-args.future_len:] = jtrs1[-args.future_len:] + (2 * jtrs1[-args.future_len-1] - jtrs1[-args.future_len-2] - jtrs1[-args.future_len])
    pelvis1[-args.future_len:] = pelvis2[-args.future_len:] + (2 * pelvis2[-args.future_len-1] - pelvis2[-args.future_len-2] - pelvis2[-args.future_len])
    body2[-args.future_len:] = body2[-args.future_len:] + (2 * body2[-args.future_len-1] - body2[-args.future_len-2] - body2[-args.future_len])
    verts2[-args.future_len:] = verts2[-args.future_len:] + (2 * verts2[-args.future_len-1] - verts2[-args.future_len-2] - verts2[-args.future_len])
    jtrs2[-args.future_len:] = jtrs2[-args.future_len:] + (2 * jtrs2[-args.future_len-1] - jtrs2[-args.future_len-2] - jtrs2[-args.future_len])
    pelvis2[-args.future_len:] = pelvis2[-args.future_len:] + (2 * pelvis2[-args.future_len-1] - pelvis2[-args.future_len-2] - pelvis2[-args.future_len])
    return obj, body1, verts1, jtrs1, pelvis1, body2, verts2, jtrs2, pelvis2

def get_gt(batch):
    with torch.no_grad():
        embedding, gt = model.model._get_embeddings(batch, device)
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()

        body1_gt, body2_gt, obj_gt = torch.split(gt.squeeze(1).permute(2, 0, 1).contiguous(), [args.smpl_dim+3, args.smpl_dim+3, 9], dim=2)
        T, B, _ = body1_gt[:, :, :-3].shape
        body1_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body1_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        body2_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(body2_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        obj_rot_gt = matrix_to_axis_angle(rotation_6d_to_matrix(obj_gt[:, :, :-3].view(T, B, -1, 6))).view(T, B, -1)
        body1_gt = torch.cat([body1_rot_gt, body1_gt[:, :, -3:]], dim=2)
        body2_gt = torch.cat([body2_rot_gt, body2_gt[:, :, -3:]], dim=2)

        obj_gt = torch.cat([obj_rot_gt, obj_gt[:, :, -3:]], dim=2)

        smplx_model = smplx.create(MODEL_PATH, model_type="smplx", gender="neutral", batch_size=T*B, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=12, flat_hand_mean=True)
        faces = smplx_model.faces_tensor.detach().cpu()
        
        body1_betas=torch.cat([record['person1_params']['betas'] for record in batch['frames']], dim=0)
        body1_betas_batch = body1_betas.view(T * B, -1)
        body1_gt_batch = body1_gt.view(T * B, -1)
        body1_pts = smplx_model(body1_betas_batch.detach().cpu(), body_pose=body1_gt_batch[:, 3:-3].detach().cpu(), global_orient=body1_gt_batch[:, :3].detach().cpu(), transl=body1_gt_batch[:, -3:].detach().cpu())
        verts1_gt, jtr1_gt = body1_pts.vertices, body1_pts.joints
        body2_betas=torch.cat([record['person2_params']['betas'] for record in batch['frames']], dim=0)
        body2_betas_batch = body2_betas.view(T * B, -1)
        body2_gt_batch = body2_gt.view(T * B, -1)
        body2_pts = smplx_model(body2_betas_batch.detach().cpu(), body_pose=body2_gt_batch[:, 3:-3].detach().cpu(), global_orient=body2_gt_batch[:, :3].detach().cpu(), transl=body2_gt_batch[:, -3:].detach().cpu())
        verts2_gt, jtr2_gt = body2_pts.vertices, body2_pts.joints
        
    return obj_gt, jtr1_gt.view(T, B, -1, 3).to(device), body1_gt, jtr2_gt.view(T, B, -1, 3).to(device), body2_gt, faces.to(device), verts1_gt.view(T, B, -1, 3).to(device), verts2_gt.view(T, B, -1, 3).to(device)

def sample(name):
    if name == 'correction':
        sample_func = sample_once_proj
    else:
        sample_func = sample_once

    metric_dict = dict(
        p1_global_mpjpe = 0,
        p1_hand_mpjpe = 0,
        p1_local_mpjpe = 0,
        p1_translation = 0,
        p2_global_mpjpe = 0,
        p2_hand_mpjpe = 0,
        p2_local_mpjpe = 0,
        p2_translation = 0,
        obj_translation = 0,
        obj_rot_error = 0,
        p1_contact_accuracy = 0,
        p2_contact_accuracy = 0,
        penetrate_o2p1 = 0,
        penetrate_o2p2 = 0,
    )

    # motion_save_dir = join(save_dir, 'pred_motion')
    # os.makedirs(motion_save_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            p1_global_mpjpe = torch.zeros(args.batch_size).to(device) + 1e10
            p1_hand_mpjpe = torch.zeros(args.batch_size).to(device) + 1e10
            p1_local_mpjpe = torch.zeros(args.batch_size).to(device) + 1e10
            p1_translation = torch.zeros(args.batch_size).to(device) + 1e10
            p2_global_mpjpe = torch.zeros(args.batch_size).to(device) + 1e10
            p2_hand_mpjpe = torch.zeros(args.batch_size).to(device) + 1e10
            p2_local_mpjpe = torch.zeros(args.batch_size).to(device) + 1e10
            p2_translation = torch.zeros(args.batch_size).to(device) + 1e10
            obj_translation = torch.zeros(args.batch_size).to(device) + 1e10
            obj_rot_error = torch.zeros(args.batch_size).to(device) + 1e10
            p1_contact_accuracy = torch.zeros(args.batch_size).to(device) + 1e10
            p2_contact_accuracy = torch.zeros(args.batch_size).to(device) + 1e10
            penetrate_o2p1 = torch.zeros(args.batch_size).to(device) + 1e10
            penetrate_o2p2 = torch.zeros(args.batch_size).to(device) + 1e10
            obj_gt, jtr1_gt, body1_gt, jtr2_gt, body2_gt, faces, verts1_gt, verts2_gt = get_gt(batch)
            for j in range(args.diverse_samples):
                new_batch = deepcopy(batch)
                obj, body1, verts1, jtrs1, pelvis1, body2, verts2, jtrs2, pelvis2 = sample_func(new_batch)

                # save predicted motion
                # save_path = join(motion_save_dir, "batch_{}_sample_{}.pkl".format(i, j))
                # results = {
                #     "obj_gt": obj_gt,  # (15+15, B, 6)
                #     "jtr1_gt": jtr1_gt,  # (15+15, B, 127, 3)
                #     "body1_gt": body1_gt,  # (15+15, B, 69, 3)
                #     "jtr2_gt": jtr2_gt,  # (15+15, B, 127, 3)
                #     "body2_gt": body2_gt,  # (15+15, B, 69, 3)
                #     "faces": faces,  # (N_face, 3)
                #     "obj": obj,  # (15+15, B, 6)
                #     "body1": body1,  # (15+15, B, 69, 3)
                #     "verts1": verts1,  # (15+15, B, 10475, 3)
                #     "jtrs1": jtrs1,  # (15+15, B, 127, 3)
                #     "pelvis1": pelvis1,  # (15+15, B, 3)
                #     "body2": body2,  # (15+15, B, 69, 3)
                #     "verts2": verts2,  # (15+15, B, 10475, 3)
                #     "jtrs2": jtrs2,  # (15+15, B, 127, 3)
                #     "pelvis2": pelvis2,  # (15+15, B, 3)
                # }
                # pickle.dump(results, open(save_path, "wb"))
                
                metric = metrics(obj[args.past_len:], jtrs1[args.past_len:], body1[args.past_len:], jtrs2[args.past_len:], body2[args.past_len:], obj_gt[args.past_len:], jtr1_gt[args.past_len:], body1_gt[args.past_len:], jtr2_gt[args.past_len:], body2_gt[args.past_len:], verts1[args.past_len:], verts2[args.past_len:], faces, batch['obj_points'][:, :, :3].float().to(device))
                p1_global_mpjpe = torch.stack([p1_global_mpjpe, metric['p1_global_mpjpe']])
                p1_local_mpjpe = torch.stack([p1_local_mpjpe, metric['p1_local_mpjpe']])
                p1_translation = torch.stack([p1_translation, metric['p1_translation']])
                p2_global_mpjpe = torch.stack([p2_global_mpjpe, metric['p2_global_mpjpe']])
                p2_local_mpjpe = torch.stack([p2_local_mpjpe, metric['p2_local_mpjpe']])
                p2_translation = torch.stack([p2_translation, metric['p2_translation']])
                obj_translation = torch.stack([obj_translation, metric['obj_translation']])
                obj_rot_error = torch.stack([obj_rot_error, metric['obj_rot_error']])
                p1_contact_accuracy = torch.stack([p1_contact_accuracy, metric['p1_contact_accuracy']])
                p2_contact_accuracy = torch.stack([p2_contact_accuracy, metric['p2_contact_accuracy']])
                penetrate_o2p1 = torch.stack([penetrate_o2p1, metric['penetrate_o2p1']])
                penetrate_o2p2 = torch.stack([penetrate_o2p2, metric['penetrate_o2p2']])
                p1_hand_mpjpe = torch.stack([p1_hand_mpjpe, metric['p1_hand_mpjpe']])
                p2_hand_mpjpe = torch.stack([p2_hand_mpjpe, metric['p2_hand_mpjpe']])

                # obj, body1, verts1, jtrs1, pelvis1, body2, verts2, jtrs2, pelvis2 = smooth(obj, body1, verts1, jtrs1, pelvis1, body2, verts2, jtrs2, pelvis2)
                if i % args.render_epoch == 0:
                    visualize(batch, i, obj[:, 0], verts1[:, 0], verts2[:, 0], faces, name, gt=False)
                    visualize(batch, i, obj_gt[:, 0], verts1_gt[:, 0], verts2_gt[:, 0], faces, name, gt=True)

            metric_dict['p1_global_mpjpe'] += p1_global_mpjpe.min(dim=0)[0].mean().item()
            metric_dict['p1_local_mpjpe'] += p1_local_mpjpe.min(dim=0)[0].mean().item()
            metric_dict['p1_translation'] += p1_translation.min(dim=0)[0].mean().item()
            metric_dict['p2_global_mpjpe'] += p2_global_mpjpe.min(dim=0)[0].mean().item()
            metric_dict['p2_local_mpjpe'] += p2_local_mpjpe.min(dim=0)[0].mean().item()
            metric_dict['p2_translation'] += p2_translation.min(dim=0)[0].mean().item()
            metric_dict['obj_translation'] += obj_translation.min(dim=0)[0].mean().item()
            metric_dict['obj_rot_error'] += obj_rot_error.min(dim=0)[0].mean().item()
            metric_dict['p1_contact_accuracy'] += p1_contact_accuracy.min(dim=0)[0].mean().item()
            metric_dict['p2_contact_accuracy'] += p2_contact_accuracy.min(dim=0)[0].mean().item()
            metric_dict['penetrate_o2p1'] += penetrate_o2p1.min(dim=0)[0].mean().item()
            metric_dict['penetrate_o2p2'] += penetrate_o2p2.min(dim=0)[0].mean().item()
            metric_dict['p1_hand_mpjpe'] += p1_hand_mpjpe.min(dim=0)[0].mean().item()
            metric_dict['p2_hand_mpjpe'] += p2_hand_mpjpe.min(dim=0)[0].mean().item()
            print("[evaluated {} batches]".format(i+1))
            print('p1_global_mpjpe', metric_dict['p1_global_mpjpe'] / (i+1))
            print('p1_hand_mpjpe', metric_dict['p1_hand_mpjpe'] / (i+1))
            print('p1_local_mpjpe', metric_dict['p1_local_mpjpe'] / (i+1))
            print('p1_translation', metric_dict['p1_translation'] / (i+1))
            print('p2_global_mpjpe', metric_dict['p2_global_mpjpe'] / (i+1))
            print('p2_hand_mpjpe', metric_dict['p2_hand_mpjpe'] / (i+1))
            print('p2_local_mpjpe', metric_dict['p2_local_mpjpe'] / (i+1))
            print('p2_translation', metric_dict['p2_translation'] / (i+1))
            print('obj_translation', metric_dict['obj_translation'] / (i+1))
            print('obj_rot_error', metric_dict['obj_rot_error'] / (i+1))
            print('p1_contact_accuracy', metric_dict['p1_contact_accuracy'] / (i+1))
            print('p2_contact_accuracy', metric_dict['p2_contact_accuracy'] / (i+1))
            print('penetrate_o2p1', metric_dict['penetrate_o2p1'] / (i+1))
            print('penetrate_o2p2', metric_dict['penetrate_o2p2'] / (i+1))
            
                        
def visualize(batch, j, obj, verts1, verts2, faces, name, gt=False):
    verts1 = verts1.detach().cpu().numpy()
    verts2 = verts2.detach().cpu().numpy()
    faces = faces.cpu().numpy()
    obj_verts = []
    # visualize
    export_file = Path.joinpath(save_dir, 'render')
    export_file.mkdir(exist_ok=True, parents=True)
    # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
    
    if not gt:
        rend_video_path = os.path.join(export_file, 's{}_l{}_r{}_{}_{}.gif'.format(batch['start_frame'][0], obj.shape[0], args.sample_rate, j, name))
    else:
        rend_video_path = os.path.join(export_file, 's{}_l{}_r{}_{}_{}_gt.gif'.format(batch['start_frame'][0], obj.shape[0], args.sample_rate, j, name))

    mesh_obj = Mesh()
    mesh_obj.load_from_file(batch['obj_model_path'][0])

    for t in range(obj.shape[0]):
        mesh_obj_v = mesh_obj.v.copy()
        
        angle, trans = obj[t][:-3].detach().cpu().numpy(), obj[t][-3:].detach().cpu().numpy()
        rot = Rotation.from_rotvec(angle).as_matrix()
        # transform canonical mesh to fitting
        mesh_obj_v = np.matmul(mesh_obj_v, rot.T) + trans
        obj_verts.append(mesh_obj_v)

    m1 = visualize_body_obj_hho(verts1, verts2, faces, np.array(obj_verts), mesh_obj.f, past_len=args.past_len, save_path=rend_video_path, sample_rate=args.sample_rate)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='Diffusion')
    parser.add_argument("--use_pointnet2", type=int, default=1)
    parser.add_argument("--num_obj_keypoints", type=int, default=256)
    parser.add_argument("--sample_rate", type=int, default=1)

    # transformer
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--latent_usage", type=str, default='memory')
    parser.add_argument("--template_type", type=str, default='zero')
    parser.add_argument('--star_graph', default=False, action='store_true')

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)
    parser.add_argument("--robust_kl", type=int, default=1)
    parser.add_argument("--weight_template", type=float, default=0.1)
    parser.add_argument("--weight_kl", type=float, default=1e-2)
    parser.add_argument("--weight_contact", type=float, default=0)
    parser.add_argument("--weight_dist", type=float, default=1)
    parser.add_argument("--weight_penetration", type=float, default=0)  #10

    parser.add_argument("--weight_smplx_rot", type=float, default=1)
    parser.add_argument("--weight_smplx_nonrot", type=float, default=0.2)
    parser.add_argument("--weight_obj_rot", type=float, default=0.1)
    parser.add_argument("--weight_obj_nonrot", type=float, default=0.2)
    parser.add_argument("--weight_past", type=float, default=0.5)
    parser.add_argument("--weight_jtr", type=float, default=0.1)
    parser.add_argument("--weight_jtr_v", type=float, default=500)
    parser.add_argument("--weight_v", type=float, default=1)

    parser.add_argument("--use_contact", type=int, default=0)
    parser.add_argument("--use_annealing", type=int, default=0)

    # dataset
    parser.add_argument("--past_len", type=int, default=15)
    parser.add_argument("--future_len", type=int, default=15)

    # train
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--second_stage", type=int, default=20,
                        help="annealing some loss weights in early epochs before this num")
    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%H:%M:%S.%f"))
    parser.add_argument("--render_epoch", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--resume_checkpoint_obj", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--mode", type=str, default='correction')
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--dct", type=int, default=10)
    parser.add_argument("--autoregressive", type=int, default=0)

    # diffusion
    parser.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                        help="Noise schedule type")
    parser.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--cond_mask_prob", default=0, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    parser.add_argument("--diverse_samples", type=int, default=1)
    
    # paths
    parser.add_argument("--dataset_root", type=str, default="/share/datasets/hhodataset/prepared_motion_forecasting_data", help="Directory of prepared motion forecasting data (each file is named 'data.npz').")
    parser.add_argument("--smplx_model_dir", type=str, default="/share/human_model/models", help="Directory of SMPL-X models.")
    parser.add_argument("--results_folder", type=str, default="./results", help="Directory of saved results.")
    
    # test set
    parser.add_argument("--test_set", type=str, default="all", help="Test set ('all' / 'seen' / 'unseen').")
    
    # test set selection
    parser.add_argument("--set", type=str, default="unseen")
    
    args = parser.parse_args()
    idx_pad = list(range(args.past_len)) + [args.past_len - 1] * args.future_len
    # make demterministic
    pl.seed_everything(233, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # rendering and results
    results_folder = args.results_folder
    os.makedirs(results_folder, exist_ok=True)
    test_dataset = Dataset(mode="test", dataset_root=args.dataset_root, smplx_model_dir=args.smplx_model_dir, past_len=args.past_len, future_len=args.future_len, sample_rate=args.sample_rate, test_set=args.test_set)

    args.smpl_dim = 66 * 2
    args.num_obj_points = test_dataset.num_obj_points
    args.num_verts = len(markerset_ssm67_smplx)

    #pin_memory cause warning in pytorch 1.9.0
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                            drop_last=True, pin_memory=False)
    print('dataset loaded')
    
    model = LitInteraction.load_from_checkpoint(args.resume_checkpoint, args=args).to(device)
    if args.mode != "no_correction":
        print("[select an object ckeckpoint] obj_ckpt_path =", args.resume_checkpoint_obj)
        obj_model = LitObj.load_from_checkpoint(args.resume_checkpoint_obj, args=args).to(device)
    else:
        obj_model = LitObj(args)
        
    model.eval()
    obj_model.eval()
    tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/sample_hho_' + args.set + "_" + args.mode), name=args.expr_name)
    save_dir = Path(tb_logger.log_dir)  # for this version
    print(save_dir)
    sample(args.mode)
