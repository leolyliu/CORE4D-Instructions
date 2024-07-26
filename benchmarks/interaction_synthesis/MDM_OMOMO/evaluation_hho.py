import os
from os.path import join, dirname, abspath
import sys
import numpy as np
import pickle
import torch
from tqdm import tqdm
from evaluation_metrics import compute_metrics, compute_s1_metrics, compute_collision
sys.path.insert(0, join(dirname(abspath(__file__)), "../.."))
from dataset_statistics.train_test_split import load_train_test_split


def evaluate(seq_data_paths):
    
    _, test_seqnames_seen, test_seqnames_unseen = load_train_test_split()
    
    eval_results = {
        "seen": {
            "p1_mpvpe_list": [],
            "p2_mpvpe_list": [],
            "p1_mpjpe_list": [],
            "p2_mpjpe_list": [],
            "p1_contact_acc": [],
            "p2_contact_acc": [],
        },
        "unseen": {
            "p1_mpvpe_list": [],
            "p2_mpvpe_list": [],
            "p1_mpjpe_list": [],
            "p2_mpjpe_list": [],
            "p1_contact_acc": [],
            "p2_contact_acc": [],
        },
    }
    
    for seq_data_path in tqdm(seq_data_paths):
        
        seq_data = pickle.load(open(seq_data_path, "rb"))
        seq_name = seq_data["seq_name"]
        
        pred_p1_trans_list = seq_data["pred_p1_trans_list"]
        pred_p1_rot_list = seq_data["pred_p1_rot_list"]
        pred_p1_jnts_list = seq_data["pred_p1_jnts_list"]
        pred_p1_verts_list = seq_data["pred_p1_verts_list"]
        pred_p2_trans_list = seq_data["pred_p2_trans_list"]
        pred_p2_rot_list = seq_data["pred_p2_rot_list"]
        pred_p2_jnts_list = seq_data["pred_p2_jnts_list"]
        pred_p2_verts_list = seq_data["pred_p2_verts_list"]
                            
        gt_p1_trans_list = seq_data["gt_p1_trans_list"]
        gt_p1_rot_list = seq_data["gt_p1_rot_list"]
        gt_p1_jnts_list = seq_data["gt_p1_jnts_list"]
        gt_p1_verts_list = seq_data["gt_p1_verts_list"]
        gt_p2_trans_list = seq_data["gt_p2_trans_list"]
        gt_p2_rot_list = seq_data["gt_p2_rot_list"]
        gt_p2_jnts_list = seq_data["gt_p2_jnts_list"]
        gt_p2_verts_list = seq_data["gt_p2_verts_list"]
                            
        p1_faces_list = seq_data["p1_faces_list"]
        p2_faces_list = seq_data["p2_faces_list"]
        obj_verts_list = seq_data["obj_verts_list"]
        obj_faces_list = seq_data["obj_faces_list"]
        actual_len_list = seq_data["actual_len_list"]
        
        # person 1
        lhand_jpe, rhand_jpe, hand_jpe, mpvpe_1, mpjpe_1, rot_dist, trans_err, gt_contact_dist, contact_dist, \
            gt_foot_sliding_jnts, foot_sliding_jnts, contact_precision, contact_recall, \
            contact_acc_1, contact_f1_score = \
            compute_metrics(gt_p1_verts_list, pred_p1_verts_list, gt_p1_jnts_list, pred_p1_jnts_list, p1_faces_list, \
            gt_p1_trans_list, pred_p1_trans_list, gt_p1_rot_list, pred_p1_rot_list, \
            obj_verts_list, obj_faces_list, actual_len_list, use_joints24=True)
        lhand_jpe, rhand_jpe, hand_jpe, mpvpe_2, mpjpe_2, rot_dist, trans_err, gt_contact_dist, contact_dist, \
            gt_foot_sliding_jnts, foot_sliding_jnts, contact_precision, contact_recall, \
            contact_acc_2, contact_f1_score = \
            compute_metrics(gt_p2_verts_list, pred_p1_verts_list, gt_p2_jnts_list, pred_p1_jnts_list, p1_faces_list, \
            gt_p2_trans_list, pred_p1_trans_list, gt_p2_rot_list, pred_p1_rot_list, \
            obj_verts_list, obj_faces_list, actual_len_list, use_joints24=True)
        mpvpe = min(mpvpe_1, mpvpe_2)
        mpjpe = min(mpjpe_1, mpjpe_2)
        contact_acc = max(contact_acc_1, contact_acc_2)
        
        if seq_name in test_seqnames_seen:
            eval_results["seen"]["p1_mpvpe_list"].append(mpvpe)
            eval_results["seen"]["p1_mpjpe_list"].append(mpjpe)
            eval_results["seen"]["p1_contact_acc"].append(contact_acc)
        elif seq_name in test_seqnames_unseen:
            eval_results["unseen"]["p1_mpvpe_list"].append(mpvpe)
            eval_results["unseen"]["p1_mpjpe_list"].append(mpjpe)
            eval_results["unseen"]["p1_contact_acc"].append(contact_acc)
        else:
            assert False
        
        # person 2
        lhand_jpe, rhand_jpe, hand_jpe, mpvpe_1, mpjpe_1, rot_dist, trans_err, gt_contact_dist, contact_dist, \
            gt_foot_sliding_jnts, foot_sliding_jnts, contact_precision, contact_recall, \
            contact_acc_1, contact_f1_score = \
            compute_metrics(gt_p2_verts_list, pred_p2_verts_list, gt_p2_jnts_list, pred_p2_jnts_list, p2_faces_list, \
            gt_p2_trans_list, pred_p2_trans_list, gt_p2_rot_list, pred_p2_rot_list, \
            obj_verts_list, obj_faces_list, actual_len_list, use_joints24=True)
        lhand_jpe, rhand_jpe, hand_jpe, mpvpe_2, mpjpe_2, rot_dist, trans_err, gt_contact_dist, contact_dist, \
            gt_foot_sliding_jnts, foot_sliding_jnts, contact_precision, contact_recall, \
            contact_acc_2, contact_f1_score = \
            compute_metrics(gt_p1_verts_list, pred_p2_verts_list, gt_p1_jnts_list, pred_p2_jnts_list, p2_faces_list, \
            gt_p1_trans_list, pred_p2_trans_list, gt_p1_rot_list, pred_p2_rot_list, \
            obj_verts_list, obj_faces_list, actual_len_list, use_joints24=True)
        mpvpe = min(mpvpe_1, mpvpe_2)
        mpjpe = min(mpjpe_1, mpjpe_2)
        contact_acc = max(contact_acc_1, contact_acc_2)
        
        if seq_name in test_seqnames_seen:
            eval_results["seen"]["p2_mpvpe_list"].append(mpvpe)
            eval_results["seen"]["p2_mpjpe_list"].append(mpjpe)
            eval_results["seen"]["p2_contact_acc"].append(contact_acc)
        elif seq_name in test_seqnames_unseen:
            eval_results["unseen"]["p2_mpvpe_list"].append(mpvpe)
            eval_results["unseen"]["p2_mpjpe_list"].append(mpjpe)
            eval_results["unseen"]["p2_contact_acc"].append(contact_acc)
        else:
            assert False
        
    for data_split in eval_results:
        for metric in eval_results[data_split]:
            eval_results[data_split][metric] = np.float32(eval_results[data_split][metric]).mean()
    
    print("###### overall score ######")
    print("[seen]")
    print("mean MPVPE (mm) =", (eval_results["seen"]["p1_mpvpe_list"] + eval_results["seen"]["p2_mpvpe_list"]) / 2)
    print("mean MPJPE (mm) =", (eval_results["seen"]["p1_mpjpe_list"] + eval_results["seen"]["p2_mpjpe_list"]) / 2)
    print("mean contact accuracy (%) =", (eval_results["seen"]["p1_contact_acc"] + eval_results["seen"]["p2_contact_acc"]) / 2 * 100)
    print("[unseen]")
    print("mean MPVPE (mm) =", (eval_results["unseen"]["p1_mpvpe_list"] + eval_results["unseen"]["p2_mpvpe_list"]) / 2)
    print("mean MPJPE (mm) =", (eval_results["unseen"]["p1_mpjpe_list"] + eval_results["unseen"]["p2_mpjpe_list"]) / 2)
    print("mean contact accuracy (%) =", (eval_results["unseen"]["p1_contact_acc"] + eval_results["unseen"]["p2_contact_acc"]) / 2 * 100)


if __name__ == "__main__":
    results_dir = "/cephfs_yili/backup/liuyun_localdata_10.210.5.10/HHO-dataset/omomo_runs/stage2_exp_out/prediction_results"
    seq_data_paths = []
    for fn in os.listdir(results_dir):
        if not fn.endswith(".pkl"):
            continue
        seq_data_paths.append(join(results_dir, fn))
        # if len(seq_datas) >= 100:
        #     break
    
    evaluate(seq_data_paths)
