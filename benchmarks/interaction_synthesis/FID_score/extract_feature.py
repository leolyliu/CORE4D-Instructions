import argparse
import os
from os.path import join, dirname, abspath, isfile
import sys
sys.path.append(dirname(abspath(__file__)))
import numpy as np
import cv2
import torch
import json
from torch.utils.data import DataLoader
from dataset import HHODataset
from model import ActionRecogNet


def extract_feature():
    
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--latent_usage", type=str, default='memory')
    parser.add_argument("--template_type", type=str, default='zero')
    parser.add_argument('--star_graph', default=False, action='store_true')
    args = parser.parse_args()
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    
    model = ActionRecogNet(args)
    model.load_state_dict(torch.load("./save_checkpoints/checkpoint_0100.pth"))
    model.to(device)
    model.eval()
    
    batch_size = 32
    suffix = "_3"
    
    print("1 preparing datasets ...")
    dataset = HHODataset(dataset_roots=["/cephfs_yili/backup/liuyun_localdata_10.210.5.10/HHO-dataset/omomo_runs/hho_singlestage/prediction_results_on_training_set"], device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    N_batch = len(dataloader)
    features = np.zeros((N_batch*batch_size, args.latent_dim))
    for batch_idx, data in enumerate(dataloader):
        pred_actions, motion_features = model(data["motion"])  # (B, 9)
        assert motion_features.shape == (batch_size, args.latent_dim)
        features[batch_idx*batch_size : (batch_idx+1)*batch_size] = motion_features.detach().cpu().numpy()
    np.save("./features_gt_train.npy", features)
    
    print("2 preparing datasets ...")
    dataset = HHODataset(dataset_roots=["/cephfs_yili/backup/liuyun_localdata_10.210.5.10/HHO-dataset/omomo_runs/hho_singlestage/prediction_results{}".format(suffix)], device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    N_batch = len(dataloader)
    features = np.zeros((N_batch*batch_size, args.latent_dim))
    for batch_idx, data in enumerate(dataloader):
        pred_actions, motion_features = model(data["motion"])  # (B, 9)
        assert motion_features.shape == (batch_size, args.latent_dim)
        features[batch_idx*batch_size : (batch_idx+1)*batch_size] = motion_features.detach().cpu().numpy()
    np.save("./features_gt_test.npy", features)
    
    print("3 preparing datasets ...")
    dataset = HHODataset(dataset_roots=["/cephfs_yili/backup/liuyun_localdata_10.210.5.10/HHO-dataset/omomo_runs/hho_singlestage/prediction_results{}".format(suffix)], use_pred_results=True, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    N_batch = len(dataloader)
    features = np.zeros((N_batch*batch_size, args.latent_dim))
    for batch_idx, data in enumerate(dataloader):
        pred_actions, motion_features = model(data["motion"])  # (B, 9)
        assert motion_features.shape == (batch_size, args.latent_dim)
        features[batch_idx*batch_size : (batch_idx+1)*batch_size] = motion_features.detach().cpu().numpy()
    np.save("./features_pred_singlestage{}.npy".format(suffix), features)
    
    print("4 preparing datasets ...")
    dataset = HHODataset(dataset_roots=["/cephfs_yili/backup/liuyun_localdata_10.210.5.10/HHO-dataset/omomo_runs/stage2_exp_out/prediction_results{}".format(suffix)], use_pred_results=True, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    N_batch = len(dataloader)
    features = np.zeros((N_batch*batch_size, args.latent_dim))
    for batch_idx, data in enumerate(dataloader):
        pred_actions, motion_features = model(data["motion"])  # (B, 9)
        assert motion_features.shape == (batch_size, args.latent_dim)
        features[batch_idx*batch_size : (batch_idx+1)*batch_size] = motion_features.detach().cpu().numpy()
    np.save("./features_pred_twostages{}.npy".format(suffix), features)


if __name__ == "__main__":
    extract_feature()
