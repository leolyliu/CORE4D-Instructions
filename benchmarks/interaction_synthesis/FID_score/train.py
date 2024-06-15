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


def train():

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
    
    print("")
    model = ActionRecogNet(args)
    # model.load_state_dict(torch.load(join("./save_checkpoints", "checkpoints_0025.pth")))
    model.to(device)

    batch_size = 32
    epoch = 1000
    learning_rate = 1e-3

    print("preparing datasets ...")
    train_dataset = HHODataset(device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    print("training ...")
    for epoch_idx in range(epoch):
        model.train()
        mean_training_loss = 0
        mean_acc = 0
        batch_cnt = 0
        for batch_idx, data in enumerate(train_dataloader):
            pred_actions, motion_features = model(data["motion"])  # (B, 9)
            gt_actions = data["action"]  # (B, 9)
            
            # print(pred_actions, gt_actions)

            loss = 1.0 * torch.mean((pred_actions - gt_actions)**2)

            # print("batch {} / {}, loss = {}".format(batch_idx, len(train_dataloader), loss.item()))
            mean_training_loss += loss.item()
            
            pred_action_labels = pred_actions.max(dim=1)[1]
            gt_action_labels = gt_actions.max(dim=1)[1]
            acc = (pred_action_labels == gt_action_labels).sum() / pred_action_labels.shape[0]
            mean_acc += acc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_cnt += 1
        
        mean_training_loss /= batch_cnt
        mean_acc /= batch_cnt
        print("epoch = {}, mean training loss = {}, mean accuracy = {}".format(epoch_idx, mean_training_loss, mean_acc))
        
        if epoch_idx % 10 == 0:
            os.makedirs("./save_checkpoints", exist_ok=True)
            torch.save(model.state_dict(), join("./save_checkpoints", "checkpoint_{}.pth".format(str(epoch_idx).zfill(4))))


if __name__ == "__main__":
    train()
