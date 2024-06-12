import os
from os.path import join, dirname, abspath, isfile, isdir
import numpy as np
import pickle
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import trimesh


ACTION_DICT = {
    "move1": 0,
    "move2": 1,
    "raise": 2,
    "rot": 3,
    "pass1": 4,
    "pass2": 5,
    "join": 6,
    "leave": 7,
}


def load_action_info(fp):
    action_rawdata = json.load(open(fp, "r"))
    action_info = {}
    for key in action_rawdata:
        x = action_rawdata[key]
        if "_" in x:
            x = x.split("_")[0]
        if not x in ACTION_DICT:
            print("other action:", x)
            v = 8
        else:
            v = ACTION_DICT[x]
        action_info[key.replace("/", ".")] = v
        
    return action_info


class HHODataset(Dataset):
    def __init__(self, N_obj_point=32, dataset_roots=None, use_pred_results=False, device="cuda:0"):
        
        self.device = device
        self.N_obj_point = N_obj_point
        
        if dataset_roots is None:
            self.dataset_roots = [
                "/localdata/liuyun/HHO-dataset/omomo_runs/hho_singlestage/prediction_results",
                "/localdata/liuyun/HHO-dataset/omomo_runs/hho_singlestage/prediction_results_on_training_set",
            ]
        else:
            self.dataset_roots = dataset_roots
        
        self.action_info = load_action_info(fp="/share/datasets/hhodataset/annotations.json")
        
        self.data = []
        for dr in self.dataset_roots:
            for fn in tqdm(os.listdir(dr)):
                if not fn.endswith(".pkl"):
                    continue
                
                seq_data = pickle.load(open(join(dr, fn), "rb"))
                seq_name = seq_data["seq_name"]
                
                if not use_pred_results:                 
                    gt_p1_jnts_list = seq_data["gt_p1_jnts_list"]
                    gt_p2_jnts_list = seq_data["gt_p2_jnts_list"]
                else:
                    gt_p1_jnts_list = seq_data["pred_p1_jnts_list"]
                    gt_p2_jnts_list = seq_data["pred_p2_jnts_list"]
                
                obj_verts_list = seq_data["obj_verts_list"]
                obj_faces_list = seq_data["obj_faces_list"]
                
                ids = np.random.choice(obj_verts_list.shape[1], self.N_obj_point, replace=True)
                obj_pts = obj_verts_list[:, ids].to(gt_p1_jnts_list.device)  # (T, 32, 3)
                
                # print(seq_name)
                
                if not seq_name in self.action_info:
                    print("error:", seq_name)
                    continue
                
                motion_data = torch.cat([gt_p1_jnts_list.reshape(-1, 24*3), gt_p2_jnts_list.reshape(-1, 24*3), obj_pts.reshape(-1, self.N_obj_point*3)], dim=1).to(torch.float32).to(self.device)  # (120, 24*3+24*3+32*3)
                action = torch.zeros(9).to(torch.float32).to(self.device)  # (9,)
                action[self.action_info[seq_name]] = 1
                
                self.data.append({
                    "motion": motion_data,
                    "action": action,
                })
                
                # if len(self.data) > 100:
                #     break
        
        print(len(self.data))
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    dataset = HHODataset()
