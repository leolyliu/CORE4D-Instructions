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
from data.tools import vertex_normals
from data.utils import markerset_ssm67_smplx
sys.path.insert(0, join(dirname(abspath(__file__)), "../../../.."))
from data_processing.smplx import smplx
from dataset_statistics.train_test_split import load_train_test_split, load_train_test_split_retargeted
import open3d as o3d

MODEL_PATH = "/share/human_model/models"


class Dataset(Dataset):
    def __init__(self, mode='train', past_len=15, future_len=15, sample_rate=1, dataset_root="/data2/datasets/hhodataset/prepared_motion_forecasting_data", test_set=None):
        # super.__init__(Dataset)
        
        # TODO: select specific categories
        # self.obj_categories = ["chair", "desk"]
        self.obj_categories = ["chair", "desk", "board", "box", "bucket", "stick"]
        # TODO: select whether adding real/retargeted data
        self.add_real = True
        self.add_retarget = True
        
        # # get sequence dirs
        # if mode == 'train':
        #     self.clip_names = ["20231002", "20231003_1", "20231008", "20231018", "20231020", "20231023", "20231108"]
        # elif mode == 'test':
        #     self.clip_names = ["20231003_2", "20231011", "20231030"]
        # else:
        #     raise Exception('mode must be train or test.')
        # self.seq_dirs = []
        # for clip_name in self.clip_names:
        #     clip_dir = join(dataset_root, clip_name)
        #     for seq_name in os.listdir(clip_dir):
        #         if isfile(join(clip_dir, seq_name, "data.npz")):
        #             self.seq_dirs.append(join(clip_dir, seq_name))
        
        # get sequence dirs
        train_sequence_names, test_sequence_names_seen_obj, test_sequence_names_unseen_obj = load_train_test_split()
        if not self.add_real:
            train_sequence_names = []
        if self.add_retarget:
            train_sequence_names_retargeted = load_train_test_split_retargeted()
            train_sequence_names += train_sequence_names_retargeted
        if mode == 'train':
            sequence_names = train_sequence_names
        elif mode == "test":
            if test_set is None:
                sequence_names = test_sequence_names_seen_obj + test_sequence_names_unseen_obj
            elif test_set == "seen":
                sequence_names = test_sequence_names_seen_obj
            elif test_set == "unseen":
                sequence_names = test_sequence_names_unseen_obj
            else:
                raise NotImplementedError
        else:
            raise Exception('mode must be train or test.')
        self.seq_dirs = []
        for sn in sequence_names:
            seq_dir = join(dataset_root, sn.replace(".", "/"))
            if isfile(join(seq_dir, "data.npz")):
                self.seq_dirs.append(seq_dir)
        self.seq_dirs.sort()
        
        # # rescaling
        # if mode == "train":
        #     N = len(self.seq_dirs)
        #     ids = np.random.choice(np.arange(0, N), 700)
        #     self.seq_dirs_copy = self.seq_dirs.copy()
        #     self.seq_dirs = []
        #     for x in ids:
        #         self.seq_dirs.append(self.seq_dirs_copy[x])
        
        print("###### Number of sequences in total =", len(self.seq_dirs))
        
        # self.repeat_ratio = 10 if mode == "train" else 1
        self.repeat_ratio = 1
        
        self.past_len = past_len
        self.future_len = future_len
        
        self.smplx_model = smplx.create(MODEL_PATH, model_type="smplx", gender="neutral", batch_size=1, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=12, flat_hand_mean=True)
        
        # read data
        self.data = []
        self.idx2frame = [] # (seq_id, sub_seq_id, bias)
        seq_idx = -1
        for k, seq_dir in tqdm(enumerate(self.seq_dirs)):
            d = np.load(join(seq_dir, "data.npz"), allow_pickle=True)["data"].item()
            clip_name, seq_name = seq_dir.split("/")[-2:]
            N_frame = d["N_frame"]  # int
            obj_model_path = d["obj_model_path"].replace("/data3/datasets/HHO_object_dataset_final", "/data2/datasets/HHO_object_dataset_final_simplified")  # str
            
            if (obj_model_path.find("vchair") > -1) or (obj_model_path.find("vtable") > -1):
                obj_cat = obj_model_path.split("/")[-1][1:6].replace("table", "desk")
            else:
                obj_cat = obj_model_path.split("/")[-3]
            if not obj_cat in self.obj_categories:
                continue
            seq_idx += 1
            
            obj_points = d["object_points"]  # shape = (2048, 7)
            obj_poses = d["obj_poses"]  # shape = (N_frame, 4, 4)
            human_params_p1 = d["human_params"]["person1"]
            human_params_p2 = d["human_params"]["person2"]
            contact_o_to_p1 = d["contact_object_to_person1"]
            contact_o_to_p2 = d["contact_object_to_person2"]
            contact_p1_to_o = d["contact_person1_to_object"]
            contact_p2_to_o = d["contact_person2_to_object"]
            foot_contact_label_p1 = d["foot_contact_label_person1"]
            foot_contact_label_p2 = d["foot_contact_label_person2"]
            
            p1_verts = torch.from_numpy(human_params_p1["vertices"])
            p1_jts = torch.from_numpy(human_params_p1["joints"])
            p1_faces = torch.from_numpy(human_params_p1["faces"])
            p2_verts = torch.from_numpy(human_params_p2["vertices"])
            p2_jts = torch.from_numpy(human_params_p2["joints"])
            p2_faces = torch.from_numpy(human_params_p2["faces"])
            
            # compute human normals
            normal_file = os.path.join(seq_dir, "human_normal.npz")
            if os.path.isfile(normal_file):
                with np.load(normal_file, allow_pickle=True) as f:
                    d = f["data"].item()
                    p1_normals = d["person1_normals"]
                    p2_normals = d["person2_normals"]
            else:
                p1_normals = vertex_normals(p1_verts, p1_faces.unsqueeze(0).repeat(N_frame, 1, 1)).numpy()
                p2_normals = vertex_normals(p2_verts, p2_faces.unsqueeze(0).repeat(N_frame, 1, 1)).numpy()
                np.savez(normal_file, data={"person1_normals": p1_normals, "person2_normals": p2_normals})
            
            p1_normals = torch.from_numpy(p1_normals)
            p2_normals = torch.from_numpy(p2_normals)
            p1_verts = torch.cat([p1_verts, p1_normals], dim=2)
            p2_verts = torch.cat([p2_verts, p2_normals], dim=2)
            p1_pelvis, p2_pelvis = np.float32(p1_jts[:, 0]), np.float32(p2_jts[:, 0])
            p1_left_foot, p2_left_foot = np.float32(p1_jts[:, 10]), np.float32(p2_jts[:, 10])
            p1_right_foot, p2_right_foot = np.float32(p1_jts[:, 11]), np.float32(p2_jts[:, 11])
            
            records = {
                "obj_model_path": obj_model_path,
                "obj_poses": obj_poses,
                "p1_poses": np.concatenate((human_params_p1["global_orient"], human_params_p1["body_pose"].reshape(N_frame, 63), human_params_p1["left_hand_pose"], human_params_p1["right_hand_pose"]), axis=1),  # np, (N_frame, 3+21*3+12+12)
                "p2_poses": np.concatenate((human_params_p2["global_orient"], human_params_p2["body_pose"].reshape(N_frame, 63), human_params_p2["left_hand_pose"], human_params_p2["right_hand_pose"]), axis=1),  # np, (N_frame, 3+21*3+12+12)
                "p1_betas": human_params_p1["betas"],  # np, (N_frame, 10)
                "p2_betas": human_params_p2["betas"],  # np, (N_frame, 10)
                "p1_trans": human_params_p1["transl"],  # np, (N_frame, 3)
                "p2_trans": human_params_p2["transl"],  # np, (N_frame, 3)
                "p1_pelvis": p1_pelvis,  # np, (N_frame, 3)
                "p1_left_foot": p1_left_foot,  # np, (N_frame, 3)
                "p1_right_foot": p1_right_foot,  # np, (N_frame, 3)
                "p2_pelvis": p2_pelvis,  # np, (N_frame, 3)
                "p2_left_foot": p2_left_foot,  # np, (N_frame, 3)
                "p2_right_foot": p2_right_foot,  # np, (N_frame, 3)
                "clip_name": clip_name,  # str
                "seq_name": seq_name,  # str
                "obj_points": obj_points,  # (2048, 7)
                "contact_object_to_person1": contact_o_to_p1,  # list, len=N_frame, 元素都是list且不定长
                "contact_object_to_person2": contact_o_to_p2,  # list, len=N_frame, 元素都是list且不定长
                "contact_person1_to_object": contact_p1_to_o,  # list, len=N_frame, 元素都是list且不定长
                "contact_person2_to_object": contact_p2_to_o,  # list, len=N_frame, 元素都是list且不定长
                "p1_verts": np.float32(p1_verts),  # np, (N_frame, 10475, 3)
                "p2_verts": np.float32(p2_verts),  # np, (N_frame, 10475, 3)
                "foot_contact_label_person1": foot_contact_label_p1,  # list, len=N_frame, 元素是10/11
                "foot_contact_label_person2": foot_contact_label_p2,  # list, len=N_frame, 元素是10/11
            }
            self.data.append(records)
            
            # 切片
            fragment = (past_len + future_len) * sample_rate # 30
            for i in range(N_frame // fragment):
                if mode == "test":
                    self.idx2frame.append((seq_idx, i * fragment, 1))
                elif i == N_frame // fragment - 1:
                    self.idx2frame.append((seq_idx, i * fragment, N_frame + 1 - (N_frame // fragment) * fragment))
                else:
                    self.idx2frame.append((seq_idx, i * fragment, fragment))
            
            # TODO: for debug
            # if k > 30:
            #     break
        
        self.num_verts = p1_verts.shape[1]  # 10475
        self.num_markers = len(markerset_ssm67_smplx)  # 67
        self.num_obj_points = records["obj_points"].shape[0]  # 2048
        self.smpl_dim = records["p1_poses"][0].shape[0]  # 90 = 3 + 21*3 + 12 + 12
        self.sample_rate = sample_rate  # 1
        print("====> The number of clips for " + mode + " data: " + str(len(self.idx2frame)) + " <====")
        assert len(self.idx2frame) > 0
    
    def __getitem__(self, idx):
        
        idx = idx % len(self.idx2frame)
        
        index, frame_idx, bias = self.idx2frame[idx]
        data = self.data[index]
        start_frame = np.random.choice(bias) + frame_idx
        end_frame = start_frame + (self.past_len + self.future_len) * self.sample_rate
        centroid = None
        rotation = None
        rotation_v = None
        
        frames = []
        for i in range(start_frame, end_frame, self.sample_rate):
            person1_params = {"pose": data["p1_poses"][i].copy(), "trans": data["p1_trans"][i].copy(), "betas": data["p1_betas"][i].copy()}
            person2_params = {"pose": data["p2_poses"][i].copy(), "trans": data["p2_trans"][i].copy(), "betas": data["p2_betas"][i].copy()}
            objfit_params = {"angle": Rotation.from_matrix(data["obj_poses"][i][:3, :3].copy()).as_rotvec(), "trans": data["obj_poses"][i][:3, 3].copy()}
            p1_pelvis = data["p1_pelvis"][i].copy()
            p2_pelvis = data["p2_pelvis"][i].copy()
            # NOTE: Canonicalize the first human pose
            if i == start_frame:
                centroid = p1_pelvis.copy()
                global_orient = Rotation.from_rotvec(person1_params["pose"][:3]).as_matrix()
                rotation_v = np.eye(3).astype(np.float32)
                cos, sin = global_orient[0, 0] / np.sqrt(global_orient[0, 0]**2 + global_orient[2, 0]**2), global_orient[2, 0] / np.sqrt(global_orient[0, 0]**2 + global_orient[2, 0]**2)
                rotation_v[[0, 2, 0, 2], [0, 2, 2, 0]] = np.array([cos, cos, -sin, sin])
                rotation = np.linalg.inv(rotation_v).astype(np.float32)
            
            person1_params["trans"] = person1_params["trans"] - centroid
            person2_params["trans"] = person2_params["trans"] - centroid
            p1_pelvis = p1_pelvis - centroid
            p2_pelvis = p2_pelvis - centroid
            p1_pelvis_original = p1_pelvis - person1_params["trans"] # pelvis position in original smpl coords system
            p2_pelvis_original = p2_pelvis - person2_params["trans"] # pelvis position in original smpl coords system
            person1_params["trans"] = np.dot(person1_params["trans"] + p1_pelvis_original, rotation.T) - p1_pelvis_original
            p1_pelvis = np.dot(p1_pelvis, rotation.T)
            person2_params["trans"] = np.dot(person2_params["trans"] + p2_pelvis_original, rotation.T) - p2_pelvis_original
            p2_pelvis = np.dot(p2_pelvis, rotation.T)
            
            # human vertex in the canonical system
            p1_verts_tran = data["p1_verts"][i].copy()[:, :3] - centroid
            p1_verts_tran = np.dot(p1_verts_tran, rotation.T)
            p2_verts_tran = data["p2_verts"][i].copy()[:, :3] - centroid
            p2_verts_tran = np.dot(p2_verts_tran, rotation.T)

            # human vertex normal in the canonical system
            p1_verts_normal = np.dot(data["p1_verts"][i].copy()[:, 3:], rotation.T)
            p1_verts = np.concatenate([p1_verts_tran, p1_verts_normal], axis=1)
            p2_verts_normal = np.dot(data["p2_verts"][i].copy()[:, 3:], rotation.T)
            p2_verts = np.concatenate([p2_verts_tran, p2_verts_normal], axis=1)

            # smpl pose parameter in the canonical system
            r1_ori = Rotation.from_rotvec(person1_params["pose"][:3])
            r1_new = Rotation.from_matrix(rotation) * r1_ori
            person1_params["pose"][:3] = r1_new.as_rotvec()
            r2_ori = Rotation.from_rotvec(person2_params["pose"][:3])
            r2_new = Rotation.from_matrix(rotation) * r2_ori
            person2_params["pose"][:3] = r2_new.as_rotvec()

            # object in the canonical system
            objfit_params["trans"] = objfit_params["trans"] - centroid
            objfit_params["trans"] = np.dot(objfit_params["trans"], rotation.T)

            r_ori = Rotation.from_rotvec(objfit_params["angle"])
            r_new = Rotation.from_matrix(rotation) * r_ori
            objfit_params["angle"] = r_new.as_rotvec()
            
            # object pointcloud in the canonical system
            obj_points = data["obj_points"].copy()
            rot = r_new.as_matrix()
            obj_points[:, :3] = np.matmul(obj_points[:, :3], rot.T) + objfit_params["trans"]
            obj_points[:, 3:6] = np.matmul(obj_points[:, 3:6], rot.T)

            obj_contact_to_p1 = data["contact_object_to_person1"][i]
            label = np.zeros([obj_points.shape[0], 1])
            label[obj_contact_to_p1, 0] = 1
            obj_points = np.concatenate([obj_points, label], axis=1)
            obj_contact_to_p2 = data["contact_object_to_person2"][i]
            label = np.zeros([obj_points.shape[0], 1])
            label[obj_contact_to_p2, 0] = 1
            obj_points = np.concatenate([obj_points, label], axis=1)  # (2048, 8)

            p1_contact_label = np.zeros([self.num_verts, 1])
            p1_contact_label[data["contact_person1_to_object"][i], 0] = 1
            p1_verts = np.concatenate([p1_verts, p1_contact_label], axis=1)  # (10475, 4)
            p2_contact_label = np.zeros([self.num_verts, 1])
            p2_contact_label[data["contact_person2_to_object"][i], 0] = 1
            p2_verts = np.concatenate([p2_verts, p2_contact_label], axis=1)  # (10475, 4)

            # The label indicating if the foot is contacting ground
            p1_ground_joint_label = np.zeros([2])
            if i > 0:
                delta_left = np.linalg.norm(data["p1_left_foot"][i] - data["p1_left_foot"][i-1])
                delta_right = np.linalg.norm(data["p1_right_foot"][i] - data["p1_right_foot"][i-1])
                p1_ground_joint_label[0] = int(delta_left < 0.01)
                p1_ground_joint_label[1] = int(delta_right < 0.01)
            else:
                p1_ground_joint_label[data["foot_contact_label_person1"][i] - 10] = 1
            p2_ground_joint_label = np.zeros([2])
            if i > 0:
                delta_left = np.linalg.norm(data["p2_left_foot"][i] - data["p2_left_foot"][i-1])
                delta_right = np.linalg.norm(data["p2_right_foot"][i] - data["p2_right_foot"][i-1])
                p2_ground_joint_label[0] = int(delta_left < 0.01)
                p2_ground_joint_label[1] = int(delta_right < 0.01)
            else:
                p2_ground_joint_label[data["foot_contact_label_person2"][i] - 10] = 1
            
            # shape check
            assert person1_params["pose"].shape == (90,)
            assert person1_params["trans"].shape == (3,)
            assert person1_params["betas"].shape == (10,)
            assert person2_params["pose"].shape == (90,)
            assert person2_params["trans"].shape == (3,)
            assert person2_params["betas"].shape == (10,)
            assert p1_pelvis.shape == (3,)
            assert p2_pelvis.shape == (3,)
            assert obj_points.shape == (2048, 8)
            assert p1_contact_label.shape == (10475, 1)
            assert p2_contact_label.shape == (10475, 1)
            assert p1_ground_joint_label.shape == (2,)
            assert p2_ground_joint_label.shape == (2,)
            assert p1_verts.shape == (10475, 7)
            assert p2_verts.shape == (10475, 7)
            assert p1_verts[markerset_ssm67_smplx, :].shape == (67, 7)
            assert p2_verts[markerset_ssm67_smplx, :].shape == (67, 7)
            
            record = {
                'person1_params': person1_params,  # 当前帧person1相对第一帧person1的SMPLH theta, {"pose": shape = (3+21*3+12+12,), "betas": shape = (10,), "trans": shape = (3,)})
                'person2_params': person2_params,  # 当前帧person2相对第一帧person1的SMPLH theta, {"pose": shape = (3+21*3+12+12,), "betas": shape = (10,), "trans": shape = (3,)})
                'objfit_params': objfit_params,  # 当前帧物体相对第一帧person1的pose, {"trans": (3,), "angle": (3,)}
                'p1_pelvis': p1_pelvis,  # 当前帧person1的pelvis相对第一帧person1的3D位置, shape = (3,)
                'p2_pelvis': p2_pelvis,  # 当前帧person2的pelvis相对第一帧person1的3D位置, shape = (3,)
                'obj_points': obj_points,  # 当前帧物体相对第一帧person1的点云, shape = (2048, 8), 3D位置 + 3D法向 + 1D contact_to_person1 + 1D contact_to_person2
                'p1_contact_label': p1_contact_label,  # 当前帧person1顶点的contact label, shape = (10475, 1), value = 0/1
                'p2_contact_label': p2_contact_label,  # 当前帧person2顶点的contact label, shape = (10475, 1), value = 0/1
                'p1_ground_joint_label': p1_ground_joint_label,  # 当前帧person1的左/右脚是否接触地面, shape = (2,)
                'p2_ground_joint_label': p2_ground_joint_label,  # 当前帧person2的左/右脚是否接触地面, shape = (2,)
                'p1_verts': p1_verts,  # 当前帧person1相对第一帧person1的顶点, shape = (10475, 7), 3D位置 + 3D法向 + 1D contact_to_object
                'p2_verts': p2_verts,  # 当前帧person2相对第一帧person1的顶点, shape = (10475, 7), 3D位置 + 3D法向 + 1D contact_to_object
                'p1_markers': p1_verts[markerset_ssm67_smplx, :],  # 下采样了指定顶点的p1_verts, shape=(67, 7)
                'p2_markers': p2_verts[markerset_ssm67_smplx, :],  # 下采样了指定顶点的p2_verts, shape=(67, 7)
            }
            frames.append(record)
        
        records = {
            "centroid": centroid,  # 第一帧里person1 pelvis在世界系下的3D坐标, shape = (3,)
            "rotation": rotation,  # rotation_v的逆, shape = (3, 3)
            "rotation_v": rotation_v,  # 第一帧里person1的global orientation仅保留绕地面法向的旋转, shape = (3, 3)
            "frames": frames,  # 每一帧的数据, 定义见上面record的注释
            "obj_model_path": data["obj_model_path"],  # str
            "seq_name": data['clip_name'] + "-" + data['seq_name'],  # str, e.g.: 20231020-000
            "start_frame": start_frame,  # int
            "obj_points": data["obj_points"],  # 物体在canonical space下的点云, shape = (2048, 6), 3D位置+3D法向
        }
        return records
        

    def __len__(self):
        return len(self.idx2frame) * self.repeat_ratio


if __name__ == "__main__":
    dataset = Dataset(mode="train", past_len=15, future_len=15)

    x = dataset[0]["frames"]
    for i in range(len(x)):
        print(i, x[i]["objfit_params"])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x[0]["p1_markers"][:, :3])
    o3d.io.write_point_cloud("./test_smplx_marker_pcd.ply", pcd)
