import os
from os.path import join, isdir, isfile, dirname, abspath
import sys
sys.path.insert(0, join(dirname(abspath(__file__)), "../.."))
import numpy as np
import joblib 
import json 
import trimesh 
import time
from copy import deepcopy
import torch
from torch.utils.data import Dataset
import pytorch3d.transforms as transforms 
from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from bps_torch.tools import sample_uniform_cylinder
from human_body_prior.body_model.body_model import BodyModel
from manip.lafan1.utils import rotate_at_frame_w_obj, rotate_at_frame_w_obj_specify_frame
sys.path.insert(0, join(dirname(abspath(__file__)), "../../../.."))
from dataset_statistics.train_test_split import load_train_test_split
import open3d as o3d

SMPLH_PATH = "/data2/datasets/OMOMO_data/smpl_all_models/smplh_amass"

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)

def rotate(points, R):
    shape = list(points.shape)
    points = to_tensor(points)
    R = to_tensor(R)
    if len(shape)>3:
        points = points.squeeze()
    if len(shape)<3:
        points = points.unsqueeze(dim=1)
    if R.shape[0] > shape[0]:
        shape[0] = R.shape[0]
    r_points = torch.matmul(points, R.transpose(1,2))
    return r_points.reshape(shape)

def get_smpl_parents(use_joints24=False):
    """
    use_joint24=True: SMPLX的22个body joint + 左手食指root(24) + 右手食指root(40), 等价于SMPLH的[0, ..., 21, 22, 37]
    """
    
    bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 

    if use_joints24:
        parents = ori_kintree_table[0, :23] # 23
        parents[0] = -1 # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list) # 24 
    else:
        parents = ori_kintree_table[0, :22] # 22 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.
    
    return parents

def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3 
    kintree = get_smpl_parents()

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose # T X J X 3 X 3 

def quat_ik_torch(grot_mat):
    # grot: T X J X 3 X 3
    parents = get_smpl_parents()

    grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4

    res = torch.cat(
            [
                grot[..., :1, :],
                transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                grot[..., 1:, :]),
            ],
            dim=-2) # T X J X 4

    res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3

    return res_mat

def quat_fk_torch(lrot_mat, lpos, use_joints24=False):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J/(J+2) X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    if use_joints24:
        parents = get_smpl_parents(use_joints24=True)
    else:
        parents = get_smpl_parents()

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        if i < lrot.shape[-2]:
            gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res

def merge_two_parts(verts_list, faces_list):
    verts_num = 0
    merged_verts_list = []
    merged_faces_list = []
    for p_idx in range(len(verts_list)):
        # part_verts = torch.from_numpy(verts_list[p_idx]) # T X Nv X 3
        part_verts = verts_list[p_idx] # T X Nv X 3
        part_faces = torch.from_numpy(faces_list[p_idx]) # T X Nf X 3

        if p_idx == 0:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces)
        else:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces+verts_num)

        verts_num += part_verts.shape[1]

    # merged_verts = torch.cat(merged_verts_list, dim=1).data.cpu().numpy()
    merged_verts = torch.cat(merged_verts_list, dim=1)
    merged_faces = torch.cat(merged_faces_list, dim=0).data.cpu().numpy()

    return merged_verts, merged_faces

    
class HHODataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder="/data2/datasets/hhodataset/prepared_motion_forecasting_data",
        human_model_folder="/data2/datasets/OMOMO_data/smpl_all_models",
        window=120,
        use_object_splits=False,
    ):
        self.train = train
        
        self.window = window

        self.use_joints24 = True
        
        # liuyun
        if use_object_splits:
            raise NotImplementedError

        self.use_object_splits = use_object_splits
        
        # self.train_dates = ["20231002", "20231003_1", "20231003_2", "20231011", "20231020", "20231023", "20231108"]
        # self.test_dates = ["20231008", "20231018", "20231030"]
        self.train_sequence_names, self.test_sequence_names_seen_obj, self.test_sequence_names_unseen_obj = load_train_test_split()

        self.parents = get_smpl_parents()  # 22

        self.data_root_folder = data_root_folder

        self.bps_path = join(dirname(abspath(__file__)), "../../manip/data/bps.pt")

        dest_obj_bps_npy_folder = os.path.join(data_root_folder, "object_bps_npy_files_joints24")
        dest_obj_bps_npy_folder_for_test = os.path.join(data_root_folder, "object_bps_npy_files_for_eval_joints24")
        os.makedirs(dest_obj_bps_npy_folder, exist_ok=True)
        os.makedirs(dest_obj_bps_npy_folder_for_test, exist_ok=True)

        if self.train:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder
        else:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder_for_test 

        if self.train:
            processed_data_path = os.path.join(data_root_folder, "train_diffusion_manip_window_"+str(self.window)+"_cano_joints24.p")
        else:
            processed_data_path = os.path.join(data_root_folder, "test_diffusion_manip_window_"+str(self.window)+"_processed_joints24.p")
           
        min_max_mean_std_data_path = os.path.join(data_root_folder, "min_max_mean_std_data_window_"+str(self.window)+"_cano_joints24.p")
        
        self.prep_bps_data()
        
        # Prepare SMPLX model 
        soma_work_base_dir = human_model_folder
        support_base_dir = soma_work_base_dir 
        surface_model_type = "smplx"
        surface_model_neutral_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_NEUTRAL.npz")
        dmpl_fname = None
        num_dmpls = None
        num_expressions = None
        num_betas = 10

        self.neutral_bm = BodyModel(bm_fname=surface_model_neutral_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)
        for p in self.neutral_bm.parameters():
            p.requires_grad = False
        self.neutral_bm = self.neutral_bm.cuda()
        self.bm_dict = {"neutral": self.neutral_bm}

        if os.path.exists(processed_data_path):  # 使用预处理过的训测数据
        # if False:
            self.window_data_dict = joblib.load(processed_data_path)
        else:
            # date_list = self.train_dates if self.train else self.test_dates
            sequence_names = self.train_sequence_names if self.train else (self.test_sequence_names_seen_obj + self.test_sequence_names_unseen_obj)
            self.data_dict = self.load_data(data_root_folder, sequence_name_list=sequence_names, date_list=None)
            
            self.cal_normalize_data_input()
            joblib.dump(self.window_data_dict, processed_data_path)            

        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        else:
            if self.train:
                min_max_mean_std_jpos_data = self.extract_min_max_mean_std_from_data()
                joblib.dump(min_max_mean_std_jpos_data, min_max_mean_std_data_path)
        
        self.global_jpos_boundary = {}
        for person in ["person1", "person2"]:
            self.global_jpos_boundary[person] = {}
            self.global_jpos_boundary[person]["global_jpos_min"] = torch.from_numpy(min_max_mean_std_jpos_data[person]['global_jpos_min']).float().reshape(24, 3)[None]
            self.global_jpos_boundary[person]["global_jpos_max"] = torch.from_numpy(min_max_mean_std_jpos_data[person]['global_jpos_max']).float().reshape(24, 3)[None]

        # Get train and validation statistics.
        if self.train:
            print("Total number of windows for training:{0}".format(len(self.window_data_dict)))
        else:
            print("Total number of windows for validation:{0}".format(len(self.window_data_dict)))
    
    def load_data(self, data_root_folder, sequence_name_list=None, date_list=None):
        """
        for HHO dataset, prepare self.data_dict
        self.data_dict: raw data, a list containing motion sequences
        self.data_dict[0]:
            * seq_name: str
            * betas: SMPLX beta, shape = (1, 16)
            * gender: SMPLX gender: "male" / "female"
            * trans: SMPLX root position for each frame, shape = (T, 3)
            * root_orient: SMPLX root rotation for each frame, shape = (T, 3)
            * pose_body: SMPLX body_pose, shape = (T, 63)
            * rest_offsets: T-POSE下各个joint相对父节点的3D translation, root对应[0,0,0], shape = (24, 3)
            * trans2joint: T-POSE下root的位置的负值, shape = (3,)
            * obj_trans: object translation, shape = (T, 3, 1)
            * obj_rot: object rotation, shape = (T, 3, 3)
            * obj_scale: object scale, shape = (T,)
            * obj_com_pos: TODO, 应该设置成物体bbox中心或者几何中心均可, 最好和translation一致, shape = (T, 3)
        """
        
        # self.smplx_model = smplx.create("/share/human_model/models", model_type="smplx", gender="neutral", batch_size=1, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=True, num_pca_comps=12, flat_hand_mean=True)
        
        if not sequence_name_list is None:
            seq_dirs = []
            for sn in sequence_name_list:
                seq_dir = join(data_root_folder, sn.replace(".", "/"))
                if isfile(join(seq_dir, "data.npz")):
                    seq_dirs.append(seq_dir)
        elif not date_list is None:
            seq_dirs = []
            for date in date_list:
                date_dir = join(data_root_folder, date)
                if not isdir(date_dir):
                    continue
                for seq_name in os.listdir(date_dir):
                    seq_dir = join(date_dir, seq_name)
                    if not isfile(join(seq_dir, "data.npz")):
                        continue
                    seq_dirs.append(seq_dir)
        else:
            raise NotImplementedError
        
        seq_dirs.sort()
        
        data_dict = []
        seq_cnt = 0
        for seq_dir in seq_dirs:
            date, seq_name = seq_dir.split("/")[-2:]
            data = np.load(join(seq_dir, "data.npz"), allow_pickle=True)["data"].item()
                
            print("processing {} ...".format(seq_dir))
            
            prepared_data = {}
            prepared_data["seq_name"] = date + "." + seq_name
            prepared_data["obj_model_path"] = data["obj_model_path"].replace("/data3/", "/data2/")
            if not "_simplified" in prepared_data["obj_model_path"]:
                prepared_data["obj_model_path"] = prepared_data["obj_model_path"].replace("HHO_object_dataset_final", "HHO_object_dataset_final_simplified")
            T = data["obj_poses"].shape[0]
            for person in ["person1", "person2"]:
                prepared_data[person] = {}
                prepared_data[person]["betas"] = data["human_params"][person]["betas"][0].reshape(1, -1)  # (1, 10)
                prepared_data[person]["gender"] = "neutral"
                prepared_data[person]["trans"] = data["human_params"][person]["transl"]  # (T, 3)
                prepared_data[person]["root_orient"] = data["human_params"][person]["global_orient"]  # (T, 3)
                prepared_data[person]["pose_body"] = data["human_params"][person]["body_pose"].reshape(T, 63)  # (T, 63)
                    
                TPOSE_info = self.bm_dict["neutral"](betas=torch.from_numpy(prepared_data[person]["betas"]).cuda())
                joints = TPOSE_info.Jtr[0]
                    
                # # liuyun: debug
                # print(joints.shape)
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(joints.detach().cpu().numpy())
                # o3d.io.write_point_cloud("./TPOSE.ply", pcd)
                # print(self.parents)
                    
                rest_offsets = [[0,0,0]]
                for i in range(1, 22):
                    rest_offsets.append(list((joints[i] - joints[self.parents[i]]).detach().cpu().numpy()))
                rest_offsets.append(list((joints[25] - joints[20]).detach().cpu().numpy()))
                rest_offsets.append(list((joints[40] - joints[21]).detach().cpu().numpy()))
                rest_offsets = np.float32(rest_offsets)
                    
                prepared_data[person]["rest_offsets"] = rest_offsets
                prepared_data[person]["trans2joint"] = - joints[0].detach().cpu().numpy()
                
            prepared_data["obj_trans"] = data["obj_poses"][:, :3, 3:]  # (T, 3, 1)
            prepared_data["obj_rot"] = data["obj_poses"][:, :3, :3]  # (T, 3, 3)
            prepared_data["obj_scale"] = np.ones(T)
            prepared_data["obj_com_pos"] = data["obj_poses"][:, :3, 3]  # 直接设置成translation, (T, 3)
            prepared_data["obj_verts"] = data["object_points"][:, :3]  # canonical space下的object pcd, (N_point, 3)
                
            data_dict.append(prepared_data)
                
            seq_cnt += 1
            # if (seq_cnt == 20):  # TODO: remove this
            #     return data_dict
        
        return data_dict

    def apply_transformation_to_obj_geometry(self, mesh, obj_scale, obj_rot, obj_trans):
        obj_mesh_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3 

        ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1) # T X Nv X 3 
    
        seq_scale = torch.from_numpy(obj_scale).float() # T 
        seq_rot_mat = torch.from_numpy(obj_rot).float() # T X 3 X 3 
        if obj_trans.shape[-1] != 1:
            seq_trans = torch.from_numpy(obj_trans).float()[:, :, None] # T X 3 X 1 
        else:
            seq_trans = torch.from_numpy(obj_trans).float() # T X 3 X 1 
        transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
        seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2)) + seq_trans
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 
        
        return transformed_obj_verts, obj_mesh_faces

    def compute_object_geo_bps(self, obj_verts, obj_trans):
        # obj_verts: T X Nv X 3, obj_trans: T X 3
        bps_object_geo = self.bps_torch.encode(x=obj_verts, \
                    feature_type=['deltas'], \
                    custom_basis=self.obj_bps.repeat(obj_trans.shape[0], \
                    1, 1)+obj_trans[:, None, :])['deltas'] # T X N X 3 

        return bps_object_geo

    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 1.0
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(1, -1, 3)
            
            bps = {
                'obj': bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            print("Generate new bps data to:{0}".format(self.bps_path))
            torch.save(bps, self.bps_path)
        
        self.bps = torch.load(self.bps_path)

        self.bps_torch = bps_torch()

        self.obj_bps = self.bps['obj']

    # def get_bps_from_window_data_dict(self):
    #     # Given window_data_dict which contains canonizalized information, compute its corresponding BPS representation. 
    #     for k in self.window_data_dict:
    #         window_data = self.window_data_dict[k]

    #         seq_name = window_data['seq_name']
    #         object_name = seq_name.split("_")[1]

    #         curr_obj_scale = window_data['obj_scale']
    #         new_obj_x = window_data['obj_trans']
    #         new_obj_rot_mat = window_data['obj_rot_mat']

    #         # Get object geometry 
    #         if object_name in ["mop", "vacuum"]:
    #             curr_obj_bottom_scale = window_data['obj_bottom_scale']
    #             new_obj_bottom_x = window_data['obj_bottom_trans']
    #             new_obj_bottom_rot_mat = window_data['obj_bottom_rot_mat']

    #             obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale, \
    #                     new_obj_x, new_obj_rot_mat, \
    #                     curr_obj_bottom_scale, new_obj_bottom_x, \
    #                     new_obj_bottom_rot_mat) # T X Nv X 3, tensor

    #         else:
    #             obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale, \
    #                         new_obj_x, new_obj_rot_mat) # T X Nv X 3, tensor

    #         center_verts = obj_verts.mean(dim=1) # T X 3
    #         dest_obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+str(k)+".npy")

    #         if not os.path.exists(dest_obj_bps_npy_path):
    #             object_bps = self.compute_object_geo_bps(obj_verts, center_verts)
    #             np.save(dest_obj_bps_npy_path, object_bps.data.cpu().numpy())

    #     import pdb 
    #     pdb.set_trace() 

    def cal_normalize_data_input(self):
        
        self.window_data_dict = {}
        s_idx = 0
        for index in range(len(self.data_dict)):
            seq_name = self.data_dict[index]['seq_name']

            human_info = {}
            for person in ["person1", "person2"]:
                human_info[person] = {}
                human_info[person]["betas"] = self.data_dict[index][person]['betas']  # (1, 10)
                human_info[person]["gender"] = self.data_dict[index][person]['gender']
                human_info[person]["seq_root_trans"] = self.data_dict[index][person]['trans']  # (T, 3)
                human_info[person]["seq_root_orient"] = self.data_dict[index][person]['root_orient']  # (T, 3)
                human_info[person]["seq_pose_body"] = self.data_dict[index][person]['pose_body'].reshape(-1, 21, 3)  # (T, 21, 3)
                human_info[person]["rest_human_offsets"] = self.data_dict[index][person]['rest_offsets']  # (24, 3)
                human_info[person]["trans2joint"] = self.data_dict[index][person]['trans2joint']  # (3,)

            obj_trans = self.data_dict[index]['obj_trans'][:, :, 0]  # (T, 3)
            obj_rot = self.data_dict[index]['obj_rot']  # (T, 3, 3)
            obj_scale = self.data_dict[index]['obj_scale']  # (T,)
            obj_com_pos = self.data_dict[index]['obj_com_pos']  # (T, 3)
            canonical_obj_verts = self.data_dict[index]['obj_verts']  # (N_point, 3)
           
            num_steps = obj_trans.shape[0]
            for start_t_idx in range(0, num_steps, self.window//2):
                end_t_idx = min(start_t_idx + self.window - 1, num_steps)

                # Skip the segment that has a length < 30 
                if end_t_idx - start_t_idx < 30:
                    continue 

                self.window_data_dict[s_idx] = {}
                self.window_data_dict[s_idx]['seq_name'] = seq_name
                self.window_data_dict[s_idx]['obj_model_path'] = self.data_dict[index]['obj_model_path']
                self.window_data_dict[s_idx]['start_t_idx'] = start_t_idx
                self.window_data_dict[s_idx]['end_t_idx'] = end_t_idx
                
                obj_x = obj_trans[start_t_idx:end_t_idx+1].copy() # T X 3
                obj_rot_mat = torch.from_numpy(obj_rot[start_t_idx:end_t_idx+1]).float()# T X 3 X 3
                obj_q = transforms.matrix_to_quaternion(obj_rot_mat).detach().cpu().numpy() # T X 4
                window_obj_com_pos = obj_com_pos[start_t_idx:end_t_idx+1].copy() # T X 3
                
                # 把所有数据对齐到person1的第start_t_idx帧下的global 6D pose坐标系上(特别地, 地面不上移)
                person1_X, person1_Q, person1_new_seq_root_trans = None, None, None
                for person in ["person1", "person2"]:
                    betas = human_info[person]["betas"]
                    gender = human_info[person]["gender"]
                    seq_root_orient = human_info[person]["seq_root_orient"]
                    seq_pose_body = human_info[person]["seq_pose_body"]
                    rest_human_offsets = human_info[person]["rest_human_offsets"]
                    seq_root_trans = human_info[person]["seq_root_trans"]
                    trans2joint = human_info[person]["trans2joint"]
                    
                    # Canonicalize the first frame's orientation.
                    joint_aa_rep = torch.cat((torch.from_numpy(seq_root_orient[start_t_idx:end_t_idx+1]).float()[:, None, :], \
                        torch.from_numpy(seq_pose_body[start_t_idx:end_t_idx+1]).float()), dim=1) # T X J X 3 
                    X = torch.from_numpy(rest_human_offsets).float()[None].repeat(joint_aa_rep.shape[0], 1, 1).detach().cpu().numpy() # T X J X 3 
                    X[:, 0, :] = seq_root_trans[start_t_idx:end_t_idx+1] 
                    local_rot_mat = transforms.axis_angle_to_matrix(joint_aa_rep) # T X J X 3 X 3 
                    Q = transforms.matrix_to_quaternion(local_rot_mat).detach().cpu().numpy() # T X J X 4
                    
                    if person == "person1":
                        person1_X = deepcopy(X)
                        person1_Q = deepcopy(Q)

                    curr_obj_scale = torch.from_numpy(obj_scale[start_t_idx:end_t_idx+1]).float() # T
                    
                    _, _, new_obj_x, new_obj_q = rotate_at_frame_w_obj_specify_frame(X[np.newaxis], Q[np.newaxis], person1_X[np.newaxis], person1_Q[np.newaxis], \
                    obj_x[np.newaxis], obj_q[np.newaxis], \
                    trans2joint[np.newaxis], self.parents, n_past=1, floor_z=False)
                    # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4
                
                    X, Q, new_obj_com_pos, _ = rotate_at_frame_w_obj_specify_frame(X[np.newaxis], Q[np.newaxis], person1_X[np.newaxis], person1_Q[np.newaxis], \
                    window_obj_com_pos[np.newaxis], obj_q[np.newaxis], \
                    trans2joint[np.newaxis], self.parents, n_past=1, floor_z=False)
                    # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4

                    new_seq_root_trans = X[0, :, 0, :] # T X 3 
                    new_local_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(Q[0]).float()) # T X J X 3 X 3 
                    new_local_aa_rep = transforms.matrix_to_axis_angle(new_local_rot_mat) # T X J X 3 
                    new_seq_root_orient = new_local_aa_rep[:, 0, :] # T X 3
                    new_seq_pose_body = new_local_aa_rep[:, 1:, :] # T X 21 X 3 
                    
                    if person == "person1":
                        person1_new_seq_root_trans = new_seq_root_trans.copy()
                    
                    new_obj_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(new_obj_q[0]).float()) # T X 3 X 3
                    cano_obj_mat = torch.matmul(new_obj_rot_mat[0], obj_rot_mat[0].transpose(0, 1)) # 3 X 3
                    obj_verts = canonical_obj_verts[None, ...].repeat(new_obj_rot_mat.shape[0], axis=0).copy()  # in canonical space, (T, N_point, 3)
                    obj_verts = ((obj_verts * curr_obj_scale.detach().cpu().numpy().reshape(-1, 1, 1)) @ new_obj_rot_mat.detach().cpu().numpy().transpose(0, 2, 1)) + new_obj_x[0][:, None, :]  # in the person's coordinate system from the first frame, (T, N_point, 3)
                    obj_verts = torch.from_numpy(obj_verts)
                    center_verts = obj_verts.mean(dim=1)  # (T X 3)
                    
                    query = self.process_window_data(person1_new_seq_root_trans, rest_human_offsets, trans2joint, \
                            new_seq_root_trans, new_seq_root_orient.detach().cpu().numpy(), \
                            new_seq_pose_body.detach().cpu().numpy(),  \
                            new_obj_x[0], new_obj_rot_mat.detach().cpu().numpy(), \
                            curr_obj_scale.detach().cpu().numpy(), new_obj_com_pos[0], center_verts)

                    # Compute BPS representation for this window
                    # Save to numpy file
                    if person == "person1":
                        dest_obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+str(s_idx)+".npy")
                        if not os.path.exists(dest_obj_bps_npy_path):
                            object_bps = self.compute_object_geo_bps(obj_verts, center_verts)
                            np.save(dest_obj_bps_npy_path, object_bps.data.cpu().numpy())

                    self.window_data_dict[s_idx]['cano_obj_mat'] = cano_obj_mat.detach().cpu().numpy()
                    self.window_data_dict[s_idx][person] = {}

                    curr_global_jpos = query['global_jpos'].detach().cpu().numpy()
                    curr_global_jvel = query['global_jvel'].detach().cpu().numpy()
                    curr_global_rot_6d = query['global_rot_6d'].detach().cpu().numpy()
                
                    self.window_data_dict[s_idx][person]['motion'] = np.concatenate((curr_global_jpos.reshape(-1, 24*3), \
                    curr_global_jvel.reshape(-1, 24*3), curr_global_rot_6d.reshape(-1, 22*6)), axis=1)  # (T', 24*3+24*3+22*6)

                    self.window_data_dict[s_idx][person]['betas'] = betas
                    self.window_data_dict[s_idx][person]['gender'] = gender

                    self.window_data_dict[s_idx][person]['trans2joint'] = trans2joint

                    self.window_data_dict[s_idx][person]['obj_trans'] = query['obj_trans'].detach().cpu().numpy()
                    self.window_data_dict[s_idx][person]['obj_rot_mat'] = query['obj_rot_mat'].detach().cpu().numpy()
                    self.window_data_dict[s_idx][person]['obj_scale'] = query['obj_scale'].detach().cpu().numpy()

                    self.window_data_dict[s_idx][person]['obj_com_pos'] = query['obj_com_pos'].detach().cpu().numpy()
                    self.window_data_dict[s_idx][person]['window_obj_com_pos'] = query['window_obj_com_pos'].detach().cpu().numpy()

                s_idx += 1
       
    def extract_min_max_mean_std_from_data(self):
        stats_dict = {}

        for person in ["person1", "person2"]:
            all_global_jpos_data = []
            all_global_jvel_data = []
            for s_idx in self.window_data_dict:
                curr_window_data = self.window_data_dict[s_idx][person]['motion'] # T X D 

                all_global_jpos_data.append(curr_window_data[:, :24*3])
                all_global_jvel_data.append(curr_window_data[:, 24*3:2*24*3])

            all_global_jpos_data = np.vstack(all_global_jpos_data).reshape(-1, 72)  # (N*T', 72)
            all_global_jvel_data = np.vstack(all_global_jvel_data).reshape(-1, 72)

            min_jpos = all_global_jpos_data.min(axis=0)
            max_jpos = all_global_jpos_data.max(axis=0)
            min_jvel = all_global_jvel_data.min(axis=0)
            max_jvel = all_global_jvel_data.max(axis=0)

            stats_dict[person] = {}
            stats_dict[person]['global_jpos_min'] = min_jpos
            stats_dict[person]['global_jpos_max'] = max_jpos
            stats_dict[person]['global_jvel_min'] = min_jvel
            stats_dict[person]['global_jvel_max'] = max_jvel

        # print(stats_dict)
        return stats_dict

    def normalize_jpos_min_max(self, ori_jpos, person):
        # ori_jpos: T X 22/24 X 3 
        normalized_jpos = (ori_jpos - self.global_jpos_boundary[person]["global_jpos_min"].to(ori_jpos.device)) / (self.global_jpos_boundary[person]["global_jpos_max"].to(ori_jpos.device) - self.global_jpos_boundary[person]["global_jpos_min"].to(ori_jpos.device))
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 

        return normalized_jpos # T X 22/24 X 3 

    def de_normalize_jpos_min_max(self, normalized_jpos, person):
        """
        normalized_jpos: (T, 24, 3)
        """
        normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range
        de_jpos = normalized_jpos * (self.global_jpos_boundary[person]["global_jpos_max"].to(normalized_jpos.device)-\
        self.global_jpos_boundary[person]["global_jpos_min"].to(normalized_jpos.device)) + self.global_jpos_boundary[person]["global_jpos_min"].to(normalized_jpos.device)

        return de_jpos  # (T, 24, 3)

    def normalize_jpos_min_max_hand_foot(self, ori_jpos, hand_only=True):
        # ori_jpos: (B, T, 2+2, 3)
        lhand_idx = 22 
        rhand_idx = 23

        lfoot_idx = 10
        rfoot_idx = 11 

        bs = ori_jpos.shape[0] 
        num_steps = ori_jpos.shape[1] 
        ori_jpos = ori_jpos.reshape(bs, num_steps, -1)  # (B, T, 12)

        assert hand_only
        p1_hand_foot_jpos_max = torch.cat((self.global_jpos_boundary["person1"]["global_jpos_max"][0, lhand_idx], self.global_jpos_boundary["person1"]["global_jpos_max"][0, rhand_idx]), dim=0)
        p1_hand_foot_jpos_min = torch.cat((self.global_jpos_boundary["person1"]["global_jpos_min"][0, lhand_idx], self.global_jpos_boundary["person1"]["global_jpos_min"][0, rhand_idx]), dim=0)
        p2_hand_foot_jpos_max = torch.cat((self.global_jpos_boundary["person2"]["global_jpos_max"][0, lhand_idx], self.global_jpos_boundary["person2"]["global_jpos_max"][0, rhand_idx]), dim=0)
        p2_hand_foot_jpos_min = torch.cat((self.global_jpos_boundary["person2"]["global_jpos_min"][0, lhand_idx], self.global_jpos_boundary["person2"]["global_jpos_min"][0, rhand_idx]), dim=0)

        p1_hand_foot_jpos_max = p1_hand_foot_jpos_max[None, None]
        p1_hand_foot_jpos_min = p1_hand_foot_jpos_min[None, None]
        p2_hand_foot_jpos_max = p2_hand_foot_jpos_max[None, None]
        p2_hand_foot_jpos_min = p2_hand_foot_jpos_min[None, None]
        hand_foot_jpos_max = torch.cat((p1_hand_foot_jpos_max, p2_hand_foot_jpos_max), dim=-1)  # (1, 1, 12)
        hand_foot_jpos_min = torch.cat((p1_hand_foot_jpos_min, p2_hand_foot_jpos_min), dim=-1)  # (1, 1, 12)
        normalized_jpos = (ori_jpos - hand_foot_jpos_min.to(ori_jpos.device))/(hand_foot_jpos_max.to(ori_jpos.device)\
        -hand_foot_jpos_min.to(ori_jpos.device))
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 

        normalized_jpos = normalized_jpos.reshape(bs, num_steps, -1, 3)

        return normalized_jpos  # (B, T, 2+2, 3)

    def de_normalize_jpos_min_max_hand_foot(self, normalized_jpos, hand_only=True):
        """
        for two persons
        normalized_jpos: (B, T, 2*3+2*3), hand_only=True
        """

        lhand_idx = 22
        rhand_idx = 23 
       
        lfoot_idx = 10
        rfoot_idx = 11 

        bs, num_steps, _ = normalized_jpos.shape 

        normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range

        assert hand_only
        p1_hand_foot_jpos_max = torch.cat((self.global_jpos_boundary["person1"]["global_jpos_max"][0, lhand_idx], self.global_jpos_boundary["person1"]["global_jpos_max"][0, rhand_idx]), dim=0)
        p1_hand_foot_jpos_min = torch.cat((self.global_jpos_boundary["person1"]["global_jpos_min"][0, lhand_idx], self.global_jpos_boundary["person1"]["global_jpos_min"][0, rhand_idx]), dim=0)
        p2_hand_foot_jpos_max = torch.cat((self.global_jpos_boundary["person2"]["global_jpos_max"][0, lhand_idx], self.global_jpos_boundary["person2"]["global_jpos_max"][0, rhand_idx]), dim=0)
        p2_hand_foot_jpos_min = torch.cat((self.global_jpos_boundary["person2"]["global_jpos_min"][0, lhand_idx], self.global_jpos_boundary["person2"]["global_jpos_min"][0, rhand_idx]), dim=0)

        p1_hand_foot_jpos_max = p1_hand_foot_jpos_max[None, None]
        p1_hand_foot_jpos_min = p1_hand_foot_jpos_min[None, None]
        p2_hand_foot_jpos_max = p2_hand_foot_jpos_max[None, None]
        p2_hand_foot_jpos_min = p2_hand_foot_jpos_min[None, None]
        hand_foot_jpos_max = torch.cat((p1_hand_foot_jpos_max, p2_hand_foot_jpos_max), dim=-1)  # (1, 1, 12)
        hand_foot_jpos_min = torch.cat((p1_hand_foot_jpos_min, p2_hand_foot_jpos_min), dim=-1)  # (1, 1, 12)

        de_jpos = normalized_jpos * (hand_foot_jpos_max.to(normalized_jpos.device)-\
        hand_foot_jpos_min.to(normalized_jpos.device)) + hand_foot_jpos_min.to(normalized_jpos.device)

        return de_jpos.reshape(bs, num_steps, -1, 3)  # (B, T, 2+2, 3)

    def process_window_data(self, anchor_trans, rest_human_offsets, trans2joint, seq_root_trans, seq_root_orient, seq_pose_body, \
        obj_trans, obj_rot, obj_scale, obj_com_pos, center_verts):
        random_t_idx = 0 
        end_t_idx = seq_root_trans.shape[0] - 1

        anchor_trans = torch.from_numpy(anchor_trans[random_t_idx:end_t_idx+1]).cuda()

        window_root_trans = torch.from_numpy(seq_root_trans[random_t_idx:end_t_idx+1]).cuda()
        window_root_orient = torch.from_numpy(seq_root_orient[random_t_idx:end_t_idx+1]).float().cuda()
        window_pose_body  = torch.from_numpy(seq_pose_body[random_t_idx:end_t_idx+1]).float().cuda()

        window_obj_scale = torch.from_numpy(obj_scale[random_t_idx:end_t_idx+1]).float().cuda() # T

        window_obj_rot_mat = torch.from_numpy(obj_rot[random_t_idx:end_t_idx+1]).float().cuda() # T X 3 X 3 
        window_obj_trans = torch.from_numpy(obj_trans[random_t_idx:end_t_idx+1]).float().cuda() # T X 3

        window_obj_com_pos = torch.from_numpy(obj_com_pos[random_t_idx:end_t_idx+1]).float().cuda() # T X 3
        window_center_verts = center_verts[random_t_idx:end_t_idx+1].to(window_obj_com_pos.device)

        # move_to_zero_trans = window_root_trans[0:1, :].clone()  # (1, 3)
        move_to_zero_trans = anchor_trans[0:1, :].clone()  # (1, 3)
        move_to_zero_trans[:, 1] = 0  # HHO数据集的y轴垂直于地面

        # Move motion and object translation to make the initial pose trans 0. 
        window_root_trans = window_root_trans - move_to_zero_trans 
        window_obj_trans = window_obj_trans - move_to_zero_trans 
        window_obj_com_pos = window_obj_com_pos - move_to_zero_trans 
        window_center_verts = window_center_verts - move_to_zero_trans 

        window_root_rot_mat = transforms.axis_angle_to_matrix(window_root_orient) # T' X 3 X 3 
        window_root_quat = transforms.matrix_to_quaternion(window_root_rot_mat)

        window_pose_rot_mat = transforms.axis_angle_to_matrix(window_pose_body) # T' X 21 X 3 X 3 

        # Generate global joint rotation 
        local_joint_rot_mat = torch.cat((window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1) # T' X 22 X 3 X 3 
        global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 
        global_joint_rot_quat = transforms.matrix_to_quaternion(global_joint_rot_mat) # T' X 22 X 4 

        curr_seq_pose_aa = torch.cat((window_root_orient[:, None, :], window_pose_body), dim=1)  # (T', 22, 3)
        rest_human_offsets = torch.from_numpy(rest_human_offsets).float()[None]  # (1, 24, 3)
        curr_seq_local_jpos = rest_human_offsets.repeat(curr_seq_pose_aa.shape[0], 1, 1).cuda()  # (T', 24, 3) 
        curr_seq_local_jpos[:, 0, :] = window_root_trans - torch.from_numpy(trans2joint).cuda()[None]  # (T', 24, 3)

        local_joint_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa)
        _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos, use_joints24=True)

        global_jpos = human_jnts  # (T', 24, 3)
        global_jvel = global_jpos[1:] - global_jpos[:-1]  # (T'-1, 24, 3)

        global_joint_rot_mat = local2global_pose(local_joint_rot_mat)  # (T', 22, 3, 3)

        local_rot_6d = transforms.matrix_to_rotation_6d(local_joint_rot_mat)
        global_rot_6d = transforms.matrix_to_rotation_6d(global_joint_rot_mat)

        query = {}
        query['local_rot_mat'] = local_joint_rot_mat  # (T', 22, 3, 3)
        query['local_rot_6d'] = local_rot_6d  # (T', 22, 6)
        query['global_jpos'] = global_jpos  # (T', 24, 3)
        query['global_jvel'] = torch.cat((global_jvel, torch.zeros(1, global_jvel.shape[1], 3).to(global_jvel.device)), dim=0)  # (T', 24, 3)
        query['global_rot_mat'] = global_joint_rot_mat  # (T', 22, 3, 3)
        query['global_rot_6d'] = global_rot_6d  # (T', 22, 6)
        query['obj_trans'] = window_obj_trans  # (T', 3)
        query['obj_rot_mat'] = window_obj_rot_mat  # (T', 3, 3)
        query['obj_scale'] = window_obj_scale  # (T',)
        query['obj_com_pos'] = window_obj_com_pos  # 这个字段没用在__getitem__中, (T', 3)
        query['window_obj_com_pos'] = window_center_verts  # (T', 3)

        return query 

    def __len__(self):
        return len(self.window_data_dict)

    def __getitem__(self, index):
        
        seq_name = self.window_data_dict[index]['seq_name']
        obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+str(index)+".npy")
        obj_bps_data = np.load(obj_bps_npy_path)  # (T, 1024, 3)
        obj_bps_data = torch.from_numpy(obj_bps_data)
        num_joints = 24
        
        data_input_dict = {}
        data_input_dict['seq_name'] = seq_name
        data_input_dict['obj_model_path'] = self.window_data_dict[index]['obj_model_path']
        # data_input_dict['obj_name'] = ?
        
        for person in ["person1", "person2"]:
            data_input = self.window_data_dict[index][person]['motion']
            data_input = torch.from_numpy(data_input).float()
            trans2joint = self.window_data_dict[index][person]['trans2joint']

            normalized_jpos = self.normalize_jpos_min_max(data_input[:, :num_joints*3].reshape(-1, num_joints, 3), person) # T X 22 X 3
       
            global_joint_rot = data_input[:, 2*num_joints*3:] # T X (22*6)

            new_data_input = torch.cat((normalized_jpos.reshape(-1, num_joints*3), global_joint_rot), dim=1)
            ori_data_input = torch.cat((data_input[:, :num_joints*3], global_joint_rot), dim=1)

            # Add padding. 
            actual_steps = new_data_input.shape[0]
            if actual_steps < self.window:
                paded_new_data_input = torch.cat((new_data_input, torch.zeros(self.window-actual_steps, new_data_input.shape[-1])), dim=0)
                paded_ori_data_input = torch.cat((ori_data_input, torch.zeros(self.window-actual_steps, ori_data_input.shape[-1])), dim=0)  

                paded_obj_bps = torch.cat((obj_bps_data.reshape(actual_steps, -1), \
                    torch.zeros(self.window-actual_steps, obj_bps_data.reshape(actual_steps, -1).shape[1])), dim=0)
                paded_obj_com_pos = torch.cat((torch.from_numpy(self.window_data_dict[index][person]['window_obj_com_pos']).float(), \
                    torch.zeros(self.window-actual_steps, 3)), dim=0)
           
                paded_obj_rot_mat = torch.cat((torch.from_numpy(self.window_data_dict[index][person]['obj_rot_mat']).float(), \
                    torch.zeros(self.window-actual_steps, 3, 3)), dim=0)
                paded_obj_scale = torch.cat((torch.from_numpy(self.window_data_dict[index][person]['obj_scale']).float(), \
                    torch.zeros(self.window-actual_steps,)), dim=0)
                paded_obj_trans = torch.cat((torch.from_numpy(self.window_data_dict[index][person]['obj_trans']).float(), \
                    torch.zeros(self.window-actual_steps, 3)), dim=0)

            else:
                paded_new_data_input = new_data_input 
                paded_ori_data_input = ori_data_input 

                paded_obj_bps = obj_bps_data.reshape(new_data_input.shape[0], -1)
                paded_obj_com_pos = torch.from_numpy(self.window_data_dict[index][person]['window_obj_com_pos']).float()
        
                paded_obj_rot_mat = torch.from_numpy(self.window_data_dict[index][person]['obj_rot_mat']).float()
                paded_obj_scale = torch.from_numpy(self.window_data_dict[index][person]['obj_scale']).float()
                paded_obj_trans = torch.from_numpy(self.window_data_dict[index][person]['obj_trans']).float()
    
            data_input_dict['seq_len'] = actual_steps
            data_input_dict['obj_bps'] = paded_obj_bps
            
            if person == "person1":  # person1
                data_input_dict['motion'] = paded_new_data_input  # normalized joint positions + global rotations, shape = (T, 24*3+22*6)
                data_input_dict['ori_motion'] = paded_ori_data_input  # original joint positions + global rotations, shape = (T, 24*3+22*6)
                data_input_dict['obj_com_pos'] = paded_obj_com_pos  # (T, 3)
                data_input_dict['obj_rot_mat'] = paded_obj_rot_mat  # (T, 3, 3)
                data_input_dict['obj_scale'] = paded_obj_scale  # (T,)
                data_input_dict['obj_trans'] = paded_obj_trans  # (T, 3)
                data_input_dict['obj_verts'] = paded_obj_trans  # (T, 3)
                data_input_dict['person1_betas'] = self.window_data_dict[index][person]['betas']  # (1, 10)
                data_input_dict['person1_gender'] = str(self.window_data_dict[index][person]['gender'])  # "neutral"
                data_input_dict['person1_trans2joint'] = trans2joint  # 这个beta在SMPLX TPOSE下的pelvis坐标的负值, (3,)
            else:  # person2
                data_input_dict['motion'] = torch.cat((data_input_dict['motion'], paded_new_data_input), dim=-1)
                data_input_dict['ori_motion'] = torch.cat((data_input_dict['ori_motion'], paded_ori_data_input), dim=-1)
                data_input_dict['person2_betas'] = self.window_data_dict[index][person]['betas']  # (1, 10)
                data_input_dict['person2_gender'] = str(self.window_data_dict[index][person]['gender'])  # "neutral"
                data_input_dict['person2_trans2joint'] = trans2joint  # 这个beta在SMPLX TPOSE下的pelvis坐标的负值, (3,)
        
        assert data_input_dict['motion'].shape == (self.window, 24*3+22*6+24*3+22*6)  # 先是person1的24*3+22*6, 再是person2的
        assert data_input_dict['ori_motion'].shape == (self.window, 24*3+22*6+24*3+22*6)  # 先是person1的24*3+22*6, 再是person2的
        assert data_input_dict['obj_com_pos'].shape == (self.window, 3)
        assert data_input_dict['obj_rot_mat'].shape == (self.window, 3, 3)
        assert data_input_dict['obj_scale'].shape == (self.window,)
        assert data_input_dict['obj_trans'].shape == (self.window, 3)
        
        # # data check
        # frame_idx = 0
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(data_input_dict['ori_motion'][frame_idx, 0:24*3].reshape(24, 3))
        # o3d.io.write_point_cloud("person1_ori_motion.ply", pcd)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(data_input_dict['ori_motion'][frame_idx, 24*3+22*6:24*3+22*6+24*3].reshape(24, 3))
        # o3d.io.write_point_cloud("person2_ori_motion.ply", pcd)
        # print("[__getitem__]", "person1", data_input_dict['ori_motion'][frame_idx, 0:72].reshape(24, 3))
        # print("[__getitem__]", "person2", data_input_dict['ori_motion'][frame_idx, 72+132:72+132+72].reshape(24, 3))
        # print("[__getitem__]", "object com_pos, trans =", paded_obj_com_pos[frame_idx], paded_obj_trans[frame_idx])

        return data_input_dict


if __name__ == "__main__":
    dataset = HHODataset(train=True)
    N = len(dataset)
    for i in range(N):
        x = dataset[i]
        print(x.keys())
        break
    
    dataset = HHODataset(train=False)
    N = len(dataset)
    for i in range(N):
        x = dataset[i]
        print(x.keys())
        break
