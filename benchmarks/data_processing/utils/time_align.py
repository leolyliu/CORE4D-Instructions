import os
from os.path import join, isdir, isfile
import sys
sys.path.append("..")
import numpy as np
from transforms3d.quaternions import quat2mat
from utils.process_timestamps import txt2timestamps, paired_frameids_to_txt, txt_to_paried_frameids
from utils.VTS_object import get_obj_info
from utils.pose import lerp
from utils.denoising import denoise_pose


def time_align(data_dir, cfg, threshould=40000000, VTS_add_time=None):
    """
    VTS_add_time = None: 强行匹配VTS和rgb1的首帧, 以此计算VTS_add_time并返回
    """
    
    if not cfg["camera1"]:
        raise NotImplementedError
    
    # camera data
    rgb1_timestamps = txt2timestamps(join(data_dir, "_d455_camera1_color_image_raw_timestamp.txt"))
    N_rgb1 = len(rgb1_timestamps)
    if cfg["camera2"]:
        rgb2_timestamps = txt2timestamps(join(data_dir, "_d455_camera2_color_image_raw_timestamp.txt"))
        N_rgb2 = len(rgb2_timestamps)
    if cfg["camera3"]:
        rgb3_timestamps = txt2timestamps(join(data_dir, "_d455_camera3_color_image_raw_timestamp.txt"))
        N_rgb3 = len(rgb3_timestamps)
    if cfg["camera4"]:
        rgb4_timestamps = txt2timestamps(join(data_dir, "_d455_camera4_color_image_raw_timestamp.txt"))
        N_rgb4 = len(rgb4_timestamps)
    
    # VTS data
    VTS_data = np.load(join(data_dir, "VTS_data.npz"), allow_pickle=True)["data"].item()
    
    # (optional) compute VTS_add_time
    return_VTS_add_time = False
    if VTS_add_time is None:
        return_VTS_add_time = True
        assert "VTS_rigid_timestamp" in VTS_data
        VTS_add_time = rgb1_timestamps[0] - VTS_data["VTS_rigid_timestamp"][0]
    
    if cfg["person1"]:
        person1_timestamps = [x+VTS_add_time for x in VTS_data["person1_timestamp"]]
        N_person1 = len(person1_timestamps)
    if cfg["person2"]:
        person2_timestamps = [x+VTS_add_time for x in VTS_data["person2_timestamp"]]
        N_person2 = len(person2_timestamps)
    if cfg["object"]:
        rigid_timestamps = [x+VTS_add_time for x in VTS_data["rigid_timestamp"]]
        N_rigid = len(rigid_timestamps)
    
    paired_frameids = []
    p_rgb2, p_rgb3, p_rgb4, p_person1, p_person2, p_rigid = 0, 0, 0, 0, 0, 0

    for rgb1_idx in range(N_rgb1):
        t = rgb1_timestamps[rgb1_idx]
        # rgb2 align with rgb1
        if cfg["camera2"]:
            while (p_rgb2 + 1 < N_rgb2) and (abs(t - rgb2_timestamps[p_rgb2 + 1]) <= abs(t - rgb2_timestamps[p_rgb2])):
                p_rgb2 += 1
        # rgb3 align with rgb1
        if cfg["camera3"]:
            while (p_rgb3 + 1 < N_rgb3) and (abs(t - rgb3_timestamps[p_rgb3 + 1]) <= abs(t - rgb3_timestamps[p_rgb3])):
                p_rgb3 += 1
        # rgb4 align with rgb1
        if cfg["camera4"]:
            while (p_rgb4 + 1 < N_rgb4) and (abs(t - rgb4_timestamps[p_rgb4 + 1]) <= abs(t - rgb4_timestamps[p_rgb4])):
                p_rgb4 += 1
        # person1_pose align with rgb1
        if cfg["person1"]:
            while (p_person1 + 1 < N_person1) and (abs(t - person1_timestamps[p_person1 + 1]) <= abs(t - person1_timestamps[p_person1])):
                p_person1 += 1
        # person2_pose align with rgb1
        if cfg["person2"]:
            while (p_person2 + 1 < N_person2) and (abs(t - person2_timestamps[p_person2 + 1]) <= abs(t - person2_timestamps[p_person2])):
                p_person2 += 1
        # obj_pose align with rgb1
        if cfg["object"]:
            while (p_rigid + 1 < N_rigid) and (abs(t - rigid_timestamps[p_rigid + 1]) <= abs(t - rigid_timestamps[p_rigid])):
                p_rigid += 1
        
        flag = True
        if cfg["camera2"]:
            flag &= abs(t - rgb2_timestamps[p_rgb2]) < threshould
        if cfg["camera3"]:
            flag &= abs(t - rgb3_timestamps[p_rgb3]) < threshould
        if cfg["camera4"]:
            flag &= abs(t - rgb4_timestamps[p_rgb4]) < threshould
        if cfg["person1"]:
            flag &= abs(t - person1_timestamps[p_person1]) < threshould
        if cfg["person2"]:
            flag &= abs(t - person2_timestamps[p_person2]) < threshould
        if cfg["object"]:
            flag &= abs(t - rigid_timestamps[p_rigid]) < threshould

        if not flag:
            print("[error in preparing paired data] wrong frame idx =", rgb1_idx)
            continue
        
        paired_frameids.append([rgb1_idx, p_rgb2, p_rgb3, p_rgb4, p_person1, p_person2, p_rigid])
    
    paired_frameids_to_txt(paired_frameids, join(data_dir, "aligned_frame_ids.txt"))
    
    if return_VTS_add_time:
        return VTS_add_time


def prepare_objpose(data_dir, obj_name, threshould=40000000, VTS_add_time=0):
    """
    根据time_align(cfg["object"]=False)得到的结果, 计算每个paired_frame对应的object pose, 包含object pose的插值
    """
    assert isfile(join(data_dir, "aligned_frame_ids.txt"))
    paired_frameids = txt_to_paried_frameids(join(data_dir, "aligned_frame_ids.txt"))
    
    assert isfile(join(data_dir, "_d455_camera1_color_image_raw_timestamp.txt"))  # 时间戳的匹配均以camera1为基准, 要求这个相机的数据必须存在
    rgb1_timestamps = txt2timestamps(join(data_dir, "_d455_camera1_color_image_raw_timestamp.txt"))
    N_rgb1 = len(rgb1_timestamps)
    
    VTS_data = np.load(join(data_dir, "VTS_data.npz"), allow_pickle=True)["data"].item()
    if not "/labels" in VTS_data:
        print("[prepare_objpose] warning: no objpose data!!!")
        return
    rigid_timestamps_all = [x+VTS_add_time for x in VTS_data["rigid_timestamp"]]
    rigid_poses_all = VTS_data["/rigid"]
    labels = VTS_data["/labels"]
    assert (len(rigid_poses_all) == len(labels)) and (len(rigid_poses_all) == len(rigid_timestamps_all))
    
    # 获取和obj_name相关的信息
    rigid_timestamps = []
    rigid_poses = []
    for (poses, device_names, ts) in zip(rigid_poses_all, labels, rigid_timestamps_all):
        for (pose, device_name) in zip(poses, device_names):
            if device_name == obj_name:
                rigid_timestamps.append(ts)
                T = np.eye(4)
                T[:3, 3] = pose["position"]
                T[:3, :3] = quat2mat(pose["orientation"])
                rigid_poses.append(T)
                continue
    
    rigid_poses, rigid_timestamps = denoise_pose(rigid_poses, rigid_timestamps)
    
    N_rigid = len(rigid_timestamps)
    assert N_rigid > 0
    
    objposes = []
    
    p_rigid = 0  # 指向时间戳>=t的编号最小的帧
    for pair_idx in range(len(paired_frameids)):
        rgb1_idx = paired_frameids[pair_idx][0]
        t = rgb1_timestamps[rgb1_idx]
        
        while (p_rigid < N_rigid) and (rigid_timestamps[p_rigid] < t):
            p_rigid += 1
        
        # interpolate object pose
        if (p_rigid > 0) and (t - rigid_timestamps[p_rigid - 1] < threshould):
            print("rgb1 align to previous objpose,", rgb1_idx)
            objposes.append(rigid_poses[p_rigid - 1])
            continue
        if (p_rigid < N_rigid) and (rigid_timestamps[p_rigid] - t < threshould):
            print("rgb1 align to subsequent objpose,", rgb1_idx)
            objposes.append(rigid_poses[p_rigid])
            continue
        if p_rigid == 0:
            objposes.append(rigid_poses[0])
            continue
        if p_rigid == N_rigid:
            objposes.append(rigid_poses[-1])
            continue
        print("rgb1 frame {} need objpose interpolation".format(str(rgb1_idx)))
        pose_0 = rigid_poses[p_rigid - 1]
        t_0 = rigid_timestamps[p_rigid - 1]
        pose_1 = rigid_poses[p_rigid]
        t_1 = rigid_timestamps[p_rigid]
        objposes.append(lerp(pose_0, pose_1, alpha=(t_1-t)/(t_1-t_0)))
    
    objposes = np.float32(objposes)
    print("objposes.shape =", objposes.shape)
    # for idx in range(objposes.shape[0]):
    #     print(idx, objposes[idx])
    np.save(join(data_dir, "aligned_objposes.npy"), objposes)


if __name__ == "__main__":
    
    #########################################################################
    obj_dataset_dir = "/share/datasets/HHO_object_dataset_final"
    root_dir = "/share/datasets/HHO_dataset/data/20230806_1"
    #########################################################################
    
    seq_names = []
    for sequence_name in os.listdir(root_dir):
        sequence_dir = join(root_dir, sequence_name)
        if not isdir(sequence_dir):
            continue
        seq_names.append(sequence_name)
    
    seq_names.sort()

    for seq_name in seq_names:
        seq_dir = join(root_dir, seq_name)
        if seq_name != "001":
            continue
        print("processing {} ...".format(seq_dir))
        cfg = {
            "camera1": True,
            "camera2": True,
            "camera3": True,
            "camera4": True,
            "person1": True,
            "person2": True,
            "object": False,
        }
        VTS_add_time = time_align(seq_dir, cfg, threshould=40000000, VTS_add_time=None)
        
        obj_name, obj_model_path = get_obj_info(seq_dir, obj_dataset_dir)
        print(VTS_add_time)
        prepare_objpose(seq_dir, obj_name, threshould=40000000, VTS_add_time=VTS_add_time)
