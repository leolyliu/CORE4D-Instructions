import os
import numpy as np
import pandas as pd
import open3d as o3d
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat
# from transforms3d.axangles import *


def default_end_link_trans():  # liuyun's
    end_link_trans = np.float32([
        [0, 8.5, 0],
        [4, -8, 20],
        [-4, -8, 17],
        [-4, -8, 20],
        [4, -8, 17],
        [-2.5, 0, 0],
        [-2.5, 0, 0],
        [-2.5, 0, 0],
        [-2.5, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [2.5, 0, 0],
        [2.5, 0, 0],
        [2.5, 0, 0],
        [2.5, 0, 0],
    ])
    return end_link_trans


def get_joint_info():
    joint_info = [
        ["Hips", None],
        ["RightUpLeg", "Hips"],
        ["RightLeg", "RightUpLeg"],
        ["RightFoot", "RightLeg"],
        ["LeftUpLeg", "Hips"],
        ["LeftLeg", "LeftUpLeg"],
        ["LeftFoot", "LeftLeg"],
        ["Spine", "Hips"],
        ["Spine1", "Spine"],
        ["Spine2", "Spine1"],
        ["Neck", "Spine2"],
        ["Neck1", "Neck"],
        ["Head", "Neck1"],
        ["RightShoulder", "Spine2"],
        ["RightArm", "RightShoulder"],
        ["RightForeArm", "RightArm"],
        ["RightHand", "RightForeArm"],
        ["RightHandThumb1", "RightHand"],
        ["RightHandThumb2", "RightHandThumb1"],
        ["RightHandThumb3", "RightHandThumb2"],
        ["RightHandIndex", "RightHand"],
        ["RightHandIndex1", "RightHandIndex"],
        ["RightHandIndex2", "RightHandIndex1"],
        ["RightHandIndex3", "RightHandIndex2"],
        ["RightHandMiddle", "RightHand"],
        ["RightHandMiddle1", "RightHandMiddle"],
        ["RightHandMiddle2", "RightHandMiddle1"],
        ["RightHandMiddle3", "RightHandMiddle2"],
        ["RightHandRing", "RightHand"],
        ["RightHandRing1", "RightHandRing"],
        ["RightHandRing2", "RightHandRing1"],
        ["RightHandRing3", "RightHandRing2"],
        ["RightHandPinky", "RightHand"],
        ["RightHandPinky1", "RightHandPinky"],
        ["RightHandPinky2", "RightHandPinky1"],
        ["RightHandPinky3", "RightHandPinky2"],
        ["LeftShoulder", "Spine2"],
        ["LeftArm", "LeftShoulder"],
        ["LeftForeArm", "LeftArm"],
        ["LeftHand", "LeftForeArm"],
        ["LeftHandThumb1", "LeftHand"],
        ["LeftHandThumb2", "LeftHandThumb1"],
        ["LeftHandThumb3", "LeftHandThumb2"],
        ["LeftHandIndex", "LeftHand"],
        ["LeftHandIndex1", "LeftHandIndex"],
        ["LeftHandIndex2", "LeftHandIndex1"],
        ["LeftHandIndex3", "LeftHandIndex2"],
        ["LeftHandMiddle", "LeftHand"],
        ["LeftHandMiddle1", "LeftHandMiddle"],
        ["LeftHandMiddle2", "LeftHandMiddle1"],
        ["LeftHandMiddle3", "LeftHandMiddle2"],
        ["LeftHandRing", "LeftHand"],
        ["LeftHandRing1", "LeftHandRing"],
        ["LeftHandRing2", "LeftHandRing1"],
        ["LeftHandRing3", "LeftHandRing2"],
        ["LeftHandPinky", "LeftHand"],
        ["LeftHandPinky1", "LeftHandPinky"],
        ["LeftHandPinky2", "LeftHandPinky1"],
        ["LeftHandPinky3", "LeftHandPinky2"],
    ]
    return joint_info


def load_joint_csv(joint_csv_path):
    video_data = pd.read_csv(joint_csv_path)
    N_frame = video_data.shape[0]
    assert video_data.shape[1] == 217  # 1 (time) + 72 * 3 (3D joints in the world coordinate system)
    # 13个end-joint的3D坐标和其父节点的3D坐标相等!!!
    joints_list = []
    for i in range(N_frame):
        frame_data = np.float32(video_data.iloc[i])
        # print(video_data.iloc[i][["LeftHandPinky3.X","LeftHandPinky3.Y","LeftHandPinky3.Z","LeftHandPinky3End.X","LeftHandPinky3End.Y","LeftHandPinky3End.Z"]])
        # print(video_data.iloc[i][["Hips.X", "Hips.Y", "Hips.Z", "RightUpLeg.X", "RightUpLeg.Y", "RightUpLeg.Z"]])
        joints_list.append({
            "time": frame_data[0],
            "joints": frame_data[1:].reshape(72, 3),
        })

    return joints_list


def bvh2joint_fake(bvh_path):
    """
    使用的是默认机器人的骨节长度, 不是真人的！
    """
    assert bvh_path[-4:] == ".bvh"
    os.system("bvh-converter {}".format(bvh_path))

    joint_csv_path = bvh_path[:-4] + "_worldpos.csv"
    joints_list = load_joint_csv(joint_csv_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(joints_list[0]["joints"])
    o3d.io.write_point_cloud("./ex.ply", pcd)


def local2world(pos, local2world):
    """
    pos: shape = (3,)
    把局部link坐标系的3D坐标转化到全局坐标系
    """
    pos = np.float32(pos)
    assert pos.shape == (3,)
    pos = np.hstack((pos, [1.0])).reshape(4, 1)
    pos = local2world @ pos
    pos = pos[:3].reshape(3)
    return pos


def add_end_link(joint_3Dpos, joint_acc_pose, end_link_trans):
    assert len(joint_3Dpos) == 59

    joint_3Dpos.append(local2world(end_link_trans[0], joint_acc_pose["Head"]))
    joint_3Dpos.append(local2world(end_link_trans[1], joint_acc_pose["RightFoot"]))
    joint_3Dpos.append(local2world(end_link_trans[2], joint_acc_pose["RightFoot"]))
    joint_3Dpos.append(local2world(end_link_trans[3], joint_acc_pose["LeftFoot"]))
    joint_3Dpos.append(local2world(end_link_trans[4], joint_acc_pose["LeftFoot"]))
    joint_3Dpos.append(local2world(end_link_trans[5], joint_acc_pose["RightHandThumb3"]))
    joint_3Dpos.append(local2world(end_link_trans[6], joint_acc_pose["RightHandIndex3"]))
    joint_3Dpos.append(local2world(end_link_trans[7], joint_acc_pose["RightHandMiddle3"]))
    joint_3Dpos.append(local2world(end_link_trans[8], joint_acc_pose["RightHandRing3"]))
    joint_3Dpos.append(local2world(end_link_trans[9], joint_acc_pose["RightHandPinky3"]))
    joint_3Dpos.append(local2world(end_link_trans[10], joint_acc_pose["LeftHandThumb3"]))
    joint_3Dpos.append(local2world(end_link_trans[11], joint_acc_pose["LeftHandIndex3"]))
    joint_3Dpos.append(local2world(end_link_trans[12], joint_acc_pose["LeftHandMiddle3"]))
    joint_3Dpos.append(local2world(end_link_trans[13], joint_acc_pose["LeftHandRing3"]))
    joint_3Dpos.append(local2world(end_link_trans[14], joint_acc_pose["LeftHandPinky3"]))

    assert len(joint_3Dpos) == 74
    return joint_3Dpos


def bvh2joint(bvh_path, frame_ids=None, end_link_trans=None, return_local_rot=False, selected_frames=None, sampling_rate=1):
    """
    bvh_path: 单人bvh文件路径
    end_link_trans: 人体15个叶节点在T-POSE下相对于其父节点的3D偏移, 顺序: [下颚到鼻尖的垂直距离, 右脚(大脚趾,小脚趾)尖到脚腕, 左脚(大脚趾,小脚趾)尖到脚腕, 右手5个指尖长(从thumb到pinky), 左手5个指尖长(从thumb到pinky)], 单位: cm
    头: [0, 下颚到鼻尖的垂直距离, 0]
    左右脚: [(左脚-+,右脚+1)一半脚宽, -脚高, 大/小脚趾尖到脚腕在向前方向上的距离]
    右指尖长: [-指尖长, 0, 0]
    左指尖长: [指尖长, 0, 0]

    return: shape = (N, 59/74, 3)
    """

    if not end_link_trans is None:
        assert end_link_trans.shape == (15, 3)

    joint_info = get_joint_info()
    joint_ids = {}
    for i in range(59):
        joint_ids[joint_info[i][0]] = i

    frame_dict = None
    if not frame_ids is None:
        frame_dict = {}
        for i in list(frame_ids):
            frame_dict[i] = 1

    joint_data = []
    local_rot_data = []

    cnt = -1
    with open(bvh_path, "r") as f:
        for line in f:
            values = line.strip().split(" ")
            if len(values) != 354:
                continue
            cnt += 1
            if (not frame_dict is None) and (not cnt in frame_dict):
                continue
            values = np.float32([float(i) for i in values])

            joint_3Dpos = []
            local_rot = []
            joint_acc_pose = {}
            for i in range(59):
                pos = values[6*i : 6*i+3]
                if not joint_info[i][1] is None:
                    pos = local2world(pos, joint_acc_pose[joint_info[i][1]])
                joint_3Dpos.append(pos)
                pose = np.eye(4)
                pose[:3, :3] = euler2mat(values[6*i+3]/180*np.pi, values[6*i+4]/180*np.pi, values[6*i+5]/180*np.pi, axes="ryxz")
                pose[:3, 3] = values[6*i : 6*i+3]
                local_rot.append(pose)
                if not joint_info[i][1] is None:
                    pose = joint_acc_pose[joint_info[i][1]] @ pose
                joint_acc_pose[joint_info[i][0]] = pose
    
            if not end_link_trans is None:  # 在原先的59个joint的世界系3D坐标后面拼接15个末端的世界系3D坐标，顺序: [头顶, 右脚2个顶点, 左脚2个顶点, 右手5个指尖(从thumb到pinky), 左手5个指尖(从thumb到pinky)]
                joint_3Dpos = add_end_link(joint_3Dpos, joint_acc_pose, end_link_trans)

            joint_data.append(np.float32(joint_3Dpos) / 100)  # 单位: m
            local_rot_data.append(np.float32(local_rot))

    if selected_frames is None:
        joint_data = np.float32(joint_data)[::sampling_rate]
        local_rot_data = np.float32(local_rot_data)[::sampling_rate]
    else:
        selected_frames = np.int32(selected_frames)
        joint_data = np.float32(joint_data)[selected_frames]
        local_rot_data = np.float32(local_rot_data)[selected_frames]
        
    local_rot_data[:, :, :3, 3] /= 100  # unit: m
    
    if return_local_rot:
        return joint_data, local_rot_data
    else:
        return joint_data


def bvh_rawdata_to_joint(bvh_rawdata, frame_ids=None, end_link_trans=None, return_local_rot=False, selected_frames=None, sampling_rate=1):
    if not end_link_trans is None:
        assert end_link_trans.shape == (15, 3)

    joint_info = get_joint_info()
    joint_ids = {}
    for i in range(59):
        joint_ids[joint_info[i][0]] = i

    frame_dict = None
    if not frame_ids is None:
        frame_dict = {}
        for i in list(frame_ids):
            frame_dict[i] = 1

    joint_data = []
    local_rot_data = []
    
    if selected_frames is None:
        selected_data = bvh_rawdata[::sampling_rate]
    else:
        selected_data = [bvh_rawdata[x] for x in selected_frames]

    for frame_data in selected_data:
        joint_3Dpos = []
        local_rot = []
        joint_acc_pose = {}
        for i in range(59):
            pos = frame_data[i]["position"]
            if not joint_info[i][1] is None:
                pos = local2world(pos, joint_acc_pose[joint_info[i][1]])
            joint_3Dpos.append(pos)
            pose = np.eye(4)
            pose[:3, :3] = quat2mat(frame_data[i]["orientation"])
            pose[:3, 3] = frame_data[i]["position"]
            local_rot.append(pose)
            if not joint_info[i][1] is None:
                pose = joint_acc_pose[joint_info[i][1]] @ pose
            joint_acc_pose[joint_info[i][0]] = pose
    
        if not end_link_trans is None:  # 在原先的59个joint的世界系3D坐标后面拼接15个末端的世界系3D坐标，顺序: [头顶, 右脚2个顶点, 左脚2个顶点, 右手5个指尖(从thumb到pinky), 左手5个指尖(从thumb到pinky)]
            joint_3Dpos = add_end_link(joint_3Dpos, joint_acc_pose, end_link_trans)

        joint_data.append(np.float32(joint_3Dpos) / 100)  # 单位: m
        local_rot_data.append(np.float32(local_rot))

    joint_data = np.float32(joint_data)
    local_rot_data = np.float32(local_rot_data)
    local_rot_data[:, :, :3, 3] /= 100  # unit: m
    if return_local_rot:
        return joint_data, local_rot_data
    else:
        return joint_data


def get_joint_data(data_path, person_id=None, frame_ids=None, end_link_trans=None, return_local_rot=False, selected_frames=None, sampling_rate=1):
    assert return_local_rot  # 必须是True, 因为多帧联合优化需要local rotation的信息做初始化和做约束项

    if data_path.split(".")[-1] == "bvh":
        joint_data, bvh_rot = bvh2joint(data_path, frame_ids=frame_ids, end_link_trans=end_link_trans, return_local_rot=return_local_rot, selected_frames=selected_frames, sampling_rate=sampling_rate)
    elif data_path.split(".")[-1] == "npz":
        VTS_data = np.load(data_path, allow_pickle=True)["data"].item()
        topic_name = "/joints" + ("" if person_id == 1 else str(person_id))
        assert topic_name in VTS_data
        joint_data, bvh_rot = bvh_rawdata_to_joint(VTS_data[topic_name], frame_ids=frame_ids, end_link_trans=end_link_trans, return_local_rot=return_local_rot, selected_frames=selected_frames, sampling_rate=sampling_rate)

    return joint_data, bvh_rot


if __name__ == "__main__":
    bvh_path = "/home/liuyun/HHO-dataset/data_processing/exp_data/20230407_data/mocap+rgbd_person1.bvh"
    # bvh2joint(bvh_path, frame_ids=[0])

    end_link_trans = default_end_link_trans()
    bvh2joint(bvh_path, frame_ids=[1029], end_link_trans=end_link_trans)
