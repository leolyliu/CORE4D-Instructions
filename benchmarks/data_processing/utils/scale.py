import os
import sys
sys.path.append("..")
import numpy as np
from utils.bvh2joint import bvh2joint, get_joint_info


def get_constraints_for_optimizing_shape(human_scale, from_bvh=False):
    # mini
    constraints = [
        [7, 12],
        [8, 12],
        [19, 17],
        [18, 16],
        [21, 19],
        [20, 18],
        [5, 2],
        [4, 1],
        [8, 5],
        [7, 4],
    ]
    s = human_scale.copy()
    human_scale = [s[0], s[0], s[1], s[1], s[2], s[2], s[3], s[3], s[4], s[4]]
    return constraints, human_scale

    if from_bvh == False:
        # 基于在T-POSE下测量的尺寸
        constraints = [
            [21, 19],
            [20, 18],
            [19, 12],
            [18, 12],
            [12, 0],
            [5, 0],
            [4, 0],
            [8, 5],
            [7, 4],
            [11, 8],
            [10, 7],
            [54, 53],
            [39, 38],
            [71, 54],
            [66, 39],
            [42, 41],
            [27, 26],
            [72, 42],
            [67, 27],
            [45, 44],
            [30, 29],
            [73, 45],
            [68, 30],
            [51, 50],
            [36, 35],
            [74, 51],
            [69, 36],
            [48, 47],
            [33, 32],
            [75, 48],
            [70, 33],
        ]
        s = human_scale.copy()
        human_scale = [s[0], s[0], s[1], s[1], s[2], s[3], s[3], s[4], s[4], s[5], s[5], s[6], s[6], s[7], s[7], s[8], s[8], s[9], s[9], s[10], s[10], s[11], s[11], s[12], s[12], s[13], s[13], s[14], s[14], s[15], s[15]]
    else:
        # 身体中轴线上没有任何约束!
        constraints = [
            [21, 19],
            [20, 18],
            [19, 17],
            [18, 16],
            [5, 2],
            [4, 1],
            [8, 5],
            [7, 4],
            [54, 53],
            [39, 38],
            [41, 40],
            [26, 25],
            [42, 41],
            [27, 26],
            [44, 43],
            [29, 28],
            [45, 44],
            [30, 29],
            [50, 49],
            [35, 34],
            [51, 50],
            [36, 35],
            [47, 46],
            [32, 31],
            [48, 47],
            [33, 32],
        ]
        s = human_scale.copy()
        human_scale = [s[15], s[38], s[14], s[37], s[1], s[4], s[2], s[5], s[18], s[41], s[21], s[44], s[22], s[45], s[25], s[48], s[26], s[49], s[29], s[52], s[30], s[53], s[33], s[56], s[34], s[57]]
    
    return constraints, human_scale


def get_mean_scale_info(start, end):
    """
    start: N帧里父节点的3D坐标, shape = (N, 3)
    end: N帧里子节点的3D坐标, shape = (N, 3)
    """
    d = np.linalg.norm(start - end, axis=-1, ord=2)
    return np.float32([d.mean(), d.std()])  # shape = (2,)


def get_scale_from_data(bvh_path, frame_ids=None):
    joint_data = bvh2joint(bvh_path, frame_ids=frame_ids, end_link_trans=None)

    assert joint_data.shape[1:] == (59, 3)

    joint_info = get_joint_info()
    joint_ids = {}
    for i in range(59):
        joint_ids[joint_info[i][0]] = i

    # link的顺序定义与bvh文件相同
    link_scale_info = []
    for i in range(1, 59):
        child_j, father_j = joint_info[i][0], joint_info[i][1]
        child_idx, father_idx = joint_ids[child_j], joint_ids[father_j]
        link_scale_info.append(get_mean_scale_info(joint_data[:, father_idx, :], joint_data[:, child_idx, :]))
    
    link_scale_info = np.float32(link_scale_info)
    assert link_scale_info.shape == (58, 2)
    scale_mean, scale_std = link_scale_info[:, 0], link_scale_info[:, 1]
    return scale_mean, scale_std


if __name__ == "__main__":
    bvh_path = "../exp_data/before_20230321/SIK_Actor_02_20230311_twoperson.bvh"
    scale_mean, scale_std = get_scale_from_data(bvh_path, frame_ids=None)
    # 结论: 大臂(ForeArm)的std较大(约2cm), 大腿(Leg)和小腿(Foot)的std较大(约0.6cm), 其余link的std接近0
