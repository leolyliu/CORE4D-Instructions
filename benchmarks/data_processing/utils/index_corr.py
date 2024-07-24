import os
import sys
sys.path.append("..")
import numpy as np

#  BVH ['Hips', 0
# 'RightUpLeg', 'RightLeg', 'RightFoot', 3
# 4 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 7
# 7 'Spine', 'Spine1', 'Spine2', 
# 10 'Neck', 'Neck1', 'Head', 
# 13 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 
# 36 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3']

def smplx2bvh_for_optimizing_smplx(N, split_hand=False):
    constraints = [
        [0, 7],  # SMPL pelvis = VTS spine
        [2, 1],  # TODO: 不是同一个joint!
        [5, 2],
        [8, 3],
        [1, 4],  # TODO: 不是同一个joint!
        [4, 5],
        [7, 6],
        [9, 9],
        [12, 11],
        [17, 14],
        [19, 15],
        [21, 16],
        [16, 37],
        [18, 38],
        [20, 39],
        [53, 18],
        [54, 19],
        [40, 21],
        [41, 22],
        [42, 23],
        [43, 25],
        [44, 26],
        [45, 27],
        [49, 29],
        [50, 30],
        [51, 31],
        [46, 33],
        [47, 34],
        [48, 35],
        [38, 41],
        [39, 42],
        [25, 44],
        [26, 45],
        [27, 46],
        [28, 48],
        [29, 49],
        [30, 50],
        [34, 52],
        [35, 53],
        [36, 54],
        [31, 56],
        [32, 57],
        [33, 58],
    ]
    comp_constraints = [  # 组合约束: SMPLX两个关节的中点和VTS的某个关节重合
        [[(0.5, 3), (0.5, 6)], 8],
    ]
    direction_constraints = [  # (A to B) = (C to D)
        [(15, 12), (59, 11)],
    ]

    if N == 59:
        pass
    elif N == 74:  # 增加15个末端的坐标
        constraints += [
            [15, 59],
            [63, 60],
            [64, 61],
            [60, 62],
            [61, 63],
            [71, 64],
            [72, 65],
            [73, 66],
            [74, 67],
            [75, 68],
            [66, 69],
            [67, 70],
            [68, 71],
            [69, 72],
            [70, 73],
        ]
    else:
        print("Wrong bvh joint number:", N)
        raise NotImplementedError
    
    if split_hand:
        constraints_nohand = []
        constraints_hand = []
        for c in constraints:
            if ((25 <= c[0]) and (c[0] <= 54)) or ((66 <= c[0]) and (c[0] <= 75)):
                constraints_hand.append(c)
            else:
                constraints_nohand.append(c)
        return constraints_nohand, constraints_hand, comp_constraints, direction_constraints
    else:
        return constraints, comp_constraints, direction_constraints
