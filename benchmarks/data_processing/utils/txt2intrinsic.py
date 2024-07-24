import numpy as np


def txt2intrinsic(txt_path):
    intrin = np.eye(3)
    dist = np.zeros(5)
    cnt = -1
    with open(txt_path, "r") as f:
        for line in f:
            cnt += 1
            line = line.strip().split(",")
            values = np.float32([float(v) for v in line])
            if cnt <= 2:
                intrin[cnt] = values
            else:
                dist = values
    return intrin, dist
