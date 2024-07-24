import numpy as np
import torch
from transforms3d.axangles import mat2axangle


def np_mat2axangle(Ms):
    """
    Ms: (N, 3, 3), np.float32
    return: (N, 3), np.float32
    """
    axangles = []
    for M in Ms:
        d, a = mat2axangle(M)
        axangles.append((d * a).reshape(3))
    axangles = np.float32(axangles)
    return axangles

def torch_mat2axangle(Ms):
    """
    Ms: (N, 3, 3), torch.float32
    return: (N, 3), torch.float32
    """
    axangles = []
    for M in Ms:
        d, a = mat2axangle(M.detach().cpu().numpy())
        axangles.append((d * a).reshape(3))
    axangles = torch.tensor(axangles, dtype=torch.float32, device=Ms.device)
    return axangles