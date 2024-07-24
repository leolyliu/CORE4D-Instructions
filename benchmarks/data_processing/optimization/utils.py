import os
from os.path import join
import sys
sys.path.append("..")
import torch
from torch import nn
from smplx.smplx.lbs import batch_rodrigues


def local_pose_to_global_orientation(body_pose, global_orient):
    """
    body_pose: torch.float32, shape = (B, 21, 3)
    global_orient: torch.float32, shape = (B, 3)
    
    return:
    global_orientation: a list, len = 22, item = torch.float32 with shape (B, 3, 3)
    """
    parent = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]  # SMPLX body joint parent
    
    B = body_pose.shape[0]  # batch size
    pose = torch.cat((global_orient.unsqueeze(1), body_pose), dim=1)  # (B, 22, 3)
    pose_R = batch_rodrigues(pose.reshape(B * 22, 3)).reshape(B, 22, 3, 3)  # (B, 22, 3, 3)

    global_orientations = [pose_R[:, 0]]
    for idx in range(1, 22):
        global_orientations.append(torch.einsum('bij,bjk->bik', global_orientations[parent[idx]], pose_R[:, idx]))
    return global_orientations


def unit_test():
    class TestModel(nn.Module):
        def __init__(self, B, init_value):
            super(TestModel, self).__init__()
            self.a = nn.Parameter(init_value[:, 1:], requires_grad=True)
            self.b = nn.Parameter(init_value[:, 0], requires_grad=True)

        def forward(self):
            return self.a, self.b
    
    B = 50
    gt_value = torch.randn((B, 22, 3))
    
    model = TestModel(B, gt_value+torch.randn(B,22,3)*0.1)
    model.to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    gt_value = gt_value.to("cuda:0")
    gt_global_orientations = local_pose_to_global_orientation(gt_value[:, 1:], gt_value[:, 0])
    
    for epoch in range(1000):
        a, b = model()
        global_orientations = local_pose_to_global_orientation(a, b)  # len = 22, item = (B, 3, 3)
        loss = 0
        for idx in range(22):
            loss += torch.mean(torch.abs(global_orientations[idx] - gt_global_orientations[idx]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, loss)
    
    a, b = model()
    print(a - gt_value[:, 1:])
    print(b - gt_value[:, 0])


if __name__ == "__main__":
    unit_test()
