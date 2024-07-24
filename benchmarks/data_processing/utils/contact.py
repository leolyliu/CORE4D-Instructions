import os
import sys
sys.path.append("..")
import time
import torch
from torch import nn
from knn_cuda import KNN


def compute_contact(source_vertices, target_vertices, threshould=0.05, device="cuda:0"):
    knn = KNN(k=1, transpose_mode=True)
    source_vertices = torch.from_numpy(source_vertices)[None, ...].to(device)
    target_vertices = torch.from_numpy(target_vertices)[None, ...].to(device)
    
    # source_to_target contact
    dist, _ = knn(target_vertices, source_vertices)
    dist = dist.squeeze(dim=0).squeeze(dim=-1)
    contact = dist < threshould
    
    return contact


def compute_contact_and_closest_point(source_vertices, target_vertices, threshould=0.05):
    """
    source_vertices: torch.tensor, shape = (N, 3)
    target_vertices: torch.tensor, shape = (M, 3)
    
    return:
    contact: torch.tensor, shape = (N), bool
    dist: torch.tensor, shape = (N)
    closest_point: torch.tensor, shape = (N)
    """
    # knn = nn.DataParallel(KNN(k=1, transpose_mode=True), device_ids=[1, 2])
    knn = KNN(k=1, transpose_mode=True)
    # source_to_target contact
    # print(target_vertices.type, source_vertices.type())
    dist, closest_point = knn(target_vertices.unsqueeze(0), source_vertices.unsqueeze(0))
    # print(source_vertices.shape, target_vertices.shape)
    # dist, closest_point = knn(target_vertices, source_vertices)
    dist = dist.squeeze(dim=0).squeeze(dim=-1)
    contact = dist < threshould
    
    closest_point = closest_point[0, :, 0]
    
    return contact, dist, closest_point


class UnitTestModel(nn.Module):
    def __init__(self):
        super(UnitTestModel, self).__init__()
        self.x = nn.Parameter(torch.randn((100, 3)).to("cuda:0"), requires_grad=True)
    
    def forward(self):
        return self.x
    
    def test_computation_time(self):
        start_time = time.time()
        cnt = 300
        for _ in range(cnt):
            a = torch.randn((10000, 3)).to("cuda:0")
            b = torch.randn((10000, 3)).to("cuda:0")
            st = time.time()
            contact, dist, closest_point = compute_contact_and_closest_point(a, b, threshould=0.05)
            print("time =", time.time() - st)
        print("avg time =", (time.time() - start_time) / cnt)


if __name__ == "__main__":
    
    model = UnitTestModel()
    model.test_computation_time()
    exit(0)
    target = nn.Parameter(torch.randn((10, 3)).to("cuda:0"), requires_grad=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(1000):
        source = model()
        contact, dist, closest_point = compute_contact_and_closest_point(source, target, threshould=1.0)
        loss = torch.mean(torch.abs((source - target[closest_point]) * contact.unsqueeze(1).expand(-1, 3)))
        if epoch % 100 == 0:
            print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    source = model()
    contact, dist, closest_point = compute_contact_and_closest_point(source, target, threshould=1.0)

    print(source)
    print(target)
    print(dist)
