import cv2
import numpy as np
import random
import torch
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
DEVICE = torch.device("cpu")

grid = torch.ones(1, 3, 3, 2, device=DEVICE)
print(torch.linspace(-1, 1, 3))
print(torch.linspace(-1, 1, 3)[..., None])

print(grid,grid.size())
grid[:, :, :, 0] = torch.linspace(-1, 1, 3)
grid[:, :, :, 1] = torch.linspace(-1, 1, 3)[..., None] # [..., None]相当于增加了一个维度，变成二维向量
print(grid,grid.size())
grid = grid.view(-1, 3 * 3, 2)
print(grid,grid.size())


eps = 1e-9
D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
U = D2 * torch.log(D2 + eps)

print(U)