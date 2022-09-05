import numpy as np
import torch

# # np.argwhere的用法
# a=np.zeros((4,3), dtype=np.uint32)
# b=np.argwhere(np.zeros((4,3), dtype=np.uint32) == 0)
# print(a)
# print(b)
# print(type(b),b.shape)

# # reshape的用法，torch和numpy都有类似的用法
# a = torch.arange(4.)
# print(a.shape)
# a=torch.reshape(a, (2, 2))
# print(a.shape)
# b = torch.rand(4,4,2)
# print(b.shape)
# b=torch.reshape(b, (-1,2))
# print(b.shape)

# # np.arange和np.linspace的一点比较
# print(np.arange(-6, 6, 2))
# I_r_grid_x = (np.arange(-6, 6, 2) + 1.0) / 6 
# I_r_grid_y = (np.arange(-6, 6, 2) + 1.0) / 6
# print(I_r_grid_x)
# print(I_r_grid_y)

# I_r_grid_x=np.linspace(-1,1,6)
# print(I_r_grid_x)


# # np.concatenate用法
# a = np.concatenate([np.zeros((1, 3)), np.ones((1, 9))], axis=1)
# print(a)


# # np.fill_diagonal用法
# hat_C=5*np.ones((4,4))
# np.fill_diagonal(hat_C, 1)
# print(hat_C)


# 参考点生成
xs = torch.linspace(0, 1020, steps=61)
ys = torch.linspace(0, 1020, steps=61)
x, y = torch.meshgrid(xs, ys, indexing='xy')
P = torch.dstack([x, y])
print(P)