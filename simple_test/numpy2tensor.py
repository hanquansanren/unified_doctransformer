import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

a=np.array([[[1,2,3],
            [1,2,3],
            [1,2,3]],
            [[2,2,2],
            [2,2,2],
            [2,2,2]],
            [[3,2,1],
            [3,2,1],
            [3,2,1]]])

print('the shape of array a is {}'.format(a.shape))
print(f'the shape of array a is {a.shape}')
print('the shape of array a is %d',a.shape)
# torch.tensor()