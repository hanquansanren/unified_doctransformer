import torch 
import numpy as np
a = torch.randn((2,2))
b = torch.randn((3,2))

a = torch.from_numpy(np.pad(a.numpy(), (0,4), 'constant', constant_values=-100)[:,0:2])
b = torch.from_numpy(np.pad(b.numpy(), (0, 3), 'constant', constant_values=-100)[:,0:2])


c = torch.hstack((a,b))
print(c, c.size())







