import numpy as np


a=np.zeros((4,3), dtype=np.uint32)
b=np.argwhere(np.zeros((4,3), dtype=np.uint32) == 0)
print(a)
print(b)