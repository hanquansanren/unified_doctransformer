import numpy as np

P_tile = np.array([
    [[0,1],
     [0,1],
     [0,1],],
    [[1,1],
     [1,1],
     [1,1],],
    [[2,1],
     [2,1],
     [2,1],],
    [[3,1],
     [3,1],
     [3,1],]
])
print(P_tile.shape)
C_tile=np.array([
    [[0,1],
     [1,1],
     [2,1]]
])
print(C_tile.shape)
P_diff = P_tile - C_tile  # nxFx2 #(4, 3, 2)-(1,3,2)=(4, 3, 2)
print(P_diff)
print(P_diff.shape)

