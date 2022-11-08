from points import Polygon
import torch

b = torch.tensor([[[0, 1, 2, 3],
                   [0, 0, 0, 0]],
                   [[0, 1, 2, 8],
                    [0, 0, 0, 0]]])
print(b.shape) # (2,2,4)
# output: (2,1)


polygon1 = Polygon(b) 
polygon2 = Polygon(b) 
print(polygon1.get_perimeter())
print(polygon1.get_perimeter().shape)







# #################################################################
# # from points import Point
# from points import Polygon
# import torch


# b = torch.tensor([[[0, 1, 2, 3],
#                    [0, 0, 0, 0]],
#                    [[0, 1, 2, 3],
#                     [0, 0, 0, 0]]])
# print(b.shape) # (2,2,4)
# # output: (2,4)



# print("import OK")
# # pt1 = Point(0,0)
# # pt2 = Point(1,0)
# # pt3 = Point(3,0)
# # pt4 = Point(5,0)
# # pt5 = Point(6,0)
# polygon1 = Polygon(b) 
# print(polygon1.get_perimeter())