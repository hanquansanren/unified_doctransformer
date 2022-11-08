import torch
class Polygon():
    def __init__(self, points):
        self.points = points # (2,2,4)

    def __len__(self):
        return len(self.points)

    def get_perimeter(self):
        perimeter = 0
        # print(len(self.points)-1)
        for i in range(0, self.points.size(2)-1):
            pt1 = self.points[:,:,i]
            pt2 = self.points[:,:,i+1]
            distance = torch.sum((pt1 - pt2)**2 , 1)**0.5
            perimeter += distance

        return perimeter


# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#     def distance(self, second):
#         x_d = self.x - second.x
#         y_d = self.y - second.y
#         return (x_d**2 + y_d**2) **0.5


# class Polygon():
#     def __init__(self, points):
#         self.points = points

#     def __len__(self):
#         len_points = len(self.points)
#         return len_points

#     def get_perimeter(self,points,len_points):
#         perimeter = 0
#         for i in range(0, len_points):
#             pt1 = self.points[i]
#             pt2 = self.points[i+1]
#             perimeter += pt1.distance(pt2)
#             if i + 1 == len_points:
#                 perimeter += points[-1].distance(points[0])
#             else:
#                 continue
#         return perimeter



# class Polygon():
#     def __init__(self, points):
#         self.points = points

#     def __len__(self):
#         return len(self.points)

#     def get_perimeter(self):
#         perimeter = 0
#         for i in range(0, len(self.points)):
#             pt1 = self.points[i]
#             pt2 = self.points[i+1]
#             perimeter += pt1.distance(pt2)
#             if i + 1 == len(self.points):
#                 perimeter += self.points[-1].distance(self.points[0])
#             else:
#                 continue
#         return perimeter