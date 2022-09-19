import cv2
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import time

# 计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列


def compute_polygon_area(points):
    point_num = len(points)
    if (point_num < 3):
        return 0.0
    s = points[0][1] * (points[point_num-1][0] - points[1][0])
    # for i in range(point_num): # (int i = 1 i < point_num ++i):
    for i in range(1, point_num):  # 有小伙伴发现一个bug，这里做了修改，但是没有测试，需要使用的亲请测试下，以免结果不正确。
        s += points[i][1] * (points[i-1][0] - points[(i+1) % point_num][0])
    return abs(s/2.0)


def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


if __name__ == '__main__':
    img = np.zeros((1080, 1920, 3), dtype=np.int32)
    # pts = np.array([[200, 200], [400, 400],[600, 200],  [400, 600]])
    pts = np.array([[25, 70],
                    [25, 160],
                    [110, 200],
                    [200, 160],
                    [200, 70],
                    [110, 20]], np.int32)
    x, y = pts[:, 0], pts[:, 1]

    a = cv2.fillPoly(img, [pts], (255, 255, 255))
    cv2.imwrite('./simple_test/interpola_vis/kkkkkk_{}.png'.format(666), a)

    start = time.time()
    print(Polygon(pts).area)
    # print(PolyArea(x,y)) # 最慢
    # print(compute_polygon_area(pts)) # 次慢
    end = time.time()
    print("顺序执行时间：", end - start)


# pts = np.random.rand(6, 2) # (6,2) x,y
# # x, y = coords[:, 0], coords[:, 1]
# # pts = np.array([[25, 70],
# #                 [25, 160],
# #                 [110, 200],
# #                 [200, 160],
# #                 [200, 70],
# #                 [110, 20]], np.int32)
# pts = pts.reshape((-1, 1, 2))


# image = np.zeros((1, 1, 3), dtype=np.float64)
# roi_as = []
# roi_as.append(pts)
# a = cv2.polylines(image, roi_as, True, (255, 0, 255))
# cv2.imwrite('./simple_test/interpola_vis/kk_{}.png'.format(666), a)
# cv2.waitKey()

# print(PolyArea(x, y)-Polygon(coords).area)


# # # 计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列
# # def compute_polygon_area(points):
# #     point_num = len(points)
# #     if(point_num < 3): return 0.0
# #     s = points[0][1] * (points[point_num-1][0] - points[1][0])
# #     #for i in range(point_num): # (int i = 1 i < point_num ++i):
# #     for i in range(1, point_num): # 有小伙伴发现一个bug，这里做了修改，但是没有测试，需要使用的亲请测试下，以免结果不正确。
# #         s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
# #     return abs(s/2.0)

# # if __name__ == '__main__':
# #     # polygon = [[0,0], [2,0],[2,2], [0,2]] #4.0
# #     polygon = [[3,3],[4,2],[6,1],[7,6],[9,7],[3,16],[0,3],[2,4],[1,5],[6,6]] #62.0
# #     # polygon = [[3,3],[4,2],[6,4],[7,6],[9,7],[3,9],[0,5],[2,4],[4,4]] #29.0
# #     print(compute_polygon_area(polygon))


# # image =np.zeros((500,500,3),dtype=np.uint8)
# # roi_as = []
# # roi_as.append(np.array([[194 ,456],[172 ,82] ,[194 ,86], [172 ,382]],dtype=int))
# # a=cv2.polylines(image, roi_as, True, (255, 0, 255))  # 画任意多边形
# # cv2.imwrite('./simple_test/interpola_vis/kk_{}.png'.format(666),a)
# # cv2.waitKey()
