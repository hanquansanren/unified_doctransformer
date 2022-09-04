import cv2
import numpy as np
import random
import torch
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

DEVICE = torch.device("cpu")


class TPS(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, Y, w, h, device):
        """ 
        计算grid
        X是target点集，Y是source点集，w为1194，h为178，device='cpu'
        """
        grid = torch.ones(1, h, w, 2, device=device)
        # print(torch.linspace(-1, 1, 2))
        # print(torch.linspace(-1, 1, 2)[..., None])

        # meshgrid网格化
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None] # [..., None]相当于增加了一个维度，变成二维向量
        grid = grid.view(-1, h * w, 2) # torch.Size([1, 212532, 2])

        """ 计算W, A"""
        n, k = X.shape[:2] # k=10个点
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)

        eps = 1e-9 # 为了防止log函数输入为0
        print(X.shape)
        print(X[:, :, None, :].shape) # torch.Size([1, 10, 1, 2])
        print(X[:, None, :, :].shape) # torch.Size([1, 1, 10, 2])
        # 基于广播机制，当相减时，会统一变为torch.Size([1, 10, 10, 2])
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1) # torch.Size([1, 10, 10])
        K = D2 * torch.log(D2 + eps) # 这里是逐个元素的相乘，因为D2的主对角线上为全0，输出的K的主对角线也为全0

        P[:, :, 1:] = X # 扭曲点
        Z[:, :k, :] = Y # 标准点
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # Q = torch.solve(Z, L)[0]
        Q = torch.linalg.solve(L, Z)
        W, A = Q[:, :k, :], Q[:, k:, :] # torch.Size([1, 10, 2]),torch.Size([1, 3, 2])

        """ 计算U """
        eps = 1e-9
        D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1) # [1, 212532, 1, 2]-[1, 1, 10, 2]=[1, 212532, 10]
        U = D2 * torch.log(D2 + eps) # torch.Size([1, 212532, 10])

        """ 计算P """
        n, k = grid.shape[:2] # k=212532
        device = grid.device
        P = torch.ones(n, k, 3, device=device) # torch.Size([1, 212532, 3])
        P[:, :, 1:] = grid # torch.Size([1, 212532, 3])

        # grid = P @ A + U @ W
        grid = torch.matmul(P, A) + torch.matmul(U, W)
        return grid.view(-1, h, w, 2)


def choice3(img):
    '''
    产生波浪型文字
    param img in opencv form
    return:
        source: 原始的基准点,shape (1,10,2)
        target: 扰动(permutated)之后的点坐标 (1,10,2)
        matches:
        img: 经过padding的原图
    '''
    h, w = img.shape[0:2]
    N = 5
    pad_pix = 50
    points = []
    dx = int(w/ (N - 1))
    for i in range(N):
        points.append((dx * i,  pad_pix))
        points.append((dx * i, pad_pix + h))

    # #给输入图像添加padding，并就地覆盖原图
    print(img[0][0][0],img[0][0][1],img[0][0][2])
    img = cv2.copyMakeBorder(img, pad_pix, pad_pix, 0, 0, cv2.BORDER_CONSTANT,
                             value=(int(img[0][0][0]), int(img[0][0][1]), int(img[0][0][2])))
    # 该函数实现了padding
    # 上下填充，左右不填充

    # 基准参考点10=2*5个
    source = np.array(points, np.int32)
    source = np.expand_dims(source,axis=0) # shape: (10,2) --> (1,10,2)
    # source = source.reshape(1, -1, 2)

    #随机扰动幅度
    rand_num_pos = random.uniform(20, 30) # 从区间内，根据均匀分布，采样一个值
    rand_num_neg = -1 * rand_num_pos # 取负数

    newpoints = []
    for i in range(N):
        rand = np.random.choice([rand_num_neg, rand_num_pos], p=[0.5, 0.5])
        if(i == 1):
            nx_up = points[2 * i][0]
            ny_up = points[2 * i][1] + rand
            nx_down = points[2 * i + 1][0]
            ny_down = points[2 * i + 1][1] + rand
        elif (i == 3):
            rand = rand_num_neg if rand > 1 else rand_num_pos
            nx_up = points[2 * i][0]
            ny_up = points[2 * i][1] + rand
            nx_down = points[2 * i + 1][0]
            ny_down = points[2 * i + 1][1] + rand
        else:
            nx_up = points[2 * i][0]
            ny_up = points[2 * i][1]
            nx_down = points[2 * i + 1][0]
            ny_down = points[2 * i + 1][1]

        newpoints.append((nx_up, ny_up))
        newpoints.append((nx_down, ny_down))

    #target点
    target = np.array(newpoints, np.int32)
    target = np.expand_dims(target, axis=0)
    # target = target.reshape(1, -1, 2)

    #计算matches
    matches = []
    for i in range(1, 2*N + 1): #从1到10
        matches.append(cv2.DMatch(i, i, 0))

    return source, target, matches, img

def norm(points_int, width, height):
	"""
	将像素点坐标归一化至 -1 ~ 1
    """
	points_int_clone = torch.from_numpy(points_int).detach().float().to(DEVICE)
	x = ((points_int_clone * 2)[:, :, 0] / (width - 1) - 1)
	y = ((points_int_clone * 2)[:, :, 1] / (height - 1) - 1)
	return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)


if __name__=='__main__':
    # 弯曲水平文本
    img = cv2.imread('dataset/test.png', cv2.IMREAD_COLOR)
    source, target, matches, img = choice3(img)
    # # opencv版tps
    # tps = cv2.createThinPlateSplineShapeTransformer()
    # tps.estimateTransformation(source, target, matches)
    # img = tps.warpImage(img)
    # cv2.imshow('test.png', img)
    # cv2.imwrite('test.png', img)
    # cv2.waitKey(0)

    # torch实现tps
    ten_img = ToTensor()(img).to(DEVICE) # hwc -> chw, torch.Size([3, 178, 1194])
    h, w = ten_img.shape[1], ten_img.shape[2]
    ten_source = norm(source, w, h) # torch.Size([10, 2])
    ten_target = norm(target, w, h) # torch.Size([10, 2])
    # print(ten_source.max())
    # print(ten_source.min())

    tps = TPS()
    warped_grid = tps(ten_target[None, ...], ten_source[None, ...], w, h, DEVICE)   #这个输入的位置需要归一化，所以用norm
    # ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0,align_corners=True)
    # input: (n,c,h,w), grid: (n,h,w,2)
    ten_wrp = F.grid_sample(ten_img[None, ...], warped_grid, padding_mode='border', align_corners=True) # output: (n,c,h,w)
    new_img_torch = np.array(ToPILImage()(ten_wrp[0].cpu()))

    # cv2.imshow('test.png', new_img_torch)
    cv2.imwrite('test.png', new_img_torch)
