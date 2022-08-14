import cv2
import numpy as np
import random
import torch
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

DEVICE = torch.device("cpu")

def choice3(img):
    '''
    产生波浪型文字
    :param img:
    :return:
    '''
    h, w = img.shape[0:2]
    N = 5
    pad_pix = 50
    points = []
    dx = int(w/ (N - 1))
    for i in range( N):
        points.append((dx * i,  pad_pix))
        points.append((dx * i, pad_pix + h))

    #加边框
    img = cv2.copyMakeBorder(img, pad_pix, pad_pix, 0, 0, cv2.BORDER_CONSTANT,
                             value=(int(img[0][0][0]), int(img[0][0][1]), int(img[0][0][2])))

    #原点
    source = np.array(points, np.int32)
    source = source.reshape(1, -1, 2)

    #随机扰动幅度
    rand_num_pos = random.uniform(20, 30)
    rand_num_neg = -1 * rand_num_pos

    newpoints = []
    for i in range(N):
        rand = np.random.choice([rand_num_neg, rand_num_pos], p=[0.5, 0.5])
        if(i == 1):
            nx_up = points[2 * i][0]
            ny_up = points[2 * i][1] + rand
            nx_down = points[2 * i + 1][0]
            ny_down = points[2 * i + 1][1] + rand
        elif (i == 4):
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
    target = target.reshape(1, -1, 2)

    #计算matches
    matches = []
    for i in range(1, 2*N + 1):
        matches.append(cv2.DMatch(i, i, 0))

    return source, target, matches, img

def norm(points_int, width, height):
	"""
	将像素点坐标归一化至 -1 ~ 1
    """
	points_int_clone = torch.from_numpy(points_int).detach().float().to(DEVICE)
	x = ((points_int_clone * 2)[..., 0] / (width - 1) - 1)
	y = ((points_int_clone * 2)[..., 1] / (height - 1) - 1)
	return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)


class TPS(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, Y, w, h, device):

        """ 计算grid"""
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        grid = grid.view(-1, h * w, 2)

        """ 计算W, A"""
        n, k = X.shape[:2]
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)

        eps = 1e-9
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        K = D2 * torch.log(D2 + eps)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # Q = torch.solve(Z, L)[0]
        Q = torch.linalg.solve(L, Z)
        W, A = Q[:, :k], Q[:, k:]

        """ 计算U """
        eps = 1e-9
        D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        U = D2 * torch.log(D2 + eps)

        """ 计算P """
        n, k = grid.shape[:2]
        device = grid.device
        P = torch.ones(n, k, 3, device=device)
        P[:, :, 1:] = grid

        # grid = P @ A + U @ W
        grid = torch.matmul(P, A) + torch.matmul(U, W)
        return grid.view(-1, h, w, 2)

if __name__=='__main__':
    # 弯曲水平文本
    img = cv2.imread('dataset/test.png', cv2.IMREAD_COLOR)
    source, target, matches, img = choice3(img)
    # #opencv版tps
    # tps = cv2.createThinPlateSplineShapeTransformer()
    # tps.estimateTransformation(source, target, matches)
    # img = tps.warpImage(img)
    # cv2.imshow('test.png', img)
    # cv2.imwrite('test.png', img)
    # cv2.waitKey(0)

    #torch实现tps
    ten_img = ToTensor()(img).to(DEVICE)
    h, w = ten_img.shape[1], ten_img.shape[2]
    ten_source = norm(source, w, h)
    ten_target = norm(target, w, h)

    tps = TPS()
    warped_grid = tps(ten_target[None, ...], ten_source[None, ...], w, h, DEVICE)   #这个输入的位置需要归一化，所以用norm
    # ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0,align_corners=True)
    ten_wrp = F.grid_sample(ten_img[None, ...], warped_grid, padding_mode='border', align_corners=True)
    new_img_torch = np.array(ToPILImage()(ten_wrp[0].cpu()))

    # cv2.imshow('test.png', new_img_torch)
    cv2.imwrite('test.png', new_img_torch)
    cv2.waitKey(0)
