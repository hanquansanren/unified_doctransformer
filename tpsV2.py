'''
2022/9/30
Weiguang Zhang
V2 is a final version
'''
from gettext import bind_textdomain_codeset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class createThinPlateSplineShapeTransformer(nn.Module):

    def __init__(self, I_r_size, fiducial_num=[31, 31], device=torch.device('cuda:0')):
        """
        通过初始化中的实例化estimateTransformation，间接实现了TPS。该类中仅包含F.grid_sample的过程
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_r_size : (height, width) of the rectified image I_r
            fiducial_num : the number of fiducial_points
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(createThinPlateSplineShapeTransformer, self).__init__()
        self.f_row_num, self.f_col_num = fiducial_num # (31, 31)
        self.F = self.f_row_num * self.f_col_num # 961 总控制点数量
        self.I_r_size = list(map(int, I_r_size)) # (h, w)
        self.device = device
        self.estimateTransformation = estimateTransformation(self.F, self.I_r_size, self.device)
        # input: 961, (h, w)

    def forward(self, batch_I, batch_F, output_shape=None):
        '''
        batch_I:(tensor) [1, 3, 1521, 1137] 输入图像，tensor形式
        batch_F:(tensor) [1, 961, 2] 预测的warped 图像控制点，行序优先(因为采用了reshape，默认行序优先)
        output_shape:(list) h,w,[1521, 1137] 用于生成regular的参考点
        '''
        # 这一步就完成了shrunken h*w形式的 backward mapping构建
        build_P_prime = self.estimateTransformation.build_P_prime(batch_F)  
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])  # build_P_prime.size(0) == mini-batch size,
        # # output: [1, shrunken h, shrunken w, 2]

        if output_shape is None:
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border', align_corners=True)
        else:
            build_P_prime_reshape = build_P_prime_reshape.transpose(2, 3).transpose(1, 2) # [1, 320, 320, 2]-->[1, 2, 320, 320]
            map = F.interpolate(build_P_prime_reshape, output_shape,  mode='bilinear', align_corners=True) # [1, 2, 320, 320]-->[1, 2, 1521, 1137] # 这里的插值函数仅支持NCHW的形式，故而需要手动转换
            map = map.transpose(1, 2).transpose(2, 3) # [1, 2, 1521, 1137]-->[1, 1521, 1137, 2] 至此，获得了backward mapping
            batch_I_r = F.grid_sample(batch_I, map, padding_mode='border', align_corners=True)
        return batch_I_r # output: [1, 3, h, w]

class estimateTransformation(nn.Module):

    def __init__(self, F, I_r_size, device):
        super(estimateTransformation, self).__init__()
        self.eps = 1e-6 # 为了防止径向基函数中的log()为0，故而加入一个很小的数值
        self.I_r_height, self.I_r_width = I_r_size # shrunken (h, w)
        self.F = F # 961
        self.device = device
        self.C = self._build_C(self.F) # (961,2)
        self.P = self._build_P(self.I_r_width, self.I_r_height) # (102400,2)
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C), dtype=torch.float64, device=self.device)) 
        # self.P_hat = torch.tensor(self._build_P_hat(self.F, self.C, self.P), dtype=torch.float64, device=self.device)
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P), dtype=torch.float64, device=self.device))

    def _build_C(self, F):
        '''
        构造归一化到[-1 to 1]的参考点矩阵，一共961个点的坐标
        '''
        im_x, im_y = np.mgrid[-1:1:complex(31), -1:1:complex(31)]
        # C = np.stack((im_x,im_y), axis=2).reshape(-1,2) # (961,2),列序优先，且先w(x)后h(y)
        C = np.stack((im_y,im_x), axis=2).reshape(-1,2) # (961,2),行序优先，且先w(x)后h(y)
        return C  # (961,2) from [-1 to 1] 闭区间，边界值可以取到

    def _build_inv_delta_C(self, F, C): 
        '''
        inv_delta_C是基于964*964的大矩阵，这里不采用LU分解的方式求解，而是直接求逆矩阵。
        在[-1 to 1]闭区间内，构造参考点阵列，并写成逆矩阵形式。
        '''
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r # 以对角线为对称轴，镜像填充
        np.fill_diagonal(hat_C, 1) # 因为主对角线为全0，无法求径向基函数，因此需要用1填充
        hat_C = (hat_C ** 2) * np.log(hat_C ** 2) 
        # 20220903完善，增加了行序优先和列序优先的选项
        delta_C = np.concatenate( # 行序优先版
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1),  # 1 x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
            ],
            axis=0
        )
        # delta_C = np.concatenate( # 列序优先版
        #     [
        #         np.concatenate([hat_C, np.ones((F, 1)), C], axis=1),  # F x F+3
        #         np.concatenate([np.ones((1, F)), np.zeros((1, 3))], axis=1),  # 1 x F+3
        #         np.concatenate([np.transpose(C), np.zeros((2, 3))], axis=1),  # 2 x F+3
        #     ],
        #     axis=0
        # )
        inv_delta_C = np.linalg.inv(delta_C) # 求逆
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        # 20220903更改为闭区间
        # I_r_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width (320,)
        # I_r_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height (320,)
        I_r_x = np.linspace(-1, 1, I_r_width)   # self.I_r_width (320,)
        I_r_y = np.linspace(-1, 1, I_r_height)  # self.I_r_height (320,)        
        
        I_r_grid_x, I_r_grid_y = np.meshgrid(I_r_x, I_r_y)
        # 最开始的版本
        # P = np.stack(np.meshgrid(I_r_x, I_r_y),axis=2).reshape(-1, 2) # (320,320,2) -> (102400,2)
        P = np.stack((I_r_grid_x,I_r_grid_y),axis=2).reshape(-1, 2) # (320,320,2) -> (102400,2) # 行序优先
        # P = np.stack((I_r_grid_y,I_r_grid_x),axis=2).reshape(-1, 2) # (320,320,2) -> (102400,2) # 列序优先
        return P  

    def _build_P_hat(self, F, C, P):
        '''
        P_hat也是大矩阵，是根据参考点和P
        C: (961,2) from [-1 to 1] 行序优先
        P: (102400,2) from [-1 to 1] 行序优先
        '''
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height) # 102400
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2  ==(102400,1,2) tile in (1,961,1) = (102400, 961, 2)
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2 # (961,2)->(1,961,2)
        P_diff = P_tile - C_tile  # nxFx2 #(102400, 961, 2)-(1,961,2)=(102400, 961, 2)
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = 2 * np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3  # 102400 * 964

    # 主函数，构造backward mapping
    def build_P_prime(self, batch_C_prime):
        batch_size = batch_C_prime.size(0) # batch_C_prime= (1,961,2),是warped图像上的控制点位置（已归一化）
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(batch_size, 3, 2).double().to(self.device)), dim=1)  # batch_size x F+3 x 2 
        # 实现 (1,961,2)+(1,3,2) in dim1= (1,964,2)，获得真实图像上的控制点
        batch_T = torch.matmul(self.inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2 # [964, 964]*[1,964,2] (==[1,964, 964]*[1,964,2]) =[1, 964, 2]
        # 与求逆后的大矩阵相乘，获得参数转移矩阵T
        
        batch_P_prime = torch.matmul(self.P_hat, batch_T)  # batch_size x n x 2  # [102400, 964]*[1, 964, 2]=[1, 102400, 2]
        return batch_P_prime  # batch_size x n x 2 
