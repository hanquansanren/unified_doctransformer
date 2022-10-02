import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class get_dewarped_intermediate_result(nn.Module):

    def __init__(self, output_size=[992/5, 992/5], pt_num=[31, 31], device=torch.device('cuda:0')):
        """
        通过初始化中的实例化estimateTransformation，间接实现了TPS。该类中仅包含F.grid_sample的过程
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            output_size : (height, width) of the rectified image I_r
            pt_num : the number of fiducial_points
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(get_dewarped_intermediate_result, self).__init__()
        self.f_row_num, self.f_col_num = pt_num # (31, 31)
        self.F = self.f_row_num * self.f_col_num # 961 总控制点数量
        self.output_size = list(map(int, output_size)) # (h, w)(198,198)
        # self.device = device
        self.lowpart, self.hpf = self.fdr()
        # self.estimateTransformation = estimateTransformation(self.F, self.output_size, self.device)
        self.estimateTransformation = estimateTransformation(self.F, self.output_size)
        # input: 961, (h, w)

    def fdr(self):        
        blank_im = 255*torch.ones((1, 3, 992, 992)).int()
        h,w= blank_im.shape[2:]        
        lpf = torch.zeros((1,3,h,w))
        sh = h*0.06 
        sw = w*0.06
        for x in range(w):
            for y in range(h):
                if (x<(w/2+sw)) and (x>(w/2-sw)) and (y<(h/2+sh)) and (y>(h/2-sh)):
                    lpf[:,:,y,x] = 1
        hpf = 1-lpf

        bfreq = torch.fft.fft2(blank_im)
        bfreq = torch.fft.ifftshift(bfreq).cuda()
        return (lpf.cuda())*bfreq, hpf.cuda()   # lpf*bfreq = low part, 1-lpf = hpf


    def fourier_transform(self, im, bfreq = None, hpf = None, batch_num = None):
        '''
        im: [b, 3, 992, 992]
        bfrep: (1, 3, 992, 992)
        hpf: (1, 3, 992, 992)
        '''
        batch_img = torch.empty((1, 3, 992, 992)).cuda()
        for batch in range(batch_num):
            # fft_img[:,:,batch] = np.fft.fftshift(np.fft.fft2(img[:,:,batch]))
            freq = torch.fft.ifftshift(torch.fft.fft2(im[None,batch]))
            rhpf = bfreq + hpf * freq
            img_rhpf = torch.abs(torch.fft.ifft2(rhpf))
            img_rhpf = torch.clip(img_rhpf,0,255) 
            batch_img = torch.cat((batch_img, img_rhpf),dim=0)




        # img_rhpf = img_rhpf.int().numpy().transpose(0, 2, 3, 1).squeeze(0)
        # cv2.imwrite(file_path[:-4]+'_rhpf_torch.png',img_rhpf)
        return batch_img[1:]



    def forward(self, batch_I, batch_ref, batch_pt, output_shape = [992,992], ptd1=None):
        '''
        batch_I:(tensor) [b, 3, 992, 992] 输入warped图像, requires_grad=False
        batch_ref:(tensor) [b, 3, 992, 992] 输入reference digital图像, requires_grad=False
        batch_pt:(tensor) [b, 2, 31, 31] 预测的warped图像控制点,行序优先(因为采用了reshape/view,默认行序优先) requires_grad=True,梯度将从这里回传
        output_shape:(list) h,w,[992, 992] 用于生成regular的参考点
        '''
        if ptd1 is None:
            # 完成shrunken h*w形式的 backward mapping构建
            batch_num = batch_pt.shape[0]
            batch_pt_norm = batch_pt / 992 # [b, 2, 31, 31]
            batch_pt_norm = batch_pt_norm.permute(0,3,2,1).reshape(batch_num,-1,2) # [b, 2, 31, 31] ->  [b, 31, 31, 2] -> [b, 961, 2] 
            # batch_pt_norm = (batch_pt_norm[None,:]-0.5)*2
            batch_pt_norm = (batch_pt_norm[:]-0.5)*2
            build_P_prime = self.estimateTransformation.build_P_prime(batch_pt_norm)  
            build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.output_size[0], self.output_size[1], 2])  # build_P_prime.size(0) == mini-batch size,
            # # output: [b, shrunken h, shrunken w, 2] [b, 198, 198, 2]

            # fourier dewarp part
            batch_I_ffted = self.fourier_transform(batch_I, self.lowpart, self.hpf, batch_num)
            batch_ref_ffted = self.fourier_transform(batch_ref, self.lowpart, self.hpf, batch_num)

            build_P_prime_reshape = build_P_prime_reshape.transpose(2, 3).transpose(1, 2) # [b, 198, 198, 2]-->[b, 2, 198, 198]
            map = F.interpolate(build_P_prime_reshape, output_shape,  mode='bilinear', align_corners=True) # [b, 2, 198, 198]-->[b, 2, 992, 992] # 这里的插值函数仅支持NCHW的形式，故而需要手动转换
            map = map.transpose(1, 2).transpose(2, 3) # [b, 2, 992, 992]-->[b, 992, 992, 2] 至此，获得了backward mapping, requires_grad=True
            batch_I_r = F.grid_sample(batch_I_ffted, map, padding_mode='border', align_corners=True) # [b, 3, 992, 992], requires_grad=True
            return batch_I_r, batch_ref_ffted # output: [b, 3, h, w]
        elif ptd1 is not None:
            # 完成shrunken h*w形式的 backward mapping构建
            batch_num = batch_pt.shape[0]
            batch_pt_norm = batch_pt / 992 # [b, 2, 31, 31]
            batch_pt_norm = batch_pt_norm.permute(0,3,2,1).reshape(batch_num,-1,2) # [b, 2, 31, 31] ->  [b, 31, 31, 2] -> [b, 961, 2] 
            # batch_pt_norm = (batch_pt_norm[None,:]-0.5)*2
            batch_pt_norm = (batch_pt_norm[:]-0.5)*2
            # key difference
            build_P_prime = self.estimateTransformation.build_P_prime_for_d1(batch_pt_norm, ptd1)   
            build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.output_size[0], self.output_size[1], 2])  # build_P_prime.size(0) == mini-batch size,
            # # output: [b, shrunken h, shrunken w, 2] [b, 198, 198, 2]

            # fourier dewarp part
            batch_I_ffted = self.fourier_transform(batch_I, self.lowpart, self.hpf, batch_num)
            batch_ref_ffted = self.fourier_transform(batch_ref, self.lowpart, self.hpf, batch_num)


            build_P_prime_reshape = build_P_prime_reshape.transpose(2, 3).transpose(1, 2) # [b, 198, 198, 2]-->[b, 2, 198, 198]
            map = F.interpolate(build_P_prime_reshape, output_shape,  mode='bilinear', align_corners=True) # [b, 2, 198, 198]-->[b, 2, 992, 992] # 这里的插值函数仅支持NCHW的形式，故而需要手动转换
            map = map.transpose(1, 2).transpose(2, 3) # [b, 2, 992, 992]-->[b, 992, 992, 2] 至此，获得了backward mapping, requires_grad=True
            batch_I_r = F.grid_sample(batch_I_ffted, map, padding_mode='border', align_corners=True) # [b, 3, 992, 992], requires_grad=True
            return batch_I_r, batch_ref_ffted # output: [b, 3, h, w]            
        else:
            print("error in tps loss process")






class estimateTransformation(nn.Module):

    def __init__(self, F, output_size):
        super(estimateTransformation, self).__init__()
        self.eps = 1e-6 # 为了防止径向基函数中的log()为0，故而加入一个很小的数值
        self.I_r_height, self.I_r_width = output_size # shrunken (h, w) (198,198)
        self.F = F # 961
        self.C = self._build_C(self.F) # (961,2)
        self.P = self._build_P(self.I_r_width, self.I_r_height) # (39204,2)
        # self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C), dtype=torch.float64, device=self.device)) 
        # self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P), dtype=torch.float64, device=self.device))
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C), dtype=torch.float64)) 
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P), dtype=torch.float64))
    def _build_d1_C(self, F, pts):
        '''
        构造归一化到[-1 to 1]的参考点矩阵，一共961个点的坐标
        pts=[b,2,31,31]
        '''
        batch_num = pts.shape[0]
        # batch_pt_norm = pts / 992 # [b, 2, 31, 31]
        # batch_pt_norm = batch_pt_norm.permute(0,3,2,1).reshape(batch_num,-1,2) # [b, 2, 31, 31] ->  [b, 31, 31, 2] -> [b, 961, 2] 
        # batch_pt_norm = (batch_pt_norm[:]-0.5)*2
        pts = pts.data.cpu().numpy().transpose(0, 2, 3, 1) # [b,31,31,2]
        pts = pts / [992, 992] # 归一化，模型输出点稀疏点本身就在992*992的范围内[b,31,31,2]
        pts = pts.transpose(0,2,1,3).reshape(batch_num,-1,2) # [b,31,31,2] -> [b,961,2] HWC->WHC->(WH)C
        pts = (pts[:]-0.5)*2 

        return pts  # (b,961,2) from [-1 to 1] 闭区间，边界值可以取到

    def _build_inv_delta_d1_C(self, F, C): 
        '''
        inv_delta_C是基于964*964的大矩阵，这里不采用LU分解的方式求解，而是直接求逆矩阵。
        在[-1 to 1]闭区间内，构造参考点阵列，并写成逆矩阵形式。
        C:(b,961,2)
        '''
        hat_C = np.zeros((C.shape[0],F, F), dtype=float)  # b x F x F
        for m in range(C.shape[0]):
            for i in range(0, F):
                for j in range(i, F):
                    r = np.linalg.norm(C[m,i] - C[m,j])
                    hat_C[m,i, j] = r
                    hat_C[m,j, i] = r # 以对角线为对称轴，镜像填充
        for m in range(C.shape[0]):
            np.fill_diagonal(hat_C[m] , 1) # 因为主对角线为全0，无法求径向基函数，因此需要用1填充
            hat_C[m] = (hat_C[m] ** 2) * np.log(hat_C[m] ** 2)
        # 20220903完善，增加了行序优先和列序优先的选项
        delta_C = np.empty((C.shape[0],F+3,F+3))
        inv_delta_C = np.empty((C.shape[0],F+3,F+3))
        for m in range(C.shape[0]):
            delta_C[m] = np.concatenate( # 行序优先版
                [
                    np.concatenate([np.ones((F, 1)), C[m], hat_C[m]], axis=1),  # F x F+3
                    np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1),  # 1 x F+3
                    np.concatenate([np.zeros((2, 3)), np.transpose(C[m])], axis=1),  # 2 x F+3
                ],
                axis=0
            )
            inv_delta_C[m] = np.linalg.inv(delta_C[m]) # 求逆
        return inv_delta_C  # b x F+3 x F+3 # (b,964,964)

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
        return P   # (30204,2)

    def _build_P_hat(self, F, C, P):
        '''
        P_hat也是大矩阵，是根据参考点和P
        C: (961,2) from [-1 to 1] 行序优先
        P: (39204,2) from [-1 to 1] 行序优先
        '''
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height) # 102400
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2  ==(102400,1,2) tile in (1,961,1) = (102400, 961, 2)
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2 # (961,2)->(1,961,2)
        P_diff = P_tile - C_tile  # nxFx2 #(102400, 961, 2)-(1,961,2)=(102400, 961, 2)
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = 2 * np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3  # 39204 * 964
    
    def _build_d1_P(self, I_r_width, I_r_height, pts):
        
        I_r_x = np.linspace(-1, 1, I_r_width)   # self.I_r_width (320,)
        I_r_y = np.linspace(-1, 1, I_r_height)  # self.I_r_height (320,)        
        
        I_r_grid_x, I_r_grid_y = np.meshgrid(I_r_x, I_r_y)
        # 最开始的版本
        # P = np.stack(np.meshgrid(I_r_x, I_r_y),axis=2).reshape(-1, 2) # (320,320,2) -> (102400,2)
        P = np.stack((I_r_grid_x,I_r_grid_y),axis=2).reshape(-1, 2) # (320,320,2) -> (102400,2) # 行序优先
        # P = np.stack((I_r_grid_y,I_r_grid_x),axis=2).reshape(-1, 2) # (320,320,2) -> (102400,2) # 列序优先
        return P   # (30204,2)

    def _build_P_d1_hat(self, F, C, P):
        '''
        P_hat也是大矩阵，是根据参考点和P
        C: (b,961,2) 
        P: (b,39204,2) 
        '''
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height) # 102400
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2  ==(102400,1,2) tile in (1,961,1) = (102400, 961, 2)
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2 # (961,2)->(1,961,2)
        P_diff = P_tile - C_tile  # nxFx2 #(102400, 961, 2)-(1,961,2)=(102400, 961, 2)
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = 2 * np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3  # 39204 * 964

    # 主函数，构造backward mapping
    def build_P_prime(self, batch_C_prime):
        batch_size = batch_C_prime.size(0) # batch_C_prime= (1,961,2),是warped图像上的控制点位置（已归一化到(-1,1)）
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(batch_size, 3, 2).double().cuda()), dim=1)  # batch_size x F+3 x 2 
        # 实现 (1,961,2)+(1,3,2) in dim1= (1,964,2)，获得真实图像上的控制点
        batch_T = torch.matmul(self.inv_delta_C.repeat(batch_size,1,1), batch_C_prime_with_zeros)  # batch_size x F+3 x 2 # [b, 964, 964]*[b,964,2] = [b, 964, 2]
        # 与求逆后的大矩阵相乘，获得参数转移矩阵T
        
        batch_P_prime = torch.matmul(self.P_hat.repeat(batch_size,1,1), batch_T)  # batch_size x n x 2  # [39204, 964]*[b, 964, 2]=[1, 102400, 2] torch.Size([2, 39204, 2])
        return batch_P_prime  # batch_size x n x 2 

    # 主函数2，构造backward mapping
    def build_P_prime_for_d1(self, batch_C_prime, pt_d1):
        batch_size = batch_C_prime.size(0) # batch_C_prime= (1,961,2),是warped图像上的控制点位置（已归一化到(-1,1)）
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(batch_size, 3, 2).double().cuda()), dim=1)  # batch_size x F+3 x 2 
        # 实现 (1,961,2)+(1,3,2) in dim1= (1,964,2)，获得真实图像上的控制点
        self.D1C = self._build_d1_C(self.F, pt_d1) # (b,961,2)
        self.inv_delta_d1C = torch.tensor(self._build_inv_delta_d1_C(self.F, self.D1C), dtype=torch.float64).cuda()
        batch_T = torch.matmul(self.inv_delta_d1C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2 # [b, 964, 964]*[b,964,2] = [b, 964, 2]
        # 与求逆后的大矩阵相乘，获得参数转移矩阵T
        self.D1P = self._build_d1_P(self.I_r_width, self.I_r_height, pt_d1)
        self.d1P_hat = torch.tensor(self._build_P_d1_hat(self.F, self.D1C, self.D1P), dtype=torch.float64).cuda()

        batch_P_prime = torch.matmul(self.d1P_hat, batch_T)  # batch_size x n x 2  # [39204, 964]*[b, 964, 2]=[1, 102400, 2] torch.Size([2, 39204, 2])
        return batch_P_prime  # batch_size x n x 2 