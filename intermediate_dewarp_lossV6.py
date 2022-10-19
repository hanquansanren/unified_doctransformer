'''
2022/10/12
Weiguang Zhang
V6 means pure FDRNet + 8*8 output
'''
from ast import Not
from queue import Empty
from select import select
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import random
import time
import cv2
from tps_warp import TpsWarp

class get_dewarped_intermediate_result(nn.Module):

    def __init__(self, output_size=[200, 200], pt_num=[8, 8], device=torch.device('cuda:2')):
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
        self.f_row_num, self.f_col_num = pt_num # (8, 8)
        self.F = self.f_row_num * self.f_col_num # 64 总控制点数量
        self.output_size = list(map(int, output_size)) # (h, w)(400,400)
        self.device = device
        # self.lowpart, self.hpf = self.fdr()
        self.estimateTransformation = estimateTransformation(self.F, self.output_size)
        # input: 64, (h, w)
        self.tpswarper = TpsWarp(11)

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

    def perturb_warp(self, batch_size):
        B = batch_size
        s = 11
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s),indexing='ij') 
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to('cuda').expand(B, -1, -1, -1)
        # t:(B,2,11,11)
        perturbed_wi_list=[]
        for num in range(2):
            tt = t.clone()
            nd = random.randint(0, 40) # 4
            for ii in range(nd): # 控制点数量
                # define deformation on bd
                pm = (torch.rand(B, 1) - 0.5) * 0.2  # 从区间【-0.1,0.1】均匀采样四个点
                ps = (torch.rand(B, 1) - 0.5) * 1.95 # 从区间【-0.975,0.975】均匀采样四个点
                pt = ps + pm
                pt = pt.clamp(-0.975, 0.975) # 限幅器 [[ 0.4252,  0.7563, -0.7394, -0.1466]]
                # put it on one bd
                # [1, 1] or [-1, 1] or [-1, -1] etc
                a1 = (torch.rand(B, 2) > 0.5).float() * 2 -1 # 4行2列，为随机的+1和-1
                # select one col for every row
                a2 = torch.rand(B, 1) > 0.5 # 4行1列，[[False],[False],[False],[ True]]
                a2 = torch.cat([a2, a2.bitwise_not()], dim=1) # bitwise_not()为按元素取反
                a3 = a1.clone()
                a3[a2] = ps.view(-1)
                ps = a3.clone()
                a3[a2] = pt.view(-1)
                pt = a3.clone()
                # 2 N 4
                bds = torch.stack([
                    t[0, :, 1 : -1, 0], t[0, :, 1 : -1, -1], t[0, :, 0, 1 : -1], t[0, :, -1, 1 : -1]
                ], dim=2) # 基准点的边缘(2,9,4)

                pbd = a2.bitwise_not().float() * a1
                # id of boundary p is on
                pbd = torch.abs(0.5 * pbd[:, 0] + 2.5 * pbd[:, 1] + 0.5).long()
                # ids of other boundaries
                pbd = torch.stack([pbd + 1, pbd + 2, pbd + 3], dim=1) % 4
                # print(pbd)
                pbd = bds[..., pbd].permute(2, 0, 1, 3).reshape(B, 2, -1)            

                srcpts = torch.stack([
                    t[..., 0, 0], t[..., 0, -1], t[..., -1, 0], t[..., -1, -1],
                    ps.to('cuda')
                ], dim=2)
                srcpts = torch.cat([pbd, srcpts], dim=2).permute(0, 2, 1)
                dstpts = torch.stack([
                    t[..., 0, 0], t[..., 0, -1], t[..., -1, 0], t[..., -1, -1],
                    pt.to('cuda')
                ], dim=2)
                dstpts = torch.cat([pbd, dstpts], dim=2).permute(0, 2, 1)
                # print(srcpts)
                # print(dstpts)
                tgs = self.tpswarper(srcpts, dstpts)
                tt = F.grid_sample(tt, tgs.permute(0, 2, 3, 1), align_corners=True)

            nd = random.randint(1, 40) # 5
            for ii in range(nd):

                pm = (torch.rand(B, 2) - 0.5) * 0.2
                ps = (torch.rand(B, 2) - 0.5) * 1.95
                pt = ps + pm
                pt = pt.clamp(-0.975, 0.975)

                srcpts = torch.cat([
                    t[..., -1, :], t[..., 0, :], t[..., 1 : -1, 0], t[..., 1 : -1, -1],
                    ps.unsqueeze(2).to('cuda')
                ], dim=2).permute(0, 2, 1)
                dstpts = torch.cat([
                    t[..., -1, :], t[..., 0, :], t[..., 1 : -1, 0], t[..., 1 : -1, -1],
                    pt.unsqueeze(2).to('cuda')
                ], dim=2).permute(0, 2, 1)
                tgs = self.tpswarper(srcpts, dstpts)
                tt = F.grid_sample(tt, tgs.permute(0, 2, 3, 1), align_corners=True)
            tgs = tt
            perturbed_wi_list.append(tgs)
        return perturbed_wi_list


    def forward(self, batch_I, batch_ref=None, batch_src_pt=None, batch_trg_pt=None, output_shape = [992,992], trg_mask = None):
        '''
        batch_I:(tensor) [b, 3, 992, 992] 输入warped图像, requires_grad=False
        batch_ref:(tensor) [b, 3, 992, 992] 输入reference digital图像, requires_grad=False
        batch_src_pt:(tensor) [b, 2, 8, 8] 预测的warped图像控制点,行序优先(因为采用了reshape/view,默认行序优先) requires_grad=True,梯度将从这里回传
        batch_trg_pt:(tensor) [b, 2, 8, 8] 预测的flatten图像控制点,行序优先(因为采用了reshape/view,默认行序优先) requires_grad=False,不传梯度
        output_shape:(list) h,w,[992, 992] 用于生成regular的参考点
        trg_mask: [b,3,992,992] 需要加密的target点集
        '''
        
        if batch_trg_pt is None:
            self.lowpart, self.hpf = self.fdr()
            # 完成shrunken h*w形式的 backward mapping构建
            batch_num = batch_src_pt.shape[0]
            # batch_src_pt = batch_src_pt[:,:,::3,::3] / 992 # [b, 2, 31, 31] -> [b, 2, 11, 11]
            batch_src_pt = batch_src_pt / 992 # [b, 2, 8, 8]
            batch_src_pt = batch_src_pt.permute(0,3,2,1).reshape(batch_num,-1,2) # [b, 2, 31, 31] ->  [b, 31, 31, 2] -> [b, 961, 2] 
            # batch_pt_norm = (batch_pt_norm[None,:]-0.5)*2
            batch_src_pt = (batch_src_pt[:]-0.5)*2
            build_P_prime = self.estimateTransformation.build_P_prime(batch_src_pt)  
            build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.output_size[0], self.output_size[1], 2])  # build_P_prime.size(0) == mini-batch size,
            # # output: [b, shrunken h, shrunken w, 2] [b, 198, 198, 2]

            # fourier dewarp part
            batch_I_ffted = self.fourier_transform(batch_I, self.lowpart, self.hpf, batch_num)
            batch_ref_ffted = self.fourier_transform(batch_ref, self.lowpart, self.hpf, batch_num)
            # flat_show = batch_I_ffted[0].detach().cpu().numpy().transpose(1,2,0) # NCHW-> NHWC (h, w, 3), dtype('float32')
            # flat_show = flat_show.astype(np.uint8) # dtype('float32') -> dtype('uint8')
            # cv2.imwrite('./mark_origin_fft.png', flat_show)
            # batch_I_ffted = batch_I
            # batch_ref_ffted = batch_ref

            build_P_prime_reshape = build_P_prime_reshape.transpose(2, 3).transpose(1, 2) # [b, 198, 198, 2]-->[b, 2, 198, 198]
            map = F.interpolate(build_P_prime_reshape, output_shape,  mode='bilinear', align_corners=True) # [b, 2, 198, 198]-->[b, 2, 992, 992] # 这里的插值函数仅支持NCHW的形式，故而需要手动转换
            map = map.transpose(1, 2).transpose(2, 3) # [b, 2, 992, 992]-->[b, 992, 992, 2] 至此，获得了backward mapping, requires_grad=True
            batch_I_r = F.grid_sample(batch_I_ffted, map, padding_mode='border', align_corners=True) # [b, 3, 992, 992], requires_grad=True
            return batch_I_r, batch_ref_ffted # output: [b, 3, h, w]
        # 完成shrunken h*w形式的 backward mapping构建
        elif batch_trg_pt is not None:
            # source pts normalized to range within [-1, 1] 
            batch_num = batch_src_pt.shape[0]
            # batch_src_pt = batch_src_pt[:,:,::3,::3] / 992 # [b, 2, 8, 8] -> [b, 2, 8, 8]
            batch_src_pt = batch_src_pt / 992 # [b, 2, 8, 8]
            batch_src_pt = batch_src_pt.permute(0,3,2,1).reshape(batch_num,-1,2) # [b, 2, 8, 8] ->  [b, 8, 8, 2] -> [b, 64, 2] 
            batch_src_pt = (batch_src_pt[:]-0.5)*2 # [b, 64, 2]
            
            # mask downsampling: [b, 3, 992, 992] -> [b, 3, 400, 400] 
            trg_mask = trg_mask[:,0:1,:,:].float() # [b, 3, 992, 992] -> [b, 1, 992, 992] 
            trg_mask = F.interpolate(trg_mask, [200,200],  mode='bilinear', align_corners=True) # [b, 1, 992, 992] -> [b, 1, 400, 400] 
            trg_mask = trg_mask.cpu().numpy() # (b, 1, 400, 400)
            
            # mask to coodinate
            batch_coodinate = np.zeros((batch_num, 40000, 2))
            for b in range(batch_num):
                c_short = np.argwhere(trg_mask[b,0,...]>0)
                c_short = np.pad(c_short, (0,40000-len(c_short)), 'constant', constant_values=-100)[:,0:2]
                batch_coodinate[b] = c_short

            build_P_prime = self.estimateTransformation.build_P_prime_for_d1(batch_src_pt, batch_trg_pt, batch_coodinate)
            # out: (b, 160000, 2)，可能在尾部包含大量冗余，需要结合batch_coodinate(batch_num, 160000, 2)进行剔除
            # BM = torch.full((batch_num, 400, 400, 2), float('nan')).cuda().float() 
            BM = torch.full((batch_num, 200, 200, 2), 0.).cuda().float()
            

            # fill in small bm from dense local pt bm
            for b in range(batch_num):
                redundant_c = np.argwhere(batch_coodinate[b] <= -99)
                # print(len(batch_coodinate[b]))
                # print(redundant_c.size)
                if redundant_c.size != 0:
                    # print(np.argwhere(batch_coodinate[b] <= -99)[0][0])
                    for jj in range(np.argwhere(batch_coodinate[b] <= -99)[0][0]):
                        BM[b,np.int(batch_coodinate[b,jj,1]), np.int(batch_coodinate[b,jj,0]),:] = build_P_prime[b,jj,:]
                else:
                    # t2 = time.time()
                    for l in range(200*200):
                        # print(build_P_prime[b,l,:])
                        # print(BM[b,batch_coodinate[b,l,1], batch_coodinate[b,l,0],:])
                        BM[b,np.int(batch_coodinate[b,l,1]), np.int(batch_coodinate[b,l,0]),:] = build_P_prime[b,l,:]
                    # print(time.time() - t2) 


            # # mask to dense coordinate
            # X = torch.linspace(0, 399, steps=400) 
            # Y = torch.linspace(0, 399, steps=400)
            # X, Y = torch.meshgrid(X, Y, indexing='xy')
            # standard_pts = torch.stack((X, Y), dim=0).repeat(batch_num,1,1,1).cuda()  # [b, 2, 400, 400]
            # selected_standard_pts = (trg_mask * standard_pts) #.reshape(160000,2)
            
            # trg_mask_pt = torch.empty((batch_num, 160000, 2)).cuda() # [b,N,2]


            # print("OK")
            # for b in range(batch_num):
            #     for (i,j) in zip(selected_standard_pts[b,0,:,:], selected_standard_pts[b,1,:,:]):
            #         i[(1 < x) & (x < 5)]
            #         for (x, y) in zip(i,j):
            #             if x==0 and y==0:
            #                 continue
            #             else:
            #                 trg_mask_pt = torch.vstack((trg_mask_pt, torch.stack((x, y)).unsqueeze(0)))
            #     # input: 
            #     # batch_src_pt：[4, 64, 2] 
            #     # batch_trg_pt：[4, 2, 8, 8]
            #     # trg_mask_pt：[N,2]
            #     build_P_prime = self.estimateTransformation.build_P_prime_for_d1(batch_src_pt[b], batch_trg_pt[b], trg_mask_pt[1:,:])          
            #     # out: build_P_prime (160000,2)
 



            # 局部采样
            # # build_P_prime: (b,980000,2)
            # # dense_pt_d1: (b,980000,2)
            # # build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.output_size[0], self.output_size[1], 2])  # build_P_prime.size(0) == mini-batch size,
            # # # output: [b, shrunken h, shrunken w, 2] [b, 198, 198, 2]
            # # BM = torch.zeros((batch_num, 992, 992, 2)).cuda().float()
            # BM = torch.full((batch_num, 150, 150, 2), float('nan')).cuda().float()
            # dense_pt_d1 = ((dense_pt_d1/2+0.5)*150).astype(int) # (b,980000,2)
            # dense_pt_d1 = np.clip(dense_pt_d1, 0, 149)
            # # print(np.max(dense_pt_d1))
            # # BM = torch.full((batch_num, 992, 992, 2), float('nan')).cuda().float()            
            # # dense_pt_d1 = ((dense_pt_d1/2+0.5)*992).astype(int) # (b,980000,2)
            # # dense_pt_d1 = np.clip(dense_pt_d1, 0, 991)
            
            # for batch in range(batch_num):
            #     for l in range(198*198):
            #         BM[batch,dense_pt_d1[batch,l,1],dense_pt_d1[batch,l,0],:] = build_P_prime[batch,l,:]

        
            # # BM[:,dense_pt_d1[:,:,1],dense_pt_d1[:,:,0],:] = build_P_prime[:,:,:]
            
            # # for batch in range(batch_num):
            # #     for l in range(992*992):
            # #         BM[batch,dense_pt_d1[batch,l,1],dense_pt_d1[batch,l,0],:] = build_P_prime[batch,l,:]
            
            # # 获得BM (b, 198, 198, 2)
            # BM = BM.transpose(2, 3).transpose(1, 2) # (b, 198, 198, 2) -> (b, 2, 198, 198)
            # BM = F.interpolate(BM, output_shape,  mode='bilinear', align_corners=True) # [b, 2, 198, 198]-->[b, 2, 992, 992] # 这里的插值函数仅支持NCHW的形式，故而需要手动转换
            # BM = BM.transpose(1, 2).transpose(2, 3) # [b, 2, 992, 992]-->[b, 992, 992, 2] 至此，获得了backward mapping, requires_grad=True
            
            # batch_I_r = F.grid_sample(batch_I, BM, padding_mode='zeros', align_corners=True) # [b, 3, 992, 992], requires_grad=True
            # batch_I_r = batch_I_r.nan_to_num() # 默认将nan转成0
            # return batch_I_r # output: [b, 3, h, w]   

            # resize small bm to big bm
            # build_P_prime_reshape = BM.reshape([build_P_prime.size(0), self.output_size[0], self.output_size[1], 2])  # build_P_prime.size(0) == mini-batch size,
            # # output: [b, shrunken h, shrunken w, 2]=[b, 400, 400, 2]
            build_P_prime_reshape = BM.transpose(2, 3).transpose(1, 2) # [b, 400, 400, 2]-->[b, 2, 400, 400]
            map = F.interpolate(build_P_prime_reshape, output_shape,  mode='bilinear', align_corners=True) # [b, 2, 400, 400]-->[b, 2, 992, 992] # 这里的插值函数仅支持NCHW的形式，故而需要手动转换
            map = map.transpose(1, 2).transpose(2, 3) # [b, 2, 992, 992]-->[b, 992, 992, 2] 至此，获得了backward mapping, requires_grad=True
            batch_I_r = F.grid_sample(batch_I, map, padding_mode='border', align_corners=True) # [b, 3, 992, 992], requires_grad=True
            return batch_I_r # output: [b, 3, h, w]     
        else:
            print("error in tps loss process")



class estimateTransformation(nn.Module):

    def __init__(self, F, output_size, device=torch.device('cuda:2')):
        super(estimateTransformation, self).__init__()
        self.eps = 1e-6 # 为了防止径向基函数中的log()为0，故而加入一个很小的数值
        self.I_r_height, self.I_r_width = output_size # shrunken (h, w) (198,198)
        self.F = F # 64
        self.device = device
        self.C = self._build_C(self.F) # (64,2)
        self.P = self._build_P(self.I_r_width, self.I_r_height) # (39204,2)
        # self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C), dtype=torch.float32, device=self.device)) 
        # self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P), dtype=torch.float32, device=self.device))
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C), dtype=torch.float32)) 
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P), dtype=torch.float32))
    def _build_d1_C(self, F, pts):
        '''
        pts=[b,2,8,8]
        '''
        batch_num = pts.shape[0]
        # batch_pt_norm = pts / 992 # [b, 2, 31, 31]
        # batch_pt_norm = batch_pt_norm.permute(0,3,2,1).reshape(batch_num,-1,2) # [b, 2, 31, 31] ->  [b, 31, 31, 2] -> [b, 961, 2] 
        # batch_pt_norm = (batch_pt_norm[:]-0.5)*2
        pts = pts.data.cpu().numpy().transpose(0, 2, 3, 1) # [b,31,31,2]
        # pts = pts[:,::3,::3,:] / [992, 992] # [b,31,31,2] -> [b,11,11,2]
        pts = pts / [992, 992] # [b,31,31,2] -> [b,11,11,2]
        pts = pts.transpose(0,2,1,3).reshape(batch_num,-1,2) # [b,11,11,2] -> [b,121,2] HWC->WHC->(WH)C
        pts = (pts[:]-0.5)*2 

        # pts = pts.permute(0,1,3,2) # [b, 2, 31, 31] ->  [b, 2, 31, 31]   BCHW -> BCWH
        # for batch in range(batch_num):
        #     for axis in range(2):
        #         pts[batch,axis,:,:] = (pts[batch,axis,:,:]-torch.min(pts[batch,axis,:,:]))  / (torch.max(pts[batch,axis,:,:])-torch.min(pts[batch,axis,:,:]) )
        #         pts[batch,axis,:,:] = (pts[batch,axis,:,:]-0.5)*2
        # pts = pts.permute(0,2,3,1)  # [b, 2, 31, 31] -> [b, 31, 31, 2]
        # pts = pts.reshape(batch_num, -1, 2) # (b,961,2)          
        # pts = pts.data.cpu().numpy()

        return pts  # (b,121,2) from [-1 to 1]

    def _build_inv_delta_d1_C(self, F, C): 
        '''
        inv_delta_C是基于964*964的大矩阵，这里不采用LU分解的方式求解，而是直接求逆矩阵。
        在[-1 to 1]闭区间内，构造参考点阵列，并写成逆矩阵形式。
        C:(b,64,2)
        '''
        hat_C = np.zeros((C.shape[0],F, F), dtype=float)  # b x F x F
        for m in range(C.shape[0]):
            for i in range(0, F):
                for j in range(i, F):
                    r = np.linalg.norm(C[m,i] - C[m,j])
                    hat_C[m, i, j] = r
                    hat_C[m, j, i] = r # 以对角线为对称轴，镜像填充
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
        return inv_delta_C  # b x F+3 x F+3 # (b,124,124)

    def _build_C(self, F):
        '''
        F:64
        构造归一化到[-1 to 1]的参考点矩阵，一共64个点的坐标，闭区间
        '''
        im_x, im_y = np.mgrid[-1:1:complex(np.sqrt(F)), -1:1:complex(np.sqrt(F))]
        # C = np.stack((im_x,im_y), axis=2).reshape(-1,2) # (961,2),列序优先，且先w(x)后h(y)
        C = np.stack((im_y,im_x), axis=2).reshape(-1,2) # (121,2),行序优先，且先w(x)后h(y)
        return C  # (121,2) from [-1 to 1] 闭区间，边界值可以取到

    def _build_inv_delta_C(self, F, C): 
        '''
        inv_delta_C是基于67*67的大矩阵，这里不采用LU分解的方式求解，而是直接求逆矩阵。
        在[-1 to 1]闭区间内，构造参考点阵列，并写成逆矩阵形式。
        C：(64,2)
        F=64
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
        '''
        I_r_width = I_r_height = 198
        '''
        # 20220903更改为闭区间
        # I_r_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width (320,)
        # I_r_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height (320,)
        I_r_x = np.linspace(-1, 1, I_r_width)   # self.I_r_width (320,)
        I_r_y = np.linspace(-1, 1, I_r_height)  # self.I_r_height (320,)        
        
        I_r_grid_x, I_r_grid_y = np.meshgrid(I_r_x, I_r_y) # , indexing='xy'
        # 最开始的版本
        # P = np.stack(np.meshgrid(I_r_x, I_r_y),axis=2).reshape(-1, 2) # (320,320,2) -> (102400,2)
        P = np.stack((I_r_grid_x,I_r_grid_y),axis=2).reshape(-1, 2) # (320,320,2) -> (102400,2) # 行序优先
        # P = np.stack((I_r_grid_y,I_r_grid_x),axis=2).reshape(-1, 2) # (320,320,2) -> (102400,2) # 列序优先
        return P   # (30204,2)

    def _build_P_hat(self, F, C, P):
        '''
        P_hat也是大矩阵，是根据参考点和P
        C: (64,2) from [-1 to 1] 行序优先
        P: (39204,2) from [-1 to 1] 行序优先
        F = 64
        '''
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height) # 39204
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2  ==(39204,1,2) tile in (1,121,1) = (39204, 121, 2)
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2 # (121,2)->(1,121,2)
        P_diff = P_tile - C_tile  # nxFx2 #(39204, 121, 2)-(1,121,2)=(39204, 121, 2)
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = 2 * np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3  # 39204 * 124


    def _build_d1_P(self, I_r_width, I_r_height, pts):
        '''
        pts: (b,2,31,31)
        '''
        # 局部采样法
        batch_num = pts.shape[0]     
        pts = pts.permute(0,1,3,2) # [b, 2, 31, 31] ->  [b, 2, 31, 31]   BCHW -> BCWH
        pts = F.interpolate(pts, size=(198,198), mode='bilinear', align_corners=True) # [b, 2, 31, 31] -> [b, 2, 600, 600]
        # for batch in range(batch_num):
        #     pts[batch,:,:,:] = pts[batch,:,:,:] / 992
        #     pts[batch,:,:,:] = (pts[batch,:,:,:]-0.5)*2

        pts = pts / 992
        pts = (pts-0.5)*2
        pts = pts.permute(0,2,3,1)  # [b, 2, 31, 31] -> [b, 31, 31, 2]
        pts = pts.reshape(batch_num, -1, 2) # (b,39204,2)          
        batch_pts = pts.data.cpu().numpy()
        
        # 全图采样法
        # batch_num = pts.shape[0]
        # I_r_x = np.linspace(-1, 1, I_r_width)   # self.I_r_width (198,)
        # I_r_y = np.linspace(-1, 1, I_r_height)  # self.I_r_height (198,)        
        # I_r_grid_x, I_r_grid_y = np.meshgrid(I_r_x, I_r_y, indexing='xy') # indexing='xy'表示行序优先
        # batch_pts = np.stack((I_r_grid_x,I_r_grid_y),axis=2).reshape(-1, 2) # (320,320,2) -> (39204,2) # 行序优先
        # batch_pts = np.tile(np.expand_dims(batch_pts, axis=0), (batch_num, 1, 1)) 


        return batch_pts   # (b,39204,2)

    def _build_P_d1_hat(self, F, C, P):
        '''
        P_hat也是大矩阵，是根据参考点和P
        F=64
        d1C: (b,961,2) 
        d1P: (b,160000,2) numpy
        '''
        P = P / 200
        P = (P-0.5)*2 

        batch_num = P.shape[0]
        n = P.shape[1]  # n (= self.I_r_width x self.I_r_height) # 39204
        P_tile = np.tile(np.expand_dims(P, axis=2), (1, 1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2  ==(b,39204,1,2) tile in (1,1,961,1) = (b, 39204, 961, 2)
        C_tile = np.tile(np.expand_dims(C, axis=1), (1, n, 1, 1)) # 1 x F x 2 # (b,961,2)->(b,1,961,2)->(b,39204,961,2)
        P_diff = np.empty_like(P_tile)
        for j in range(batch_num):
            P_diff[j] = P_tile[j] - C_tile[j]  # nxFx2 # (b, 39204, 961, 2) - (b,1,961,2) = (b, 39204, 961, 2)
        rbf_norm = np.empty((batch_num,n,F))
        rbf = np.empty((batch_num,n,F))
        P_hat = np.empty((batch_num,n,F+3))
        for i in range(batch_num):
            rbf_norm[i] = np.linalg.norm(P_diff[i], ord=2, axis=2, keepdims=False)  # b x n x F
            rbf[i] = 2 * np.multiply(np.square(rbf_norm[i]), np.log(rbf_norm[i] + self.eps))  # b x n x F
            # 行序优先
            P_hat[i] = np.concatenate([np.ones((n, 1)), P[i], rbf[i]], axis=1)
            # 列序优先
            # P_hat[i] = np.concatenate([rbf[i], np.ones((n, 1)), P[i]], axis=1)
        return P_hat  # b x n x F+3  # b * 980000 * 964

    # 主函数，构造backward mapping
    def build_P_prime(self, batch_C_prime):
        batch_size = batch_C_prime.size(0) # batch_C_prime= (1,64,2),是warped图像上的控制点位置（已归一化到(-1,1)）
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(batch_size, 3, 2).float().cuda()), dim=1)  # batch_size x F+3 x 2 
        # 实现 (1,64,2)+(1,3,2) in dim1= (1,67,2)，获得真实图像上的控制点
        batch_T = torch.matmul(self.inv_delta_C.repeat(batch_size,1,1), batch_C_prime_with_zeros)  # batch_size x F+3 x 2 # [b, 67, 67]*[b,67,2] = [b, 124, 2]
        # 与求逆后的大矩阵相乘，获得参数转移矩阵T
        
        batch_P_prime = torch.matmul(self.P_hat.repeat(batch_size,1,1), batch_T)  # batch_size x n x 2  # [39204, 124]*[b, 124, 2]=[1, 39204, 2] torch.Size([2, 39204, 2])
        return batch_P_prime  # batch_size x n x 2 

    # 主函数2，构造backward mapping
    def build_P_prime_for_d1(self, batch_C_prime, pt_d1, trg_dense_pt):
        '''
        pt_d1: (b,2,8,8),是已知的target的pred,尚未归一化, p2
        batch_C_prime: [b, 64, 2] 归一化的source pred
        trg_dense_pt: (b,160000,2) numpy,未归一化的密集点坐标
        '''
        batch_size = batch_C_prime.size(0) # batch_C_prime= (1,64,2),是warped图像上的控制点位置（已归一化到(-1,1)）
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(batch_size, 3, 2).float().cuda()), dim=1)  # batch_size x F+3 x 2 
        # 实现 (1,121,2)+(1,3,2) in dim1= (1,124,2)，获得真实图像上的控制点
        self.D1C = self._build_d1_C(self.F, pt_d1) # input:(b,2,8,8) output:(b,64,2)
        self.inv_delta_d1C = torch.tensor(self._build_inv_delta_d1_C(self.F, self.D1C), dtype=torch.float32).cuda()
        batch_T = torch.matmul(self.inv_delta_d1C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2 # [b, 124, 124]*[b,124,2] = [b, 124, 2]
        # 与求逆后的大矩阵相乘，获得参数转移矩阵T
        
        # self.D1P = self._build_d1_P(self.I_r_width, self.I_r_height, pt_d1) #加密的点 (b, 39204, 2)
        self.D1P = trg_dense_pt #加密的点 (b, 160000, 2)
        self.d1P_hat = torch.tensor(self._build_P_d1_hat(self.F, self.D1C, self.D1P), dtype=torch.float32).cuda() # out: [2,160000, 67]
        batch_P_prime = torch.matmul(self.d1P_hat, batch_T)  # batch_size x n x 2  # [2,39204, 67]*[2, 67, 2]= [2, 39204, 2]

        # dense_pt_d1 = self.D1P
        return batch_P_prime # batch_size x n x 2  # [2, 39204, 2])
        # return batch_P_prime, dense_pt_d1 # batch_size x n x 2  # [2, 39204, 2])