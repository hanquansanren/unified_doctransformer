'''
2022/11/4
Weiguang Zhang
V9 means final polar-doc
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
from utilmy.points import Polygon

class Losses(object):
    def __init__(self, reduction='mean', args_gpu=0):
        # self.reduction = reduction
        # self.args_gpu = args_gpu
        # self.kernel_r = torch.ones(2, 1, 3, 3).cuda()
        self.kernel_cross_17 = torch.tensor(
                                [[[[0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0]]],
                                 [[[0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0]]]],dtype=torch.float32).cuda()
        # self.kernel = torch.tensor([[[[0, 1., 0],
        #                               [1., 1., 1.],
        #                               [0, 1., 0]]],
        #                             [[[0, 1., 0],
        #                               [1., 1., 1.],
        #                               [0, 1., 0]]]]).cuda()
        # self.keep_position = torch.tensor([[[[1, 1, 1],
        #                                     [0, 0, 0],
        #                                     [-1, -1, -1]]],
        #                                     [[[1, 0, -1],
        #                                     [1., 0, -1],
        #                                     [1, 0, -1]]],]).cuda()

        # # self.kernel = torch.ones(2, 1, 3, 3).cuda(self.args_gpu)
        # self.kernel_2_1 = torch.tensor([[[[1.], [-1.]]], [[[1.], [-1.]]]]).cuda()
        # self.kernel_1_2 = torch.tensor([[[[1., -1.]]], [[[1., -1.]]]]).cuda()
        # # self.lambda_ = 0.1
        # self.lambda_ = 0.5

        # self.matrices_2 = torch.full((1024, 960), 2, dtype=torch.float).cuda()
        # self.matrices_0 = torch.full((1024, 960), 0, dtype=torch.float).cuda()

        # self.fiducial_point_gaps = [2, 3, 4, 5, 6, 10, 12, 15, 20, 30]
        # self.fiducial_point_gaps_v2 = [2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
        
        self.lambda_loss = 1
        self.lambda_loss_a = 1
        self.fourier_loss_a = 1
        self.fourier_loss_b = 1
        self.fourier_loss_c = 1


    def loss_fn4_v5_r_4(self, input, target, reduction='mean'):
        i_t = target - input
        '''one smooth_l1_loss'''
        loss_l1 = F.smooth_l1_loss(input, target, reduction=reduction)

        '''two differential coordinates'''
        loss_local = torch.mean(torch.pow(F.conv2d(F.pad(i_t, (4, 4, 4, 4), mode='replicate'), self.kernel_cross_17, padding=0, groups=2) - i_t*17, 2))

        return loss_l1, loss_local, 0, 0
        # return loss_l1, loss_local, loss_edge, loss_rectangle

    def polar_iou_loss(self, input, target, reduction='mean'):
        '''
        input  : (2b,2,31,31)
        target : (2b,2,31,31)
        '''
        # input[:,:,15,15] = target[:,:,15,15]
        # print(input.shape)
        label_center = target[:,:,15,15].unsqueeze(-1).unsqueeze(-1).repeat(1,1,31,31)
        # print(label_center.shape)
        pred_dist = torch.sqrt(torch.sum((input - label_center)**2, dim= 1))
        label_dist = torch.sqrt(torch.sum((target - label_center)**2, dim= 1))
        # pred_dist[:,15,15] = pred_dist[:,15,15]+1e-06
        # label_dist[:,15,15] = label_dist[:,15,15]+1e-06
        # print(pred_dist.shape, label_dist.shape) # [20, 31, 31] [20, 31, 31]
        
        l_max = torch.maximum(pred_dist, label_dist) # [20, 31, 31]
        l_min = torch.minimum(pred_dist, label_dist) # [20, 31, 31]

        piou = (l_min.sum(dim=[1,2])/ l_max.sum(dim=[1,2]))
        # loss = -piou.log()
        loss = 50*3000*((1-piou)**2)*(-piou.log())
        # loss1 = (l_max.sum(dim=[1,2])/ l_min.sum(dim=[1,2])).log()
        
        
        loss = loss.mean()
        # loss1 = loss1.mean()
        # print(loss.shape)

        ##################################
        ######### perimeter ##############
        ##################################
        # print(input[:,:,0,0:31].shape)
        left = Polygon(input[:,:,0,0:31])
        right = Polygon(input[:,:,30,0:31])
        top = Polygon(input[:,:,0:31,0])
        bottom = Polygon(input[:,:,0:31,30]) 
        left_gt = Polygon(target[:,:,0,0:31])
        right_gt = Polygon(target[:,:,30,0:31])
        top_gt = Polygon(target[:,:,0:31,0])
        bottom_gt = Polygon(target[:,:,0:31,30])
        l_loss = F.smooth_l1_loss(left.get_perimeter(), left_gt.get_perimeter(), reduction=reduction)/992
        r_loss = F.smooth_l1_loss(right.get_perimeter(), right_gt.get_perimeter(), reduction=reduction)/992
        t_loss = F.smooth_l1_loss(top.get_perimeter(), top_gt.get_perimeter(), reduction=reduction)/992
        b_loss = F.smooth_l1_loss(bottom.get_perimeter(), bottom_gt.get_perimeter(), reduction=reduction)/992
        lrtb_loss = 200*(l_loss + r_loss + t_loss + b_loss)

        
        # print(loss, lrtb_loss)
        # print(loss, avg_l_loss, avg_r_loss, avg_t_loss, avg_b_loss)
        return loss, lrtb_loss
        # return loss

    def local_polar_iou_loss(self, input, target, reduction='mean'):
        '''
        input  : (2b,2,31,31)
        target : (2b,2,31,31)
        '''
        input1 = F.unfold(input, kernel_size=3, dilation=1, stride=1, padding=0)
        target1 = F.unfold(target, kernel_size=3, dilation=1, stride=1, padding=0)

        print(input1.shape) # [b, 18, 841]
        vinput = torch.cat((input1[:,None,0:9,:],input1[:,None,9:18,:]),1)# [b, 2, 9, 841]
        vtarget = torch.cat((target1[:,None,0:9,:],target1[:,None,9:18,:]),1)# [b, 2, 9, 841]
        print(vinput.shape, vtarget.shape)
        # label_center = vinput[:,:,4,:] # [b, 2, 841]
        label_center = vtarget[:,:,4,:].unsqueeze(-2).repeat(1,1,9,1) # [b, 2, 9, 841]

        pred_dist = torch.sqrt(torch.sum((vinput - label_center)**2, dim= 1)) # [b, 9, 841]
        label_dist = torch.sqrt(torch.sum((vtarget - label_center)**2, dim= 1)) # [b, 9, 841]
        # pred_dist[:,15,15] = pred_dist[:,15,15]+1e-06
        # label_dist[:,15,15] = label_dist[:,15,15]+1e-06
        # print(pred_dist.shape, label_dist.shape) # [20, 31, 31] [20, 31, 31]
        
        l_max = torch.maximum(pred_dist, label_dist) # [b, 9, 841]
        l_min = torch.minimum(pred_dist, label_dist) # [b, 9, 841]

        piou = (l_min.sum(dim=1)/ l_max.sum(dim=1)) # [b, 841]
        loss = ((1-piou)**2)*(-piou.log()) # [b, 841]
        loss = loss.sum(dim=1)
        loss = loss.mean()

        return loss
        # return loss


    # def mask_calculator(self, lbl, single_line):
    #     '''
    #     lbl:(b,31,31)
    #     single_line: 30
    #     '''
    #     batch_num = lbl.shape[0]
    #     pt_edge_batch = torch.zeros((batch_num,15,4*single_line), requires_grad=True).cuda()
    #     for batch in range(batch_num):
    #         for c in range(15): # c is contour
    #             pt_edge_batch[batch,c,0:(single_line-2*c+1)] = lbl[batch,c,c:(single_line+1)-c]
    #             for num in range(1,single_line-2*c,1):
    #                 pt_edge_batch[batch,c,single_line-2*c+num]= lbl[batch,num+c,single_line-c]
                
    #             # a = lbl[batch,single_line-c,c:single_line+1-c]
    #             pt_edge_batch[batch,c,(2*(single_line-2*c)):(3*(single_line-2*c)+1)] = lbl[batch,single_line-c,c:single_line+1-c]
    #             for num in range((single_line-1-2*c),0,-1):
    #                 pt_edge_batch[batch,c,(3*(single_line-2*c)+num)]= lbl[batch,(single_line-num-c),c]


    #     # pts = pt_edge_batch.int()
    #     return pt_edge_batch # (b,15,120)

    # def mask_calculator_l(self, lbl, single_line):
    #     '''
    #     lbl:(b,31,31)
    #     single_line: 30
    #     '''
    #     batch_num = lbl.shape[0]
    #     pt_edge_batch = torch.zeros((batch_num,15,4*single_line)).cuda()
    #     for batch in range(batch_num):
    #         for c in range(15): # c is contour
    #             pt_edge_batch[batch,c,0:(single_line-2*c+1)] = lbl[batch,c,c:(single_line+1)-c]
    #             for num in range(1,single_line-2*c,1):
    #                 pt_edge_batch[batch,c,single_line-2*c+num]= lbl[batch,num+c,single_line-c]
                
    #             # a = lbl[batch,single_line-c,c:single_line+1-c]
    #             pt_edge_batch[batch,c,(2*(single_line-2*c)):(3*(single_line-2*c)+1)] = lbl[batch,single_line-c,c:single_line+1-c]
    #             for num in range((single_line-1-2*c),0,-1):
    #                 pt_edge_batch[batch,c,(3*(single_line-2*c)+num)]= lbl[batch,(single_line-num-c),c]


    #             # pts = pt_edge_batch.int()
    #     return pt_edge_batch # (b,15,120)


    # def centerness_loss(self, input, target, reduction='mean'):
    #     '''
    #     input  : (2b,2,31,31)
    #     target : (2b,2,31,31)
    #     '''
    #     batch=input.shape[0]
    #     label_center = target[:,:,15,15].unsqueeze(-1).unsqueeze(-1).repeat(1,1,31,31)
    #     # print(label_center.shape)
    #     pred_dist = torch.sqrt(torch.sum((input - label_center)**2, dim= 1))  # [20, 31, 31]
    #     label_dist = torch.sqrt(torch.sum((target - label_center)**2, dim= 1)) # [20, 31, 31] 
    #     new_label_dist = self.mask_calculator_l(label_dist, 30) # (2b,15,120)
    #     new_pred_dist = self.mask_calculator(pred_dist, 30) # (2b,15,120)
    #     centerness_targets = torch.zeros((batch,15)).cuda()
    #     centerness_pred = torch.zeros((batch,15), requires_grad=True).cuda() 
    #     for b in range(batch):
    #         for c in range(15):
    #             centerness_targets[b,c] = torch.sqrt(new_label_dist[b,c,0:120-c*8].min()/new_label_dist[b,c,0:120-c*8].max())
    #             centerness_pred[b,c] = torch.sqrt(new_pred_dist[b,c,0:120-c*8].min()/new_pred_dist[b,c,0:120-c*8].max())
                
    #     # centerness_targets = ( new_label_dist.min(dim=-1) / new_label_dist.max(dim=-1) ).mean(dim=-1) # (2b)
    #     # centerness_targets = torch.sqrt(centerness_targets)
    #     # centerness_pred = ( new_pred_dist.mean(dim=-1) / new_pred_dist.max(dim=-1).mean(dim=-1) )
    #     # centerness_pred = torch.sqrt(centerness_pred)
    #     # loss1 = F.binary_cross_entropy_with_logits(centerness_targets,centerness_targets, reduction=reduction)
    #     # print(loss1)
    #     loss = F.binary_cross_entropy_with_logits(centerness_targets,centerness_pred, reduction=reduction)

    #     return loss

    def loss_fn_l1_loss(self, input, target, mask=None, reduction='mean'):
        '''
        three: mask L1 loss
        '''
        if mask is None:
            return F.l1_loss(input, target, reduction=reduction)
        elif mask is not None:
            return F.l1_loss(input[mask], target[mask], reduction=reduction)   
        else:
            print('error mask')        











    # def loss_fn_cood_position_loss(self, input):
    #     '''
    #     input : (3*b,2,8,8)
    #     '''
    #     # print(input[0])
    #     # print(torch.maximum(F.conv2d(F.pad(input, (1, 1, 1, 1), mode='replicate'), self.keep_position, padding=0, groups=2),torch.zeros((30,2,8,8)).cuda()))
    #     loss_position = torch.mean(100*torch.maximum(F.conv2d(F.pad(input, (1, 1, 1, 1), mode='replicate'), self.keep_position, padding=0, groups=2),torch.zeros((30,2,8,8)).cuda()))
        
    #     return loss_position


    # def loss_fn4_v5(self, input, target, size_average=False):
    #     n, c, h, w = input.size()

    #     # n_ = n*c*h*w

    #     i_t = target - input

    #     '''one'''
    #     loss_l1 = F.smooth_l1_loss(input, target, size_average=size_average)

    #     '''two'''
    #     loss_local = torch.mean(torch.pow(F.conv2d(F.pad(i_t, (1, 1, 1, 1), mode='replicate'), self.kernel, padding=0, groups=2) - i_t*5, 2))

    #     '''three   --weak'''
    #     loss_edge_a = F.mse_loss(input[:, :, 0, :], target[:, :, 0, :], size_average=size_average)
    #     loss_edge_b = F.mse_loss(input[:, :, h-1, :], target[:, :, h-1, :], size_average=size_average)
    #     loss_edge_c = F.mse_loss(input[:, :, :, 0], target[:, :, :, 0], size_average=size_average)
    #     loss_edge_d = F.mse_loss(input[:, :, :, w-1], target[:, :, :, w-1], size_average=size_average)
    #     loss_edge = loss_edge_a+loss_edge_b+loss_edge_c+loss_edge_d

    #     '''four'''
    #     loss_rectangle = self.line_cross(input, target, size_average)


    #     return loss_l1, loss_local, loss_edge, loss_rectangle


    # def line_cross(self, input, target, size_average=False):

    #     input_rectangles_h = F.conv2d(input, self.kernel_1_2, padding=0, groups=2)
    #     target_rectangles_h = F.conv2d(target, self.kernel_1_2, padding=0, groups=2)
    #     input_arget_rectangles_h = F.mse_loss(input_rectangles_h, target_rectangles_h, size_average=size_average)
    #     input_rectangles_o = F.conv2d(input, self.kernel_2_1, padding=0, groups=2)
    #     target_rectangles_o = F.conv2d(target, self.kernel_2_1, padding=0, groups=2)
    #     input_arget_rectangles_o = F.mse_loss(input_rectangles_o, target_rectangles_o, size_average=size_average)
    #     loss_rectangles = input_arget_rectangles_h + input_arget_rectangles_o

    #     return loss_rectangles

    # def loss_line_cross(self, input, target, size_average=False):
    #     i_t = target - input

    #     loss_local = torch.mean(torch.pow(F.conv2d(F.pad(i_t, (1, 1, 1, 1), mode='replicate'), self.kernel, padding=0, groups=2) - i_t*5, 2))
    #     # loss_local = torch.mean(torch.abs(F.conv2d(i_t, self.kernel, padding=1, groups=2) - i_t*5))
    #     # loss_local = torch.mean(torch.pow(F.conv2d(i_t, self.kernel, padding=1, groups=2) - i_t*5, 2))

    #     input_rectangles_h = F.conv2d(input, self.kernel_1_2, padding=0, groups=2)
    #     target_rectangles_h = F.conv2d(target, self.kernel_1_2, padding=0, groups=2)
    #     input_arget_rectangles_h = F.mse_loss(input_rectangles_h, target_rectangles_h, size_average=size_average)
    #     input_rectangles_o = F.conv2d(input, self.kernel_2_1, padding=0, groups=2)
    #     target_rectangles_o = F.conv2d(target, self.kernel_2_1, padding=0, groups=2)
    #     input_arget_rectangles_o = F.mse_loss(input_rectangles_o, target_rectangles_o, size_average=size_average)
    #     loss_rectangles = input_arget_rectangles_h + input_arget_rectangles_o

    #     return loss_local, loss_rectangles

    # def loss_line_cross_l1(self, input, target, size_average=False):
    #     i_t = target - input

    #     loss_local = torch.mean(torch.abs(F.conv2d(F.pad(i_t, (1, 1, 1, 1), mode='replicate'), self.kernel, padding=0, groups=2) - i_t*5))
    #     # loss_local = torch.mean(torch.abs(F.conv2d(i_t, self.kernel, padding=1, groups=2) - i_t*5))
    #     # loss_local = torch.mean(torch.pow(F.conv2d(i_t, self.kernel, padding=1, groups=2) - i_t*5, 2))

    #     input_rectangles_h = F.conv2d(input, self.kernel_1_2, padding=0, groups=2)
    #     target_rectangles_h = F.conv2d(target, self.kernel_1_2, padding=0, groups=2)
    #     input_arget_rectangles_h = F.l1_loss(input_rectangles_h, target_rectangles_h, size_average=size_average)
    #     input_rectangles_o = F.conv2d(input, self.kernel_2_1, padding=0, groups=2)
    #     target_rectangles_o = F.conv2d(target, self.kernel_2_1, padding=0, groups=2)
    #     input_arget_rectangles_o = F.l1_loss(input_rectangles_o, target_rectangles_o, size_average=size_average)
    #     loss_rectangles = input_arget_rectangles_h + input_arget_rectangles_o

    #     return loss_local, loss_rectangles
