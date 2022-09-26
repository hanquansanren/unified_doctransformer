import os, sys
import argparse
from tkinter.messagebox import NO
import time
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
workdir=os.getcwd()
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import torchvision



def show_wc_tnsboard(global_step,writer,images,labels, pred, grid_samples,inp_tag, gt_tag, pred_tag):
    '''
    images: [6, 3, 992, 992]
    labels: [6, 2, 31, 31]
    grid_samples = 8
    inp_tag = 'Train Inputs'
    gt_tag = 'Train WCs'
    pred_tag = 'Train Pred1 pts'
    '''
    idxs=torch.LongTensor(random.sample(range(images.shape[0]), min(grid_samples,images.shape[0])))
    # idx=tensor([2, 3, 5, 1, 4, 0])
    ax_list=[]
    for idx in idxs:
        fig, ax = plt.subplots(figsize = (9.92,9.92),facecolor='white')
        ax.imshow(np.transpose(images[idx].cpu().int(), [1, 2, 0]))
        ax.scatter(np.transpose(labels[idx].cpu(), [1, 2, 0])[:,:,0].flatten(), np.transpose(labels[idx].cpu(), [1, 2, 0])[:,:,1].flatten(),s=1.2,c='red',alpha=1)
        ax.axis('off')
        plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
        plt.savefig('./point_vis{}.png'.format(idx))
        fig.canvas.draw()
        fig_str = fig.canvas.tostring_rgb()
        data = np.frombuffer(fig_str, dtype=np.uint8).reshape((int(992), -1, 3)).astype(np.uint8)
        ax_list.append(data)
        plt.close()


    aaa=torch.from_numpy(np.array(ax_list).transpose(0, 3, 1, 2)) # ([6, 3, 992, 992])
    grid_inp = torchvision.utils.make_grid(aaa[idxs], normalize=False, scale_each=True , padding=0) # 拼图
    # output: tensor[3, 992, 5952]
    writer.add_image(inp_tag, grid_inp, global_step)
    
    # # [6, 2, 31, 31]->[2,186,31]
    # # cated_labels = torch.cat((labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]), 1).cpu()
    # # [6, 2, 31, 31]->[6,2,31,31] xy ->yx
    # labels = labels.permute((0, 1, 3, 2))
    grid_lbl = torchvision.utils.make_grid(labels[idxs],normalize=False, scale_each=True,padding=0)
    # # output: tensor[2, 31, 186]
    # grid_lbl = grid_lbl.permute(0,2,1).cpu()
    # writer.add_image(gt_tag, grid_lbl, global_step)
    # # output: tensor[2, 186, 31]
       
    # plt.figure(figsize = (60, 10), facecolor='white')
    # plt.imshow(np.transpose(grid_inp.cpu(), [1, 2, 0]))
    # plt.scatter(np.transpose(grid_lbl, [1, 2, 0])[:,:,0].flatten(), np.transpose(grid_lbl, [1, 2, 0])[:,:,1].flatten(),s=1.2,c='red',alpha=1)
    # plt.axis('off')
    # plt.margins(0,0)
    # plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
    # plt.savefig('./point_vis{}.png'.format(6969))
    # writer.add_figure('my_figure_batch', plt.gcf(), global_step)


    # # grid_pred = torchvision.utils.make_grid(pred[idxs],normalize=True, scale_each=True)
    # # writer.add_image(pred_tag, grid_pred, global_step)



show_wc_tnsboard(global_step, writer, images1, labels1, outputs1, 8,'Train Inputs', 'Train WCs', 'Train Pred1 pts')
writer.add_scalar('L1 Loss/train', loss_l1_list/(i+1), global_step)
writer.add_scalar('local Loss/train', loss_local_list/(i+1), global_step)