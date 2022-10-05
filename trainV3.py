'''
2022/9/6
Weiguang Zhang
V3 means DDCP+FDRNet
'''
import os, sys
import argparse
from tkinter.messagebox import NO
import time
import re
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
workdir=os.getcwd()
from torch.utils.tensorboard import SummaryWriter

import utilsV4 as utils
from utilsV4 import AverageMeter
from dataset_lmdbV3 import my_unified_dataset
from loss import Losses
import torch.nn.functional as F
import torch
from torch.utils import data
import random
import torchvision
from os.path import join as pjoin
from intermediate_dewarp_loss import get_dewarped_intermediate_result

def show_wc_tnsboard(global_step,writer,images,labels, pred, grid_samples,inp_tag, gt_tag, pred_tag):
    '''
    images: [16, 3, 992, 992]
    labels: [16, 2, 31, 31]
    pred:   [16, 2, 31, 31]
    grid_samples = 8
    inp_tag = 'Train Pred1 pts'
    gt_tag = 'Train WCs'
    pred_tag = 'none'
    '''
    idxs=torch.LongTensor(random.sample(range(min(grid_samples,images.shape[0])), min(grid_samples,images.shape[0])))
    # idx=tensor([2, 3, 5, 1, 4, 0])
    ax_list=[]
    for idx in idxs:
        fig, ax = plt.subplots(figsize = (9.92,9.92),facecolor='white')
        ax.imshow(np.transpose(images[idx].cpu().int(), [1, 2, 0]))
        if labels is not None:
            ax.scatter(np.transpose(labels[idx].cpu(), [1, 2, 0])[:,:,0].flatten(), np.transpose(labels[idx].cpu(), [1, 2, 0])[:,:,1].flatten(),s=1.2,c='red',alpha=1)
        ax.scatter(np.transpose(pred[idx].detach().cpu(), [1, 2, 0])[:,:,0].flatten(), np.transpose(pred[idx].detach().cpu(), [1, 2, 0])[:,:,1].flatten(),s=1.2,c='green',alpha=1)
        ax.axis('off')
        plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
        plt.savefig('./point_vis{}.png'.format(idx))
        fig.canvas.draw()
        fig_str = fig.canvas.tostring_rgb()
        data = np.frombuffer(fig_str, dtype=np.uint8).reshape((int(992), -1, 3)).astype(np.uint8)
        ax_list.append(data)
        plt.close()

    np2tensor=torch.from_numpy(np.array(ax_list).transpose(0, 3, 1, 2)) # ([6, 3, 992, 992])
    grid_inp = torchvision.utils.make_grid(np2tensor[idxs], normalize=False, scale_each=True , padding=0) # 拼图
    # output: tensor[3, 992, 5952]
    writer.add_image(inp_tag, grid_inp, global_step)
    
def location_mark(img, location, color=(0, 0, 255)):
    stepSize = 0
    for l in location.astype(np.int64).reshape(-1, 2):
        cv2.circle(img,
                    (l[0] + math.ceil(stepSize / 2), l[1] + math.ceil(stepSize / 2)), 3, color, -1)
    return img


def train(args):
    ''' log writer '''
    global _re_date
    if args.resume is not None:
        re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
        _re_date = re_date.search(str(args.resume)).group(0)
        reslut_file = open(out_path + '/' + date + date_time + ' @' + _re_date + '_' + args.arch + '.log', 'w')
    else:
        _re_date = None
        reslut_file = open(out_path+'/'+date+date_time+'_'+args.arch+'.log', 'w')
    print(args)
    print(args, file=reslut_file)
    print('log file has been written')

    ''' load device '''
    device_ids_real = list(map(int, args.parallel)) # ['2','3'] -> [2,3] 字符串批量转整形，再转生成器，再转数组 
    all_devices=''
    for num,i in enumerate(device_ids_real): # [2,3] -> ‘2,3’
        all_devices=all_devices+str(i) 
        all_devices+=',' if num<(len(device_ids_real)-1) else ''
    print("Using real gpu:{:>6s}".format(all_devices)) # print(f"using gpu:{all_devices:>6s}") 
    os.environ["CUDA_VISIBLE_DEVICES"] = all_devices
    
    n_gpu = torch.cuda.device_count()
    print('Total number of visible gpu:', n_gpu)
    device_ids_visual_list=list(range(n_gpu)) # 已经重新映射为从0开始的列表
    args.device = torch.device('cuda:'+str(device_ids_visual_list[0])) # 选择列表的第一块GPU，用于存放模型参数，模型的结构    

    ''' load model '''
    from networkV3 import model_handlebar, DilatedResnet, DilatedResnet_for_test_single_image
    n_classes = 2
    model = model_handlebar(n_classes=n_classes, num_filter=32, architecture=DilatedResnet, BatchNorm='BN', in_channels=3)
    model.double()
    # model_val = model_handlebar(n_classes=n_classes, num_filter=32, architecture=DilatedResnet_for_test_single_image, BatchNorm='BN', in_channels=3)
    # model_val.double()
    tps_for_loss = get_dewarped_intermediate_result()
    tps_for_loss.double()



    if args.is_DDP is True:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        model = model.to(args.local_rank)
        print('The main device is in: ',next(model.parameters()).device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif args.parallel is not None:
        model = model.to(args.device) # model.cuda(args.device) or model.cuda() # 模型统一移动到第一块GPU上。需要注意的是，对于模型（nn.module）来说，返回值值并非是必须的，而对于数据（tensor）来说，务必需要返回值。此处选择了比较保守的，带有返回值的统一写法
        model = torch.nn.DataParallel(model, device_ids=device_ids_visual_list) # device_ids定义了并行模式下，模型可以运行的多台机器，该函数不光支持多GPU，同时也支持多CPU，因此需要model.to()来指定具体的设备。
        tps_for_loss = tps_for_loss.to(args.device)
        tps_for_loss = torch.nn.DataParallel(tps_for_loss, device_ids=device_ids_visual_list)
        print('The main device is in: ',next(model.parameters()).device)
    else:
        args.device = torch.device('cpu')
        model=model.to(args.device) 
        print('The main device is in: ',next(model.parameters()).device)
    # model_val.cuda()

    ''' load optimizer '''
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, weight_decay=1e-12)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate, weight_decay=1e-10, amsgrad=True)
    else:
        raise Exception(args.optimizer,'<-- please choice optimizer')
    # LR Scheduler 
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40, 90, 150, 200], gamma=0.5)

    ''' load checkpoint '''
    if args.resume is not None:
        if os.path.isfile(args.resume) and args.is_DDP==True:
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            if args.parallel is not None:
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu')) 
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                scheduler.load_state_dict(checkpoint['scheduler_state'])
                # print(next(model.parameters()).device)
            else:
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
                model_parameter_dick = {}
                for k in checkpoint['model_state']:
                    model_parameter_dick[k.replace('module.', '')] = checkpoint['model_state'][k]
                model.load_state_dict(model_parameter_dick)
                # print(next(model.parameters()).device)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        elif os.path.isfile(args.resume) and args.is_DDP==False:
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            if args.parallel is not None:
                checkpoint = torch.load(args.resume, map_location=args.device) 
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                # for param_group in optimizer.param_groups:
                #     param_group["lr"] = args.l_rate 
                # scheduler.load_state_dict(checkpoint['scheduler_state'])
                # print(next(model.parameters()).device)
            else:
                checkpoint = torch.load(args.resume, map_location=args.device)
                '''cpu'''
                model_parameter_dick = {}
                for k in checkpoint['model_state']:
                    model_parameter_dick[k.replace('module.', '')] = checkpoint['model_state'][k]
                model.load_state_dict(model_parameter_dick)
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                # print(next(model.parameters()).device)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))            
        else:
            print("No checkpoint found at '{}'".format(args.resume.name))
    # for name, param in model.named_parameters():
    #     print(name, param.size())
    epoch_start = checkpoint['epoch'] if args.resume is not None else 0
    # epoch_start=0
    
    ''' load loss '''
    loss_instance = Losses(reduction='mean', args_gpu=args.device) # loss类的实例化
    loss_fun = loss_instance.loss_fn4_v5_r_4   # 调用其中一个loss function
    # loss_fun = loss_instance.loss_fn4_v5_r_3   # *
    loss_fun2 = loss_instance.loss_fn_l1_loss #普通的L1 loss，未被用到
    losses = AverageMeter() # 用于计数和计算平均loss
    
    loss_instance.lambda_loss = 1 # 主约束
    loss_instance.lambda_loss_a = 0.1 # 邻域约束

    ''' load data, dataloader'''
    FlatImg = utils.FlatImg(args = args, out_path=out_path, date=date, date_time=date_time, _re_date=_re_date, dataset=my_unified_dataset, \
                            data_path = args.data_path_train, data_path_test=args.data_path_test,\
                            model = model, model_validation=None,\
                            optimizer = optimizer, reslut_file=reslut_file) 
    FlatImg.loadTestData()
    lmdb_list = utils.get_total_lmdb(args.data_path_total)
    trainloader_list = []
    for k in lmdb_list:
        full_data_path = pjoin(args.data_path_total, k)
        trainloader_list.append(FlatImg.loadTrainData('train', full_data_path, is_DDP = args.is_DDP))
    print(lmdb_list)
    trainloader_len = len(trainloader_list[0][0]) # 这里有两个[0][0]是因为list中存的是(trainloader, train_sampler)
    print("Total number of mini-batch in each epoch: ", trainloader_len)
    
    '''load tensorboard'''
    writer = SummaryWriter(comment='train')

    if args.schema == 'train':
        train_time = AverageMeter()
        global_step=0
        for epoch in range(epoch_start, args.n_epoch):
            print('* lambda_loss :'+str(loss_instance.lambda_loss)+'\t'+'learning_rate :'+str(optimizer.param_groups[0]['lr']))
            print('* lambda_loss :'+str(loss_instance.lambda_loss)+'\t'+'learning_rate :'+str(optimizer.param_groups[0]['lr']), file=reslut_file)
            loss_list = []
            loss_l1_list, loss_local_list, loss3_list, loss4_list, loss5_list, loss6_list, loss7_list = 0,0,0,0,0,0,0
            
            begin_train = time.time() #从这里正式开始训练当前epoch
            if args.is_DDP:
                trainloader_list[0][1].set_epoch(epoch)
                print("shuffle successfully")
            model.train()
            for i, (images1, labels1, images2, labels2, w_im, d_im, mask1, mask2) in enumerate(trainloader_list[epoch%1][0]):
                # print("get",images1.size()) # [32,3,992,992]
                images1 = images1.cuda() # 后面康康要不要改成to，不知道会不会影响并行
                labels1 = labels1.cuda()
                images2 = images2.cuda()
                labels2 = labels2.cuda()
                w_im = w_im.cuda()
                d_im = d_im.cuda() 
                mask1 = mask1.cuda()
                mask2 = mask2.cuda()   

                triple_images = torch.cat((images1, images2, w_im), 0) # [3b, 3, 992, 992]
                double_labels = torch.cat((labels1, labels2), 0) # [2b, 2, 31, 31]

                optimizer.zero_grad()
                triple_outputs = model(triple_images) # (3*args.batch_size, 2, 31, 31) # [18, 2, 31, 31]
                # outputs1,outputs2,output3分别是是D1和D2以及wild的控制点坐标信息，先w(x),后h(y)，范围是（992,992）
                # losses calculation
   
                # fourier dewarp for part3 and part4
                rectified_img3, ref_img3 = tps_for_loss(w_im, d_im, triple_outputs[2*args.batch_size:3*args.batch_size])
                # input: w_im: [2, 3, 992, 992] d_im:[2, 3, 992, 992] pred: [2, 2, 31, 31]
                # output:rectified_img3 [2, 3, 992, 992];  ref_img3 [2, 3, 992, 992]

                rectified_img4, ref_img4 = tps_for_loss(w_im, triple_images[0:args.batch_size], triple_outputs[2*args.batch_size:], ptd1=labels1)
                rectified_img5, ref_img5 = tps_for_loss(w_im, triple_images[args.batch_size:2*args.batch_size], triple_outputs[2*args.batch_size:], ptd1=labels2)
                rectified_img6, ref_img6 = tps_for_loss(images1, images2, triple_outputs[0:args.batch_size], ptd1=labels2)
                # rectified_img6, ref_img6 = tps_for_loss(images1, images2, labels1, ptd1=labels2)
                rectified_img7, ref_img7 = tps_for_loss(images2, images1, triple_outputs[args.batch_size:2*args.batch_size], ptd1=labels1)
                loss1_l1, loss1_local, loss1_edge, loss1_rectangles = loss_fun(triple_outputs[0:2*args.batch_size], double_labels)
                
                loss1 = loss1_l1 + (loss1_local)*loss_instance.lambda_loss_a
                loss3 = loss_fun2(rectified_img3, ref_img3)
                loss4 = loss_fun2(mask1*rectified_img4, mask1*ref_img4)
                loss5 = loss_fun2(mask2*rectified_img5, mask2*ref_img5)
                loss6 = loss_fun2(mask2*rectified_img6, mask2*ref_img6)
                loss7 = loss_fun2(mask1*rectified_img7, mask1*ref_img7)
                loss = loss1 + (loss3 + loss4 + loss5) + 0.5*(loss6 + loss7)
                # loss = loss1 + (loss3 + loss4 + loss5)
                '''vis for fourier dewarp'''

                t1=time.time()
                # flat_show = d_im[0].detach().cpu().numpy().transpose(1,2,0) # NCHW-> NHWC (h, w, 3), dtype('float64')
                # flat_show = flat_show.astype(np.uint8) # dtype('float64') -> dtype('uint8')
                # cv2.imwrite('./mark_origin_scanned.png', flat_show)
                # w1_show = w_im[0].detach().cpu().numpy().transpose(1,2,0) # NCHW-> NHWC (h, w, 3), dtype('float64')
                # w1_show = w1_show.astype(np.uint8) # dtype('float64') -> dtype('uint8')
                # cv2.imwrite('./mark_origin_wild.png', w1_show)

                d1_show = images2[0].detach().cpu().numpy().transpose(1,2,0) # NCHW-> NHWC (h, w, 3), dtype('float64')
                d1_show = d1_show.astype(np.uint8) # dtype('float64') -> dtype('uint8')
                cv2.imwrite('./mark_origin_target.png', d1_show)
                in_show = images1[0].detach().cpu().numpy().transpose(1,2,0) # NCHW-> NHWC (h, w, 3), dtype('float64')
                in_show = in_show.astype(np.uint8) # dtype('float64') -> dtype('uint8')
                cv2.imwrite('./mark_origin_input.png', in_show)
                triple_outputs_show = triple_outputs[0*args.batch_size:1*args.batch_size].detach().data.cpu().numpy().transpose(0, 2, 3, 1)[0] # (31,31,2)
                perturbed_img_mark = location_mark(in_show.copy(), triple_outputs_show, (0, 0, 255))
                cv2.imwrite('./mark_origin_input_pred.png', perturbed_img_mark)

                # in_lbl_show = labels1.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
                # lbl_img_mark = location_mark(in_show.copy(), in_lbl_show, (0, 255, 0))
                # cv2.imwrite('./mark_origin_input_lbl.png', lbl_img_mark)

                flatten_img = mask2[0]*rectified_img6[0]
                flatten_img_show = flatten_img.detach().cpu().numpy().transpose(1,2,0) # NCHW-> NHWC (h, w, 3), dtype('float64')
                flatten_img_show = flatten_img_show.astype(np.uint8) # dtype('float64') -> dtype('uint8')
                cv2.imwrite('./mark_rectified_input.png', flatten_img_show)
                refe4_img = mask2[0]*ref_img6[0]
                refe4_img_show = refe4_img.detach().cpu().numpy().transpose(1,2,0) # NCHW-> NHWC (h, w, 3), dtype('float64')
                refe4_img_show = refe4_img_show.astype(np.uint8) # dtype('float64') -> dtype('uint8')
                cv2.imwrite('./mark_ref_target.png', refe4_img_show)
                print(time.time()-t1)

                loss.backward()
                optimizer.step()

                losses.update(loss.item()) # 自定义实例，用于计算平均值
                loss_list.append(loss.item())

                loss_l1_list +=    (loss1_l1.item()*1)
                loss_local_list += (loss1_local.item()*loss_instance.lambda_loss_a)
                loss3_list += (loss3.item()*1)
                loss4_list += (loss4.item()*1)
                loss5_list += (loss5.item()*1)
                loss6_list += (loss6.item()*0.5)
                loss7_list += (loss7.item()*0.5)
                global_step+=1
                if (global_step-1)%25==0:
                    show_wc_tnsboard(global_step, writer, images1, labels1, triple_outputs[0:args.batch_size], 8,'Train d1 pts', 'no', 'no')
                    # show_wc_tnsboard(global_step, writer, images2, labels2, triple_outputs[args.batch_size:2*args.batch_size], 8,'Train d2 pts', 'no', 'no')
                    show_wc_tnsboard(global_step, writer, w_im, None, triple_outputs[2*args.batch_size:3*args.batch_size], 8,'Train wild pts', 'no', 'no')
                    writer.add_scalar('L1 Loss/train', loss_l1_list/(i+1), global_step)
                    writer.add_scalar('local Loss/train', loss_local_list/(i+1), global_step)
                    writer.add_scalar('loss3 /train', loss3_list/(i+1), global_step)
                    writer.add_scalar('loss4 /train', loss4_list/(i+1), global_step)
                    writer.add_scalar('loss5 /train', loss4_list/(i+1), global_step)
                    writer.add_scalar('loss6 /train', loss6_list/(i+1), global_step)
                    writer.add_scalar('loss7 /train', loss6_list/(i+1), global_step)
                    writer.add_scalar('total Loss/train', losses.avg, global_step)


                
                # 每隔print_freq个mini-batch显示一次loss，或者当当前epoch训练结束时
                if (i + 1) % args.print_freq == 0 or (i + 1) == trainloader_len:
                    list_len = len(loss_list) #print_freq

                    print('[{0}][{1}/{2}]'
                        '[min{3:.2f} avg{4:.4f}]'
                        '[l1:{5:.3f} l:{6:.3f} 3:{7:.3f} 4:{8:.3f} 5:{9:.3f}'
                        ' 6:{10:.3f} 7:{11:.3f}] {loss.avg:.3f}'.format(
                        epoch + 1, i + 1, trainloader_len,
                        min(loss_list), sum(loss_list) / list_len,
                        loss_l1_list / list_len, loss_local_list / list_len, loss3_list / list_len, loss4_list / list_len, loss5_list / list_len, loss6_list/ list_len, loss7_list/ list_len,
                        loss=losses))
                    
                    print('[{0}][{1}/{2}]'
                        '[{3:.2f} {4:.4f}]'
                        '[l1:{5:.3f} l:{6:.3f} 3:{7:.3f} 4:{8:.3f} 5:{9:.3f}'
                        ' 6:{10:.3f} 7:{11:.3f}] {loss.avg:.3f}'.format(
                        epoch + 1, i + 1, trainloader_len,
                        min(loss_list), sum(loss_list) / list_len,
                        loss_l1_list / list_len, loss_local_list / list_len, loss3_list / list_len, loss4_list / list_len, loss5_list / list_len, loss6_list/ list_len, loss7_list/ list_len,
                        loss=losses), file=reslut_file)
                    # 清零累计的loss 
                    # del loss_list[:] 
                    # loss_l1_list = 0 
                    # loss_local_list = 0
            FlatImg.saveModel_epoch(epoch, model, optimizer, scheduler)     # FlatImg.saveModel(epoch, save_path=path)
            trian_t = time.time()-begin_train  #从这里宣布结束训练当前epoch
            losses.reset()
            train_time.update(trian_t)
            print("Current epoch training elapsed time:{:.2f} minutes ".format(trian_t/60))
            print("Total epoches training elapsed time:{:.2f} minutes ".format(train_time.sum/60) )
            scheduler.step(metrics=min(loss_list))

    m, s = divmod(train_time.sum, 60)
    h, m = divmod(m, 60)
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s))
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s), file=reslut_file)

    reslut_file.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='DDCP',
                        help='Architecture')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=1000,
                        help='# of the epochs')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimization')

    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')  # print frequency


    # './synthesis_code'   './dataset/WarpDoc'  './dataset/warp0.lmdb' './dataset/train.lmdb' 
    parser.add_argument('--data_path_train', default='./dataset/warp1.lmdb', type=str,
                        help='the path of train images.')  # train image path

    # './dataset_for_debug'  './dataset'  './dataset_fast_train' './dataset/biglmdb'
    parser.add_argument('--data_path_total', default='./dataset_fast_train', type=str,
                        help='the path of train images.')

    parser.add_argument('--data_path_test', default='./dataset/testset/mytest0', type=str, help='the path of test images.')

    parser.add_argument('--output-path', default='./flat/', type=str, help='the path is used to  save output --img or result.') 

    
    parser.add_argument('--batch_size', nargs='?', type=int, default=4,
                        help='Batch Size') # 8   
    
    parser.add_argument('--resume', default=None, type=str, 
                        help='Path to previous saved model to restart from')    

    parser.add_argument('--schema', type=str, default='train',
                        help='train or test')       # train  validate
    
    parser.add_argument('--is_DDP', default=False, type=bool,
                        help='whether to use DDP')


    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    
    # ICDAR
    # parser.set_defaults(resume='/Public/FMP_temp/fmp23_weiguang_zhang/DDCP2/ICDAR2021/2021-02-03 16_15_55/143/2021-02-03 16_15_55flat_img_by_fiducial_points-fiducial1024_v1.pkl')
    # parser.set_defaults(resume='/Public/FMP_temp/fmp23_weiguang_zhang/DDCP2/flat/2022-09-20/2022-09-20 14:42:26 @2021-02-03/144/2021-02-03@2022-09-20 14:42:26DDCP.pkl')
    # parser.set_defaults(resume='/Public/FMP_temp/fmp23_weiguang_zhang/DDCP2/flat/2022-09-20/2022-09-20 16:40:40/15/2022-09-20 16:40:40DDCP.pkl')
    # big 
    parser.set_defaults(resume='/Public/FMP_temp/fmp23_weiguang_zhang/DDCP2/flat/2022-09-28/2022-09-28 17:04:41/89/2022-09-28 17:04:41DDCP.pkl')
    
    parser.add_argument('--parallel', default='023', type=list,
                        help='choice the gpu id for parallel ')
                        
    args = parser.parse_args()

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise Exception(args.resume+' -- not exist')

    if args.data_path_test is None:
        raise Exception('-- No test path')
    else:
        if not os.path.exists(args.data_path_test):
            raise Exception(args.data_path_test+' -- no find')

    global out_path, date, date_time
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    date_time = time.strftime(' %H:%M:%S', time.localtime(time.time()))
    out_path = os.path.join(args.output_path, date)

    if not os.path.exists(out_path):
        os.makedirs(out_path,exist_ok=True)

    train(args)
