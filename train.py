'''
2022/9/6
Weiguang Zhang
'''
import os, sys
import argparse
from tkinter.messagebox import NO
import time
import re
from pathlib import Path
# FILE = Path(__file__).resolve() #获取绝对路径位置
# ROOT = FILE.parents[0] #获取当前目录位置
workdir=os.getcwd()
from network import model_handlebar, DilatedResnet
import utilsV4 as utils
from utilsV4 import AverageMeter
from dataset_lmdb import my_unified_dataset
from loss import Losses
import torch.nn.functional as F
import torch

def train(args):
    ''' setup path '''
    data_path = str(args.data_path_train)

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

    ''' load model '''
    n_classes = 2
    model = model_handlebar(n_classes=n_classes, num_filter=32, architecture=DilatedResnet, BatchNorm='BN', in_channels=3)
    
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

    if args.is_DDP is True:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        model = model.to(args.local_rank)
        print('The main device is in: ',next(model.parameters()).device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif args.parallel is not None:
        model = model.to(args.device) # model.cuda(args.device) or model.cuda() # 模型统一移动到第一块GPU上。需要注意的是，对于模型（nn.module）来说，返回值值并非是必须的，而对于数据（tensor）来说，务必需要返回值。此处选择了比较保守的，带有返回值的统一写法
        model = torch.nn.DataParallel(model, device_ids=device_ids_visual_list) # device_ids定义了并行模式下，模型可以运行的多台机器，该函数不光支持多GPU，同时也支持多CPU，因此需要model.to()来指定具体的设备。
        print('The main device is in: ',next(model.parameters()).device)
    else:
        args.device = torch.device('cpu')
        model=model.to(args.device) 
        print('The main device is in: ',next(model.parameters()).device)


    ''' load optimizer '''
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, weight_decay=1e-12)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate, weight_decay=1e-10, amsgrad=True)
    else:
        raise Exception(args.optimizer,'<-- please choice optimizer')
    # LR Scheduler 
    # sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 9, 150, 200], gamma=0.5)

    ''' load checkpoint '''
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            if args.parallel is not None:
                checkpoint = torch.load(args.resume, map_location=args.device) 
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
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

    
    ''' load loss '''
    loss_instance = Losses(reduction='mean', args_gpu=args.device) # loss类的实例化
    loss_fun = loss_instance.loss_fn4_v5_r_4   # 调用其中一个loss function
    # loss_fun = loss_instance.loss_fn4_v5_r_3   # *
    loss_fun2 = loss_instance.loss_fn_l1_loss #普通的L1 loss，未被用到
    losses = AverageMeter() # 用于计数和计算平均loss
    
    loss_instance.lambda_loss = 1 # 主约束
    loss_instance.lambda_loss_a = 0.1 # 邻域约束
    loss_instance.lambda_loss_b = 0.001
    loss_instance.lambda_loss_c = 0.01

    ''' load data, dataloader'''
    FlatImg = utils.FlatImg(args = args, out_path=out_path, date=date, date_time=date_time, _re_date=_re_date, dataset=my_unified_dataset, \
                            data_path = data_path, \
                            model = model, optimizer = optimizer) 

    trainloader,train_sampler = FlatImg.loadTrainData(data_split='train', is_DDP = args.is_DDP)
    trainloader_len = len(trainloader)
    print("Total number of mini-batch in each epoch: ", trainloader_len)
    
    if args.schema == 'train':
        train_time = AverageMeter()
        for epoch in range(epoch_start, args.n_epoch):
            print('* lambda_loss :'+str(loss_instance.lambda_loss)+'\t'+'learning_rate :'+str(optimizer.param_groups[0]['lr']))
            print('* lambda_loss :'+str(loss_instance.lambda_loss)+'\t'+'learning_rate :'+str(optimizer.param_groups[0]['lr']), file=reslut_file)
            loss_l1_list = 0
            loss_local_list = 0
            loss_edge_list = 0
            loss_rectangles_list = 0
            loss_list = []

            begin_train = time.time() #从这里正式开始训练当前epoch
            train_sampler.set_epoch(epoch) if args.is_DDP else None
            model.train()
            for i, (images1, labels1, images2, labels2, w_im, d_im, ref_pt) in enumerate(trainloader):
                # print("get",images1.size())
                images1 = images1.cuda() # 后面康康要不要改成to，不知道会不会影响并行
                labels1 = labels1.cuda()
                images2 = images2.cuda()
                labels2 = labels2.cuda() 
                w_im = w_im.cuda()
                d_im = d_im.cuda()
                ref_pt = ref_pt.cuda()                 

                optimizer.zero_grad()
                outputs1, outputs2 = model(images1, labels1, images2, labels2, w_im, d_im, ref_pt)
                # outputs1和outputs2分别是是D1和D2的控制点坐标信息，先w(x),后h(y)，范围是（992,992）

                loss1_l1, loss1_local, loss1_edge, loss1_rectangles = loss_fun(outputs1, labels1)
                loss2_l1, loss2_local, loss2_edge, loss2_rectangles = loss_fun(outputs2, labels2)
                
                # # fourier dewarp part
                # map = get_bm(output3, ref_pt)
                # w_im=fourier(w_im)
                # d_im=fourier(d_im)
                # flatten_w_im=F.grid_sample(w_im, map, padding_mode='border', align_corners=True)
                # loss3 = loss_instance.fourier_loss_a*loss_fun2(flatten_w_im,d_im)

                loss1 = loss1_l1 + loss1_local*loss_instance.lambda_loss_a + loss1_edge*loss_instance.lambda_loss_b + loss1_rectangles*loss_instance.lambda_loss_c
                loss2 = loss2_l1 + loss2_local*loss_instance.lambda_loss_a + loss2_edge*loss_instance.lambda_loss_b + loss2_rectangles*loss_instance.lambda_loss_c
                loss = loss1 + loss2


                losses.update(loss.item()) # 自定义实例，用于计算平均值
                loss.backward()
                optimizer.step()



                # 累加相邻的20个mini-batch中各项损失
                loss_list.append(loss.item())
                loss_l1_list += loss1_l1.item()
                loss_local_list += loss1_local.item()
                # loss_edge_list += loss_edge.item()
                # loss_rectangles_list += loss_rectangles.item()
                
                # 每隔60个mini-batch显示一次loss，或者当当前epoch训练结束时
                if (i + 1) % args.print_freq == 0 or (i + 1) == trainloader_len:
                    list_len = len(loss_list) #print_freq

                    print('[{0}][{1}/{2}]\t\t'
                          '[min{3:.2f} avg{4:.4f} max{5:.2f}]\t'
                          '[l1:{6:.4f} l:{7:.4f} e:{8:.4f} r:{9:.4f}]\t'
                          '{loss.avg:.4f}'.format(
                        epoch + 1, i + 1, trainloader_len,
                        min(loss_list), sum(loss_list) / list_len, max(loss_list),
                        loss_l1_list / list_len, loss_local_list / list_len, loss_edge_list / list_len, loss_rectangles_list / list_len,
                        loss=losses))
                    
                    print('[{0}][{1}/{2}]\t\t'
                          '[{3:.2f} {4:.4f} {5:.2f}]\t'
                          '[l1:{6:.4f} l:{7:.4f} e:{8:.4f} r:{9:.4f}]\t'
                          '{loss.avg:.4f}'.format(
                        epoch + 1, i + 1, trainloader_len,
                        min(loss_list), sum(loss_list) / list_len, max(loss_list),
                        loss_l1_list / list_len, loss_local_list / list_len, loss_edge_list / list_len, loss_rectangles_list / list_len,
                        loss=losses), file=reslut_file)
                    # 清零累计的loss
                    del loss_list[:]
                    # loss_interval_list = 0
                    loss_l1_list = 0
                    loss_local_list = 0
                    # loss_edge_list = 0
                    # loss_rectangles_list = 0
            FlatImg.saveModel_epoch(epoch, model, optimizer)     # FlatImg.saveModel(epoch, save_path=path)
            trian_t = time.time()-begin_train  #从这里宣布结束训练当前epoch
            losses.reset()
            train_time.update(trian_t)
            print("Current epoch training elapsed time:{:.2f} minutes ".format(trian_t/60))
            print("Total epoches training elapsed time:{:.2f} minutes ".format(train_time.sum/60) )
            scheduler.step()




    m, s = divmod(train_time.sum, 60)
    h, m = divmod(m, 60)
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s))
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s), file=reslut_file)

    reslut_file.close()

# train(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='DDCP',
                        help='Architecture')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                        help='# of the epochs')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimization')

    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')  # print frequency


    # './synthesis_code'   './dataset/WarpDoc'  './dataset/warp0.lmdb' './dataset/train.lmdb'
    parser.add_argument('--data_path_train', default='./dataset/train.lmdb', type=str,
                        help='the path of train images.')  # train image path

    parser.add_argument('--data_path_test', default='./dataset/testset', type=str, help='the path of test images.')

    parser.add_argument('--output-path', default='./flat/', type=str, help='the path is used to  save output --img or result.') 

    
    parser.add_argument('--batch_size', nargs='?', type=int, default=2,
                        help='Batch Size')#28   
    
    parser.add_argument('--resume', default=None, type=str, 
                        help='Path to previous saved model to restart from')    

    parser.add_argument('--schema', type=str, default='train',
                        help='train or test')       # train  validate
    
    parser.add_argument('--is_DDP', default=True, type=bool,
                        help='whether to use DDP')


    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    
    # parser.set_defaults(resume='/Data_HDD/fmp23_weiguang_zhang/DDCP2/flat/2022-09-13/2022-09-13 16:11:15/10/2022-09-13 16:11:15DDCP.pkl')
    parser.add_argument('--parallel', default='23', type=list,
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
        os.makedirs(out_path)

    train(args)
