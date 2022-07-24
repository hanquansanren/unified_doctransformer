'''
2021/2/3

Guowang Xie

args:
    n_epoch:epoch values for training
    optimizer:various optimization algorithms
    l_rate:initial learning rate
    resume:the path of trained model parameter after
    data_path_train:datasets path for training
    data_path_validate:datasets path for validating
    data_path_test:datasets path for testing
    output-path:output path
    batch_size:
    schema:test or train
    parallel:number of gpus used, like 0, or, 0123

'''
import os, sys
import argparse
from tkinter.messagebox import NO
from torch.autograd import Variable
import warnings
import time
import re
from pathlib import Path
# FILE = Path(__file__).resolve() #获取绝对路径位置
# ROOT = FILE.parents[0] #获取当前目录位置
workdir=os.getcwd()

from network import FiducialPoints, DilatedResnetForFlatByFiducialPointsS2

# import utilsV3 as utils
import utilsV4 as utils

from dataloader import PerturbedDatastsForFiducialPoints_pickle_color_v2_v2

from loss import Losses

def train(args):
    ''' setup path '''
    data_path = str(args.data_path_train)+'/'
    # data_path_validate = str(args.data_path_validate)+'/'
    data_path_test = str(args.data_path_test)+'/'

    ''' log writer '''
    global _re_date
    if args.resume is not None:
        re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
        _re_date = re_date.search(str(args.resume)).group(0)
        reslut_file = open(path + '/' + date + date_time + ' @' + _re_date + '_' + args.arch + '.log', 'w')
    else:
        _re_date = None
        reslut_file = open(path+'/'+date+date_time+'_'+args.arch+'.log', 'w')
    print(args)
    print(args, file=reslut_file)
    print('log file has been written')

    ''' load model '''
    n_classes = 2
    model = FiducialPoints(n_classes=n_classes, num_filter=32, architecture=DilatedResnetForFlatByFiducialPointsS2, BatchNorm='BN', in_channels=3)     #
    
    ''' load device '''
    if args.parallel is not None:
        device_ids_real = list(map(int, args.parallel)) # ['2','3'] -> [2,3] 字符串批量转整形，再转生成器，再转数组 
        all_devices=''
        for num,i in enumerate(device_ids_real): # [2,3] -> ‘2,3’
            all_devices=all_devices+str(i) 
            all_devices+=',' if num<(len(device_ids_real)-1) else ''
        print("Using real gpu:{:>6s}".format(all_devices)) # print(f"using gpu:{all_devices:>6s}") 
        os.environ["CUDA_VISIBLE_DEVICES"] = all_devices 
        
        import torch
        n_gpu = torch.cuda.device_count()
        print('Total number of visible gpu:', n_gpu)
        device_ids_visual=range(torch.cuda.device_count())
        args.device = torch.device('cuda:'+str(device_ids_visual[0])) # 选择被选中的第一块GPU

        model = model.to(args.device) # model.cuda(args.device) or model.cuda() # 模型统一移动到第一块GPU上。需要注意的是，对于模型（nn.module）来说，返回值值并非是必须的，而对于数据（tensor）来说，务必需要返回值。此处选择了比较保守的，带有返回值的统一写法
        model = torch.nn.DataParallel(model, device_ids=device_ids_visual) # device_ids定义了并行模式下，模型可以运行的多台机器，该函数不光支持多GPU，同时也支持多CPU，因此需要model.to()来指定具体的设备。
        print('The main device is in: ',next(model.parameters()).device)
    # elif args.distributed:
    #     model.cuda()
    #     model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        import torch
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
    
    ''' load checkpoint '''
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            if args.parallel is not None:
                checkpoint = torch.load(args.resume, map_location=args.device) 
                model.load_state_dict(checkpoint['model_state'])
                # print(next(model.parameters()).device)
            else:
                checkpoint = torch.load(args.resume, map_location=args.device)
                '''cpu'''
                model_parameter_dick = {}
                for k in checkpoint['model_state']:
                    model_parameter_dick[k.replace('module.', '')] = checkpoint['model_state'][k]
                model.load_state_dict(model_parameter_dick)
                # print(next(model.parameters()).device)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume.name))
    epoch_start = checkpoint['epoch'] if args.resume is not None else 0


    ''' load loss '''
    loss_fun_classes = Losses(reduction='mean', args_gpu=args.device)
    loss_fun = loss_fun_classes.loss_fn4_v5_r_4   # *
    # loss_fun = loss_fun_classes.loss_fn4_v5_r_3   # *
    loss_fun2 = loss_fun_classes.loss_fn_l1_loss #普通的L1 loss
    
    ''' load data '''
    FlatImg = utils.FlatImg(args=args, path=path, date=date, date_time=date_time, _re_date=_re_date, model=model, \
                            log_file=reslut_file, n_classes=n_classes, optimizer=optimizer, \
                            loss_fn=loss_fun, loss_fn2=loss_fun2, data_loader=PerturbedDatastsForFiducialPoints_pickle_color_v2_v2, \
                            data_path=data_path, data_path_validate=None, data_path_test=data_path_test) 
    
    trainloader = FlatImg.loadTrainData(data_split='train', is_shuffle=True)
    trainloader_len = len(trainloader)
    print("Total number of mini-batch in each epoch: ", trainloader_len)
    
    train_time = AverageMeter()
    losses = AverageMeter()
    FlatImg.lambda_loss = 1 # 主约束
    FlatImg.lambda_loss_interval = 0.01 # 平面图的XY间隔距离
    FlatImg.lambda_loss_a = 0.1 # 邻域约束
    FlatImg.lambda_loss_b = 0.001
    FlatImg.lambda_loss_c = 0.01


    if args.schema == 'train':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(FlatImg.optimizer, milestones=[40, 90, 150, 200], gamma=0.5)
        for epoch in range(epoch_start, args.n_epoch):
            print('* lambda_loss :'+str(FlatImg.lambda_loss)+'\t'+'learning_rate :'+str(optimizer.param_groups[0]['lr']))
            print('* lambda_loss :'+str(FlatImg.lambda_loss)+'\t'+'learning_rate :'+str(optimizer.param_groups[0]['lr']), file=reslut_file)
            loss_interval_list = 0
            loss_l1_list = 0
            loss_local_list = 0
            loss_edge_list = 0
            loss_rectangles_list = 0
            loss_list = []

            begin_train = time.time() #从这里正式开始训练当前epoch
            model.train()
            # feed several mini-batches in each loop
            for i, (images, labels, interval) in enumerate(trainloader):
                images = images.cuda()
                labels = labels.cuda()
                interval = interval.cuda()

                optimizer.zero_grad()
                outputs, outputs_interval = FlatImg.model(images, is_softmax=False)

                loss_l1, loss_local, loss_edge, loss_rectangles = loss_fun(outputs, labels)
                loss_interval = loss_fun2(outputs_interval, interval)
                loss = FlatImg.lambda_loss*(loss_l1 + loss_local*FlatImg.lambda_loss_a + loss_edge*FlatImg.lambda_loss_b + loss_rectangles*FlatImg.lambda_loss_c) + FlatImg.lambda_loss_interval*loss_interval

                losses.update(loss.item())
                loss.backward()
                optimizer.step()

                # 累加相邻的20个mini-batch中各项损失
                loss_list.append(loss.item())
                loss_interval_list += loss_interval.item()
                loss_l1_list += loss_l1.item()
                loss_local_list += loss_local.item()
                # loss_edge_list += loss_edge.item()
                # loss_rectangles_list += loss_rectangles.item()
                
                # 每隔60个mini-batch显示一次loss，或者当当前epoch训练结束时
                if (i + 1) % args.print_freq == 0 or (i + 1) == trainloader_len:
                    list_len = len(loss_list) #print_freq

                    print('[{0}][{1}/{2}]\t\t'
                          '[min{3:.2f} avg{4:.4f} max{5:.2f}]\t'
                          '[l1:{6:.4f} l:{7:.4f} e:{8:.4f} r:{9:.4f} s:{10:.4f}]\t'
                          '{loss.avg:.4f}'.format(
                        epoch + 1, i + 1, trainloader_len,
                        min(loss_list), sum(loss_list) / list_len, max(loss_list),
                        loss_l1_list / list_len, loss_local_list / list_len, loss_edge_list / list_len, loss_rectangles_list / list_len, loss_interval_list / list_len,
                        loss=losses))
                    
                    print('[{0}][{1}/{2}]\t\t'
                          '[{3:.2f} {4:.4f} {5:.2f}]\t'
                          '[l1:{6:.4f} l:{7:.4f} e:{8:.4f} r:{9:.4f} s:{10:.4f}]\t'
                          '{loss.avg:.4f}'.format(
                        epoch + 1, i + 1, trainloader_len,
                        min(loss_list), sum(loss_list) / list_len, max(loss_list),
                        loss_l1_list / list_len, loss_local_list / list_len, loss_edge_list / list_len, loss_rectangles_list / list_len, loss_interval_list / list_len,
                        loss=losses), file=reslut_file)
                    # 清零累计的loss
                    del loss_list[:]
                    loss_interval_list = 0
                    loss_l1_list = 0
                    loss_local_list = 0
                    loss_edge_list = 0
                    loss_rectangles_list = 0
            FlatImg.saveModel_epoch(epoch)     # FlatImg.saveModel(epoch, save_path=path)
            trian_t = time.time()-begin_train  #从这里宣布结束训练当前epoch
            losses.reset()
            train_time.update(trian_t)
            print("Current epoch training elapsed time: ",trian_t)
            print("Total epoches training elapsed time: ", train_time.sum)
            
            # model.eval()

            scheduler.step()




    m, s = divmod(train_time.sum, 60)
    h, m = divmod(m, 60)
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s))
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s), file=reslut_file)

    reslut_file.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='Document-Dewarping-with-Control-Points',
                        help='Architecture')

    parser.add_argument('--img_shrink', nargs='?', type=int, default=None,
                        help='short edge of the input image')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=300,
                        help='# of the epochs')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimization')

    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')  # print frequency



    parser.add_argument('--data_path_train', default='./dataset/WarpDoc', type=str,
                        help='the path of train images.')  # train image path

    # parser.add_argument('--data_path_validate', default=ROOT / 'dataset/fiducial1024/fiducial1024/fiducial1024_v1/validate/', type=str,
    #                     help='the path of validate images.')  # validate image path

    parser.add_argument('--data_path_test', default='./dataset/testset', type=str, help='the path of test images.')

    parser.add_argument('--output-path', default='./flat/', type=str, help='the path is used to  save output --img or result.') 

    
    
    parser.add_argument('--resume', default=None, type=str, 
                        help='Path to previous saved model to restart from')    

    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')#28

    parser.add_argument('--schema', type=str, default='train',
                        help='train or test')       # train  validate

    
    parser.set_defaults(resume='/Data_HDD/fmp23_weiguang_zhang/DDCP/flat/2022-06-27/2022-06-27 16:30:17 @2022-06-06/104/2022-06-06@2022-06-27 16:30:17Document-Dewarping-with-Control-Points.pkl')

    parser.add_argument('--parallel', default='0123', type=list,
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

    global path, date, date_time
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    date_time = time.strftime(' %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(args.output_path, date)

    if not os.path.exists(path):
        os.makedirs(path)

    train(args)
