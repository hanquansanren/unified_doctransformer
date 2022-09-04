'''
2022/6/26
Weiguang Zhang

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
    parallel:number of gpus used, like '0', or, '0123'

'''
import os, sys
import argparse


import time
import re
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

from network import FiducialPoints, DilatedResnetForFlatByFiducialPointsS2

# import utilsV3 as utils
import utilsV4 as utils

from dataset import my_unified_dataset


def test(args):
    ''' setup path '''
    data_path = str(args.data_path_train)+'/'
    data_path_validate = str(args.data_path_validate)+'/'
    data_path_test = str(args.data_path_test)+'/'
    
    ''' log writer '''
    global _re_date
    if args.resume is not None:
        re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
        _re_date = re_date.search(args.resume).group(0)
        log_file = open(path + '/' + date + date_time + ' @' + _re_date + '_' + args.arch + '.log', 'w+')
    else:
        _re_date = None
        log_file = open(path+'/'+date+date_time+'_'+args.arch+'.log', 'w+')

    print(args)
    print(args, file=log_file) # 编写日志功能，我们可以通过改变该参数使print()函数输出到特定的文件中。
    print('log file has been written')


    ''' load model '''
    n_classes = 2
    model = FiducialPoints(n_classes=n_classes, num_filter=32, architecture=DilatedResnetForFlatByFiducialPointsS2, BatchNorm='BN', in_channels=3)     #
    
    ''' load device '''
    if args.parallel is not None: #
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
    else:
        import torch
        args.device = torch.device('cpu')
        model=model.to(args.device) 
        print('The main device is in: ',next(model.parameters()).device)

    ''' load optimizer '''
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, weight_decay=1e-12) # 1e-4
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate, weight_decay=1e-10,amsgrad=True)
    else:
        raise Exception(args.optimizer,'<-- please choice optimizer')

    # LR Scheduler 
    # sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    ''' load checkpoint '''
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("\nLoading model and optimizer from checkpoint '{}'\n".format(args.resume))
            if args.parallel is not None:
                checkpoint = torch.load(args.resume, map_location=args.device) 
                # checkpoint = torch.load(args.resume) #不推荐使用，原因见下面的评论 ↓
                '''评论
                现象：这里以pickle格式储的模型参数，无法通过map location进行得到真正的重映射。
                
                分析：在设计之初，该方法的本意是为了保存的模型本身，而非模型参数而设计的。事实上，如果对模型进行重映射，是可以把硬盘中存储的模型，映射到现有的可见设备之中的。
                但是，在模型设计和调试时，更好的方法通常是仅保存模型参数，而不保存模型结构本身。
                此时，这样的重映射并不能真正的将保存的模型映射到特定设备中，而是占有了特定设备中的一块存储空间（例如某块GPU的200MB显存）
                在模型测试和训练时，这多占用的显存并不参与，相当于被浪费了。但即便如此，这样的方法依然具有意义：在于能够避免多卡训练后的参数，无法在测试中成功载入的到单卡中的情况。
                例如在GPU1训练好的参数，如需load进入GPU0，则该函数会报错。
                
                此外，通常来说，对于仅保存参数的模型文件，往往需要在后续显式地使用“load_state_dict”方法，这间接的使得模型参数能够和模型结合，并保存在模型所在的设备上。
                综合来看，对于map location这一功能来说，在load过程中，将参数映射到model architecture所在的GPU，这样既能避免报错，又能避免浪费空间
                '''
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

 
    ''' load data '''
    FlatImg = utils.FlatImg(args=args, path=path, date=date, date_time=date_time, _re_date=_re_date, model=model, \
                            log_file=log_file, n_classes=n_classes, optimizer=optimizer, \
                            dataset=my_unified_dataset, \
                            data_path=data_path, data_path_validate=data_path_validate, data_path_test=data_path_test)          # , valloaderSet=valloaderSet, v_loaderSet=v_loaderSet
    FlatImg.loadTestData()
    epoch = checkpoint['epoch'] if args.resume is not None else 0

    model.eval() #务必要加此步，转换为测试模式，将所有dropout层和batchnorm停止运作，直接输出
    FlatImg.validateOrTestModelV3(epoch, validate_test='t_all')
    log_file.close()
    exit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='Document-Dewarping-with-Control-Points',
                        help='Architecture')

    # parser.add_argument('--img_shrink', nargs='?', type=int, default=None,
    #                     help='short edge of the input image')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=300,
                        help='# of the epochs')

    parser.add_argument('--optimizer',nargs='?', type=str, default='adam',
                        help='optimization')

    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0002,#2e-4
                        help='Learning Rate')



    parser.add_argument('--print-freq', '-p', default=60, type=int,
                        metavar='N', help='print frequency (default: 10)')  # print frequency

    parser.add_argument('--data_path_train', default=ROOT / 'dataset/fiducial1024/fiducial1024_v1/color', type=str,
                        help='the path of train images.')  # train image path

    parser.add_argument('--data_path_validate', default=ROOT / 'dataset/fiducial1024/fiducial1024_v1/validate', type=str,
                        help='the path of validate images.')  # validate image path

    parser.add_argument('--output-path', default=ROOT / 'flat/', type=str, help='the path is used to  save output --img or result.') 


    
    parser.add_argument('--resume', default='./ICDAR2021/2021-02-03 16_15_55/143/2021-02-03 16_15_55flat_img_by_fiducial_points-fiducial1024_v1.pkl', type=str, 
                        help='Path to previous saved model to restart from')    

    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')#28

    parser.add_argument('--schema', type=str, default='test',
                        help='train or test')       # train  validate

    # parser.set_defaults(resume='/Data_HDD/fmp23_weiguang_zhang/DDCP/flat/2022-06-06/2022-06-06 17:06:59 @2022-06-06/144/2022-06-06@2022-06-06 17:06:59Document-Dewarping-with-Control-Points.pkl')
    parser.add_argument('--data_path_test', default=ROOT / 'dataset/testset/mytest0', type=str, help='the path of test images.')
    
    parser.add_argument('--parallel', default='2', type=list,
                        help='choice the gpu id for parallel ')

    args = parser.parse_args()

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise Exception(args.resume,' -- not exist')

    if args.data_path_test is None:
        raise Exception('-- No test path')
    else:
        if not os.path.exists(args.data_path_test):
            raise Exception(args.data_path_test,' -- no find')

    global path, date, date_time
    date = time.strftime('%Y-%m-%d', time.localtime(time.time())) # '2022-05-30'
    date_time = time.strftime(' %H:%M:%S', time.localtime(time.time())) # '10:23:12'
    path = os.path.join(args.output_path, date) # /Data_HDD/fmp23_weiguang_zhang/DDCP/flat/2022-05-31

    if not os.path.exists(path):
        os.makedirs(path)

    test(args)
