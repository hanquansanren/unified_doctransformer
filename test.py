'''
2022/6/26
Weiguang Zhang
'''
import os, sys
import argparse


import time
import re
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

from network import model_handlebar, DilatedResnet_for_test_single_image

# import utilsV3 as utils
import utilsV4 as utils

from dataset_lmdb import my_unified_dataset


def test(args):
    ''' setup path '''
    data_path_test = str(args.data_path_test)
    
    ''' log writer '''
    global _re_date
    if args.resume is not None:
        re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
        _re_date = re_date.search(args.resume).group(0)
        log_file = open(out_path + '/' + date + date_time + ' @' + _re_date + '_' + args.arch + '.log', 'w+')
    else:
        _re_date = None
        log_file = open(out_path+'/'+date+date_time+'_'+args.arch+'.log', 'w+')

    print(args)
    print(args, file=log_file) # 编写日志功能，我们可以通过改变该参数使print()函数输出到特定的文件中。
    print('log file has been written')


    ''' load model '''
    n_classes = 2
    model = model_handlebar(n_classes=n_classes, num_filter=32, architecture=DilatedResnet_for_test_single_image, BatchNorm='BN', in_channels=3)     #
    model.double()

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
    else:
        import torch
        args.device = torch.device('cpu')
        model=model.to(args.device) 
        print('The main device is in: ',next(model.parameters()).device)


    ''' load model parameter in checkpoint '''
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("\nLoading model and optimizer from checkpoint '{}'\n".format(args.resume))
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
    epoch = checkpoint['epoch'] if args.resume is not None else 0


    ''' load test data and test data'''
    FlatImg = utils.FlatImg(args=args, out_path=out_path, date=date, date_time=date_time, _re_date=_re_date, dataset=my_unified_dataset, \
                            data_path_test=data_path_test, \
                            model = model, optimizer = None, reslut_file=log_file) 
    FlatImg.loadTestData()


    model.eval() #务必要加此步，转换为测试模式，将所有dropout层和batchnorm停止运作，直接输出
    FlatImg.validateOrTestModelV3(epoch, validate_test='t_all')
    log_file.close()
    exit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='DDCP',
                        help='Architecture')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=300,
                        help='# of the epochs')

    parser.add_argument('--optimizer',nargs='?', type=str, default='adam',
                        help='optimization')

    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')  # print frequency

    parser.add_argument('--output-path', default=ROOT / 'flat/', type=str, help='the path is used to  save output --img or result.') 

    parser.add_argument('--resume', default=None, type=str, 
                        help='Path to previous saved model to restart from')    

    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')#28

    parser.add_argument('--schema', type=str, default='test',
                        help='train or test')       # train  validate




    # parser.set_defaults(resume='/Public/FMP_temp/fmp23_weiguang_zhang/DDCP2/flat/2022-09-20/2022-09-20 13:39:28 @2022-09-15/84/2022-09-15@2022-09-20 13:39:28DDCP.pkl')
    # parser.set_defaults(resume='/Public/FMP_temp/fmp23_weiguang_zhang/DDCP2/flat/2022-09-20/2022-09-20 16:40:40/1/2022-09-20 16:40:40DDCP.pkl')
    # parser.set_defaults(resume='/Public/FMP_temp/fmp23_weiguang_zhang/DDCP2/flat/2022-09-21/2022-09-21 21:34:00 @2022-09-20/195/2022-09-20@2022-09-21 21:34:00DDCP.pkl')  
    # ICDAR
    # parser.set_defaults(resume='/Public/FMP_temp/fmp23_weiguang_zhang/DDCP2/ICDAR2021/2021-02-03 16_15_55/143/2021-02-03 16_15_55flat_img_by_fiducial_points-fiducial1024_v1.pkl')  
    parser.set_defaults(resume='/Public/FMP_temp/fmp23_weiguang_zhang/DDCP2/flat/2022-09-22/2022-09-22 15:38:30 @2022-09-20/149/2022-09-20@2022-09-22 15:38:30DDCP.pkl')  

    
    
    
    parser.add_argument('--data_path_test', default=ROOT / 'dataset/testset/mytest0', type=str, help='the path of test images.')
    
    parser.add_argument('--parallel', default='3', type=list,
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

    global out_path, date, date_time
    date = time.strftime('%Y-%m-%d', time.localtime(time.time())) # '2022-05-30'
    date_time = time.strftime(' %H:%M:%S', time.localtime(time.time())) # '10:23:12'
    out_path = os.path.join(args.output_path, date) # /Data_HDD/fmp23_weiguang_zhang/DDCP/flat/2022-05-31

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    test(args)
