'''
Guowang Xie
from utilsV4.py
'''
import torch
from torch.utils import data
from torch.autograd import Variable, Function
import numpy as np
import sys, os, math
import cv2
import time
import re
import random
from scipy.interpolate import griddata
from tpsV2 import createThinPlateSplineShapeTransformer


class SaveFlatImage(object):
    '''
    Post-processing and save result.
    Function:
        flatByRegressWithClassiy_multiProcessV2: Selecting a post-processing method
        flatByfiducial_TPS: Thin Plate Spline, input multi-batch
        flatByfiducial_interpolation: Interpolation, input one image
    '''
    def __init__(self, path, date, date_time, _re_date, data_path_validate, data_path_test, batch_size, preproccess=False, postprocess='tps', device=torch.device('cuda:0')):
        self.path = path
        self.date = date
        self.date_time = date_time
        self._re_date = _re_date
        self.preproccess = preproccess
        self.data_path_validate =data_path_validate
        self.data_path_test = data_path_test
        self.batch_size = batch_size
        self.device = device
        self.col_gap = 0 #4
        self.row_gap = self.col_gap# col_gap + 1 if col_gap < 6 else col_gap
        # fiducial_point_gaps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]  # POINTS NUM: 61, 31, 21, 16, 13, 11, 7, 6, 5, 4, 3, 2
        self.fiducial_point_gaps = [1, 2, 3, 5, 6, 10, 15, 30]        # POINTS NUM: 31, 16, 11, 7, 6, 4, 3, 2
        self.fiducial_point_num = [31, 16, 11, 7, 6, 4, 3, 2]
        self.fiducial_num = self.fiducial_point_num[self.col_gap], self.fiducial_point_num[self.row_gap]
        map_shape = (320, 320)
        self.postprocess = postprocess
        
        if self.postprocess == 'tps':
            self.tps = createThinPlateSplineShapeTransformer(map_shape, fiducial_num=self.fiducial_num, device=self.device)


    def flatByfiducial_TPS(self, fiducial_points, segment, im_name, epoch, perturbed_img=None, scheme='validate', is_scaling=False):
        '''
        flat_shap controls the output image resolution
        '''
        # if (scheme == 'test' or scheme == 'eval') and is_scaling:
        #     pass
        # else:
        if scheme == 'test' or scheme == 'eval':
            perturbed_img_path = self.data_path_test + im_name
            perturbed_img = cv2.imread(perturbed_img_path, flags=cv2.IMREAD_COLOR)
            # perturbed_img = cv2.resize(perturbed_img, (960, 1024))
        elif scheme == 'validate' and perturbed_img is None:
            RGB_name = im_name.replace('gw', 'png')
            perturbed_img_path = self.data_path_validate + '/png/' + RGB_name
            perturbed_img = cv2.imread(perturbed_img_path, flags=cv2.IMREAD_COLOR)
        elif perturbed_img is not None:
            perturbed_img = perturbed_img.transpose(1, 2, 0)

        fiducial_points = fiducial_points / [992, 992] #归一化，模型输出点稀疏点本身就在992*992的范围内
        perturbed_img_shape = perturbed_img.shape[0:2]
        
        '''输出图尺寸设定'''
        # 将预测的参考点间距，乘以31倍
        # flat_shap = segment * [self.fiducial_point_gaps[self.col_gap], self.fiducial_point_gaps[self.row_gap]] * [self.fiducial_point_num[self.col_gap], self.fiducial_point_num[self.row_gap]]
        # 维持与输入图相同的尺寸
        flat_shap = perturbed_img_shape
        
        
        time_1 = time.time()
        '''转换为tensor，并归一化'''
        perturbed_img_ = torch.tensor(perturbed_img.transpose(2,0,1)[None,:]) # [h, w, c] -> [b, c, h, w]
        fiducial_points_ = torch.tensor(fiducial_points.transpose(1,0,2).reshape(-1,2)) # [31,31,2] -> [1,961,2]
        fiducial_points_ = (fiducial_points_[None,:]-0.5)*2 #将[0,1]的数据转换到[-1,1]范围内
        
        rectified = self.tps(perturbed_img_.double().to(self.device), fiducial_points_.to(self.device), list(flat_shap))
        
        
        
        time_2 = time.time()
        time_interval = time_2 - time_1
        print('TPS time: '+ str(time_interval))

        flat_img = rectified[0].cpu().numpy().transpose(1,2,0)

        '''save'''
        flat_img = flat_img.astype(np.uint8)

        i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date,
                              str(epoch)) if self._re_date is not None else os.path.join(self.path,
                                                                                         self.date + self.date_time,
                                                                                         str(epoch))
        ''''''

        sshape = fiducial_points[::self.fiducial_point_gaps[self.row_gap], ::self.fiducial_point_gaps[self.col_gap], :]
        perturbed_img_mark = self.location_mark(perturbed_img.copy(), sshape*perturbed_img_shape[::-1], (0, 0, 255))

        if scheme == 'test':
            i_path += '/test'
        if not os.path.exists(i_path):
            os.makedirs(i_path)

        im_name = im_name.replace('gw', 'png')
        cv2.imwrite(i_path + '/mark_' + im_name, perturbed_img_mark)
        cv2.imwrite(i_path + '/' + im_name, flat_img)

    def flatByfiducial_interpolation(self, fiducial_points, segment, im_name, epoch, perturbed_img=None, scheme='validate', is_scaling=False):
        ''''''
        if scheme == 'test' or scheme == 'eval':
            perturbed_img_path = self.data_path_test + im_name
            perturbed_img = cv2.imread(perturbed_img_path, flags=cv2.IMREAD_COLOR)
            perturbed_img = cv2.resize(perturbed_img, (960, 1024))
        elif scheme == 'validate' and perturbed_img is None:
            RGB_name = im_name.replace('gw', 'png')
            perturbed_img_path = self.data_path_validate + '/png/' + RGB_name
            perturbed_img = cv2.imread(perturbed_img_path, flags=cv2.IMREAD_COLOR)
        elif perturbed_img is not None:
            perturbed_img = perturbed_img.transpose(1, 2, 0)

        fiducial_points = fiducial_points / [992, 992] * [960, 1024]
        col_gap = 2 #4
        row_gap = col_gap# col_gap + 1 if col_gap < 6 else col_gap
        # fiducial_point_gaps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]  # POINTS NUM: 61, 31, 21, 16, 13, 11, 7, 6, 5, 4, 3, 2
        fiducial_point_gaps = [1, 2, 3, 5, 6, 10, 15, 30]        # POINTS NUM: 31, 16, 11, 7, 6, 4, 3, 2
        sshape = fiducial_points[::fiducial_point_gaps[row_gap], ::fiducial_point_gaps[col_gap], :]
        segment_h, segment_w = segment * [fiducial_point_gaps[col_gap], fiducial_point_gaps[row_gap]]
        fiducial_points_row, fiducial_points_col = sshape.shape[:2]

        im_x, im_y = np.mgrid[0:(fiducial_points_col - 1):complex(fiducial_points_col),
                     0:(fiducial_points_row - 1):complex(fiducial_points_row)]

        tshape = np.stack((im_x, im_y), axis=2) * [segment_w, segment_h]

        tshape = tshape.reshape(-1, 2)
        sshape = sshape.reshape(-1, 2)

        output_shape = (segment_h * (fiducial_points_col - 1), segment_w * (fiducial_points_row - 1))
        grid_x, grid_y = np.mgrid[0:output_shape[0] - 1:complex(output_shape[0]),
                         0:output_shape[1] - 1:complex(output_shape[1])]
        time_1 = time.time()
        # grid_z = griddata(tshape, sshape, (grid_y, grid_x), method='cubic').astype('float32')
        grid_ = griddata(tshape, sshape, (grid_y, grid_x), method='linear').astype('float32')
        flat_img = cv2.remap(perturbed_img, grid_[:, :, 0], grid_[:, :, 1], cv2.INTER_CUBIC)
        time_2 = time.time()
        time_interval = time_2 - time_1
        print('Interpolation time: '+ str(time_interval))
        ''''''
        flat_img = flat_img.astype(np.uint8)

        i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date,
                              str(epoch)) if self._re_date is not None else os.path.join(self.path,
                                                                                         self.date + self.date_time,
                                                                                         str(epoch))
        ''''''
        perturbed_img_mark = self.location_mark(perturbed_img.copy(), sshape, (0, 0, 255))

        shrink_paddig = 0   # 2 * edge_padding
        x_start, x_end, y_start, y_end = shrink_paddig, segment_h * (fiducial_points_col - 1) - shrink_paddig, shrink_paddig, segment_w * (fiducial_points_row - 1) - shrink_paddig

        x_ = (perturbed_img_mark.shape[0]-(x_end-x_start))//2
        y_ = (perturbed_img_mark.shape[1]-(y_end-y_start))//2

        flat_img_new = np.zeros_like(perturbed_img_mark)
        flat_img_new[x_:perturbed_img_mark.shape[0] - x_, y_:perturbed_img_mark.shape[1] - y_] = flat_img
        img_figure = np.concatenate(
            (perturbed_img_mark, flat_img_new), axis=1)

        if scheme == 'test':
            i_path += '/test'
        if not os.path.exists(i_path):
            os.makedirs(i_path)

        im_name = im_name.replace('gw', 'png')
        cv2.imwrite(i_path + '/' + im_name, img_figure)

    def flatByRegressWithClassiy_multiProcessV2(self, pred_fiducial_points, pred_segment, im_name, epoch, process_pool=None, perturbed_img=None, scheme='validate', is_scaling=False):
        for i_val_i in range(pred_fiducial_points.shape[0]):
            if self.postprocess == 'tps':
                self.flatByfiducial_TPS(pred_fiducial_points[i_val_i], pred_segment[i_val_i], im_name[i_val_i], epoch, None if perturbed_img is None else perturbed_img[i_val_i], scheme, is_scaling)
            elif self.postprocess == 'interpolation':
                self.flatByfiducial_interpolation(pred_fiducial_points[i_val_i], pred_segment[i_val_i], im_name[i_val_i], epoch, None if perturbed_img is None else perturbed_img[i_val_i], scheme, is_scaling)
            else:
                print('Error: Other postprocess.')
                exit()

    def location_mark(self, img, location, color=(0, 0, 255)):
        stepSize = 0
        for l in location.astype(np.int64).reshape(-1, 2):
            cv2.circle(img,
                       (l[0] + math.ceil(stepSize / 2), l[1] + math.ceil(stepSize / 2)), 3, color, -1)
        return img

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, m=1):
        self.val = val
        self.sum += val * m
        self.count += n
        self.avg = self.sum / self.count

class FlatImg(object):
    '''
    args:
        dataloader
    '''
    def __init__(self, args, path, date, date_time, _re_date, model,\
                 log_file, n_classes, optimizer, \
                 model_D=None, optimizer_D=None, \
                 loss_fn=None, loss_fn2=None, dataset=None, data_loader_hdf5=None, dataPackage_loader = None, \
                 data_path=None, data_path_validate=None, data_path_test=None, is_data_preproccess=True):  
        self.args = args
        self.path = path
        self.date = date
        self.date_time = date_time
        self._re_date = _re_date
        # self.valloaderSet = valloaderSet
        # self.v_loaderSet = v_loaderSet
        self.model = model
        self.model_D = model_D
        self.log_file = log_file
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.optimizer_D = optimizer_D
        self.loss_fn = loss_fn
        self.loss_fn2 = loss_fn2
        self.dataset = dataset
        self.data_loader_hdf5 = data_loader_hdf5
        self.dataPackage_loader = dataPackage_loader
        self.is_data_preproccess = is_data_preproccess

        self.data_path = data_path
        self.data_path_validate = data_path_validate
        self.data_path_test = data_path_test



        postprocess_list = ['tps', 'interpolation']
        self.save_flat_mage = SaveFlatImage(self.path, self.date, self.date_time, self._re_date, \
                                            self.data_path_validate, self.data_path_test, \
                                            self.args.batch_size, \
                                            self.is_data_preproccess, postprocess=postprocess_list[0], \
                                            device=torch.device(self.args.device))

        self.validate_loss = AverageMeter()
        self.validate_loss_regress = AverageMeter()
        self.validate_loss_segment = AverageMeter()
        self.lambda_loss = 1
        self.lambda_loss_segment = 1
        self.lambda_loss_a = 1
        self.lambda_loss_b = 1
        self.lambda_loss_c = 1


    def fdr(self):
        blank_im = np.uint8(255*np.ones((992, 992, 3)))
		# blank_im = Image.fromarray(np.uint8(blank_im))
        h,w,c = blank_im.shape
		#生成低通和高通滤波器
        lpf = np.zeros((h,w,3))
        sh = h*0.06
        sw = w*0.06
        for x in range(w):
            for y in range(h):
                if (x<(w/2+sw)) and (x>(w/2-sw)):
                    if (y<(h/2+sh)) and (y>(h/2-sh)):
                        lpf[y,x,:] = 1            

        bfreq = np.fft.fft2(blank_im,axes=(0,1))
        bfreq = np.fft.fftshift(bfreq)
        return lpf*bfreq, 1-lpf

    def loadTrainData(self, data_split):
        # bfreq,hpf=self.fdr()
        train_dataset = self.dataset(self.data_path, mode=data_split)
        trainloader = data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=min(self.args.batch_size, 16), drop_last=True, pin_memory=True,
                                      shuffle=True)
        return trainloader

    def loadTestData(self):
        test_dataset = self.dataset(self.data_path_test, mode='test')
        self.testloader1 = data.DataLoader(test_dataset, batch_size=self.args.batch_size, num_workers=min([os.cpu_count(), self.args.batch_size if self.args.batch_size > 1 else 0, 8]),
                                      shuffle=False, pin_memory=True)



    def saveModel_epoch(self, epoch):
        epoch += 1
        state = {'epoch': epoch,
                 'model_state': self.model.state_dict(),
                 'optimizer_state': self.optimizer.state_dict(),    # AN ERROR HAS OCCURED
                 }
        i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date,
                              str(epoch)) if self._re_date is not None else os.path.join(self.path, self.date + self.date_time, str(epoch))
        if not os.path.exists(i_path):
            os.makedirs(i_path)

        if self._re_date is None:
            torch.save(state, i_path + '/' + self.date + self.date_time + "{}".format(self.args.arch) + ".pkl")  # "./trained_model/{}_{}_best_model.pkl"
        else:
            torch.save(state,
                       i_path + '/' + self._re_date + "@" + self.date + self.date_time + "{}".format(
                           self.args.arch) + ".pkl")

    def validateOrTestModelV3(self, epoch, validate_test=None, is_scaling=False):

        if validate_test == 't_all':
            begin_test = time.time()
            with torch.no_grad():
                for i_val, (images, im_name) in enumerate(self.testloader1):
                    try:
                        if self.args.parallel is not None:
                            images=images.cuda()
                        else:
                            images=images.to(self.args.device)
                        
                        print(images.device)
                        outputs, outputs_segment = self.model(images) # outputs_segment是平整图像的点间隔
                        pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)
                        pred_segment = outputs_segment.data.round().int().cpu().numpy()

                        self.save_flat_mage.flatByRegressWithClassiy_multiProcessV2(pred_regress,
                                                                                    pred_segment, im_name,
                                                                                    epoch + 1,
                                                                                    scheme='test', is_scaling=is_scaling)
                    except:
                        print('* save image tested error :' + im_name[0])
                test_time = time.time() - begin_test

                print('total test time : {test_time:.3f} seconds'.format(
                    test_time=test_time))

                print('total test time : {test_time:.3f} seconds'.format(
                    test_time=test_time),file=self.log_file) # save log
        else:
            begin_test = time.time()
            with torch.no_grad():
                for i_valloader, valloader in enumerate(self.testloaderSet.values()):

                    for i_val, (images, im_name) in enumerate(valloader):
                        try:
                            # save_img_ = True
                            # save_img_ = random.choices([True, False], weights=[1, 0])[0]
                            save_img_ = random.choices([True, False], weights=[0.4, 0.6])[0]

                            if save_img_:
                                images = images.cuda()

                                outputs, outputs_segment = self.model(images)
                                # outputs, outputs_segment = self.model(images, is_softmax=True)

                                pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)
                                pred_segment = outputs_segment.data.round().int().cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()

                                self.save_flat_mage.flatByRegressWithClassiy_multiProcessV2(pred_regress,
                                                                                          pred_segment, im_name,
                                                                                          epoch + 1,
                                                                                          scheme='test', is_scaling=is_scaling)
                        except:
                            print('* save image tested error :' + im_name[0])
                test_time = time.time() - begin_test

                print('test time : {test_time:.3f}'.format(
                    test_time=test_time))

                print('test time : {test_time:.3f}'.format(
                    test_time=test_time), file=self.log_file)


