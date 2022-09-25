import os
import pickle
from os.path import join as pjoin
import collections
from sys import maxsize
import matplotlib.pyplot as plt
import json
import random
from cv2 import rotate
import torch
import numpy as np
from PIL import Image
import re
import cv2
from synthesis_code.perturbed_images_generation_multiProcess import get_syn_image
from torch.utils import data
import time
import lmdb

class my_unified_dataset(data.Dataset):
	def __init__(self, root, mode='train'):
		self.root = os.path.expanduser(root) # './dataset/train.lmdb/'
		self.mode = mode # 'train'
		self.image_set = collections.defaultdict(dict) # 用于存储image的路径，存为字典形式，字典的key是mode(train or test)
		self.test_imgname_list = collections.defaultdict(list)
		self.row_gap = 1
		self.col_gap = 1
		datasets_mode = ['validate', 'train']
		self.deform_type_list=['fold', 'curve']
		# self.scan_root=os.path.join(self.root, 'digital') # './dataset/WarpDoc/digital'
		# self.wild_root=os.path.join(self.root, 'image')   # './dataset/WarpDoc/image'

		self.model_input_size = (992, 992) # (h,w) 31*32=992, 61*16=976

		if self.mode == 'test':
			img_filename_list = os.listdir(pjoin(self.root))
			self.test_imgname_list[self.mode] = img_filename_list
			self.test_imgname_list[self.mode] = sorted(img_filename_list, key=lambda num: (
			int(re.match(r'(\d+)_(\d+)( copy.png)', num, re.IGNORECASE).group(1)), int(re.match(r'(\d+)_(\d+)( copy.png)', num, re.IGNORECASE).group(2))))
		elif self.mode in datasets_mode:
			self.env = lmdb.open(self.root, map_size = 1099511627776, lock = False)
			self.txn = self.env.begin()
			key_set = []
			for self.idx, (key, value) in enumerate(self.txn.cursor()):
				print(key)
				key_set.append(key)
				if ((self.idx+1)%4)==0:
					self.image_set[self.mode][key.decode().split("_")[3]]=key_set
					key_set = []
			print("finished data")
		else:
			raise Exception('load data error')
		# self.checkimg_availability()

	def __getitem__(self, item):
		if self.mode == 'test':
			im_name = self.test_imgname_list[self.mode][item]
			digital_im_path = pjoin(self.root, im_name)
			im = cv2.imread(digital_im_path, flags=cv2.IMREAD_COLOR)
			im = cv2.resize(im, self.model_input_size, interpolation=cv2.INTER_LINEAR)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
			im = self.transform_im(im) # HWC -> BCHW 
			return im, im_name # 这里的im用于输入模型，获得预测的控制点，并不用于后续的结果保存
		else: # train
			im_set_key_list = self.image_set[self.mode][str(item)] 
			# key '0' ~ '850'
			# value: [b'0_0000_2_d1', b'0_0000_2_d2', b'0_0000_2_di', b'0_0000_2_w1']
			d1, lbl1, d2, lbl2, di, w1 = self.read_img_lmdb(self.env, im_set_key_list)
			# print("get data from lmdb successfully")


			# '''visualization point 1 for synthesis result'''
			# self.check_item_vis(d1, lbl1, 1)
			# self.check_item_vis(d2, lbl2, 2)
			# self.check_item_vis(di, None, 3)
			# self.check_item_vis(w1, None, 4)

			'''参考点生成'''
			# xs = torch.linspace(0, self.model_input_size[1], steps=61)
			# ys = torch.linspace(0, self.model_input_size[0], steps=61)
			# x, y = torch.meshgrid(xs, ys, indexing='xy')
			# reference_point = torch.dstack([x, y])

			# xs = np.linspace(0, self.model_input_size[1], num=61)
			# ys = np.linspace(0, self.model_input_size[0], num=61)
			# x, y = np.meshgrid(xs, ys, indexing='xy')
			# reference_point = np.dstack([x, y])			


			'''resize images, labels and tansform to tensor'''
			lbl1 = self.resize_lbl(lbl1,d1)
			lbl2 = self.resize_lbl(lbl2,d2)

			mask1, pts1 = self.mask_calculator(lbl1) # input:(61,61,2) output:(992,992,3)
			mask2, pts2 = self.mask_calculator(lbl2) # input:(61,61,2) output:(992,992,3)
			lbl1 = self.fiducal_points_lbl(lbl1)
			lbl2 = self.fiducal_points_lbl(lbl2)


			# 两张合成图像，都resize到 (992,992)
			d1=self.resize_im0(d1)
			d2=self.resize_im0(d2)
			di=self.resize_im0(di)
			w1=self.resize_im0(w1)


			# self.check_item_vis(im=d1, lbl=pts1, idx=96)
			# self.check_item_vis(im=d2, lbl=pts2, idx=97)
			# self.check_item_vis(im=mask1, lbl=pts1, idx=98)
			# self.check_item_vis(im=mask2, lbl=pts2, idx=99)


			# '''visualization point 2 for resized synthesized image and sampled control point'''
			# self.check_item_vis(d1, lbl1, 25)
			# self.check_item_vis(d2, lbl2, 26)
			# self.check_item_vis(di, reference_point, 27)
			# self.check_item_vis(w1, None, 28)
			
			d1 = d1.transpose(2, 0, 1)
			d2 = d2.transpose(2, 0, 1)
			lbl1 = lbl1.transpose(2, 0, 1)
			lbl2 = lbl2.transpose(2, 0, 1)
			w1 = w1.transpose(2, 0, 1)
			di = di.transpose(2, 0, 1)
			mask1 = mask1.transpose(2, 0, 1)
			mask2 = mask2.transpose(2, 0, 1)
			
			d1 = torch.from_numpy(d1).double() # torch.float32 torch.Size([3, 992, 992])
			lbl1 = torch.from_numpy(lbl1).double()    # torch.float64 torch.Size([2, 31, 31])
			d2 = torch.from_numpy(d2).double() # torch.float32 torch.Size([3, 992, 992])
			lbl2 = torch.from_numpy(lbl2).double()    # torch.float64 torch.Size([2, 31, 31])
			w1 = torch.from_numpy(w1).double()     # torch.float32 torch.Size([3, 992, 992])
			di = torch.from_numpy(di).double()    # torch.float32 torch.Size([3, 992, 992])
			# reference_point = torch.from_numpy(reference_point).double() # torch.float64 torch.Size([2, 31, 31])
			mask1 = torch.from_numpy(mask1).double()
			mask2 = torch.from_numpy(mask2).double()



			# print('finished dataset preparation')
			return d1, lbl1, d2, lbl2, w1, di, mask1, mask2

	def __len__(self):
		if self.mode == 'test':
			return len(self.test_imgname_list[self.mode])
		elif self.mode == 'train':
			# print(len(range(int((self.idx+1)/4))))
			return len(range(int((self.idx+1)/4)))
		else:
			print("error __len__")

	def transform_im(self, im):
		im = im.transpose(2, 0, 1)
		im = torch.from_numpy(im).double()

		return im

	def resize_im0(self, im):
		try:
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
			im = cv2.resize(im, self.model_input_size, interpolation=cv2.INTER_LINEAR) 
			# im = cv2.resize(im, (496, 496), interpolation=cv2.INTER_LINEAR)
		except:
			pass

		return im

	def resize_lbl(self, lbl, image):
		h=image.shape[0]
		w=image.shape[1]
		lbl = lbl/[w, h]*[992, 992]
		# lbl = lbl/[960, 1024]*[496, 496]
		return lbl

	def fiducal_points_lbl(self, fiducial_points):
		'''
		根据生成的(61,61,2)，对其进行采样
		'''
		# 采样步长设定，POINTS NUM: 61, 31, 21, 16, 13, 11, 7, 6, 5, 4, 3, 2
		point_sample_step = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]  
		# 在x和y方向，根据步长`point_sample_step[self.row_gap]`来控制控制点的采样。
		fiducial_points = fiducial_points[::point_sample_step[self.row_gap], ::point_sample_step[self.col_gap], :] 
		# interval = interval * [point_sample_step[self.col_gap], point_sample_step[self.row_gap]]
		# return fiducial_points, interval
		return fiducial_points

	def read_img_lmdb(self, env, key_list):
		txn = env.begin(write=False)
		if key_list[0].decode()[-2:]=='d1' and key_list[1].decode()[-2:]=='d2' and key_list[2].decode()[-2:]=='di' and key_list[3].decode()[-2:]=='w1':
			value1 = txn.get(key_list[0])
			value1 = pickle.loads(value1)
			d1 = value1['image'] # np.uint8
			lbl1 = value1['label'] # np.float64
			value2 = txn.get(key_list[1])
			value2 = pickle.loads(value2)
			d2 = value2['image']
			lbl2 = value2['label']
			value3 = txn.get(key_list[2])
			value3 = pickle.loads(value3)
			di = value3['image']
			value4 = txn.get(key_list[3])
			value4 = pickle.loads(value4)
			w1 = value4['image']
		return d1,lbl1,d2,lbl2,di,w1

	def mask_calculator(self,lbl):
		pt_edge = lbl[0,:,:] 
		for num in range(1,60,1):
			pt_edge=np.append(pt_edge, lbl[num,60,:][None,:] ,axis=0)
		pt_edge = np.vstack((pt_edge,lbl[60,:,:][::-1,:]))
		for num in range(59,0,-1):
			pt_edge=np.append(pt_edge, lbl[num,0,:][None,:] ,axis=0)
				
		img = np.zeros((992, 992, 3), dtype=np.int32)
		pts = pt_edge.round().astype(int)

		mask = cv2.fillPoly(img, [pts], (1, 1, 1))
		cv2.imwrite('./simple_test/interpola_vis/get_item_mask{}.png'.format(11), mask)
		return mask, pts

	def check_item_vis(self, im=None, lbl=None, idx=None):
		'''
		im : distorted image   # HWC 
		lbl : fiducial_points  # 61*61*2 
		'''
		if im is not None:
			im=np.uint8(im)
			h=im.shape[0]*0.01
			w=im.shape[1]*0.01
			im = Image.fromarray(im)
			im.convert('RGB').save("./data_vis/img_vis{}.png".format(idx))
		
		if lbl is not None:
			# fig, ax = plt.subplots(figsize = (w,h),facecolor='black')
			# ax.imshow(im)
			# ax.scatter(lbl[:,:,0].flatten(),lbl[:,:,1].flatten(),s=1.2,c='red',alpha=1)
			# ax.axis('off')
			# plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
			# # plt.tight_layout()
			# plt.savefig('./data_vis/point_vis{}.png'.format(idx))
			# plt.close()
			# plt.figure(figsize = (w,h),facecolor='white')
			plt.figure(figsize = (w, h),facecolor='white')
			plt.imshow(im)
			# plt.scatter(lbl[:,:,0].flatten(),lbl[:,:,1].flatten(),s=1.2,c='red',alpha=1)
			plt.scatter(lbl[:,0].flatten(),lbl[:,1].flatten(),s=1.2,c='red',alpha=1)
			plt.axis('off')
			plt.margins(0,0)
			plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
			plt.savefig('./data_vis/point_vis{}.png'.format(idx))
			plt.close()

	def checkimg_availability(self):
		if self.mode == 'train' or self.mode == 'validate':
			for im_name in self.image_set[self.mode]:
				digital_im_path = pjoin(self.scan_root, im_name)
				try:
					img = Image.open(digital_im_path)
				except:
					print("bug image:", im_name)
					# os.remove(digital_im_path)
		print('all images is well prepared')

