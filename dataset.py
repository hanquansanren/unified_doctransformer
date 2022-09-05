import os
import pickle
from os.path import join as pjoin
import collections
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


class my_unified_dataset(data.Dataset):
	def __init__(self, root, mode='train', bfreq=None,hpf= None):
		self.root = os.path.expanduser(root) # './dataset/WarpDoc/'
		self.mode = mode
		# self.mean = np.array([104.00699, 116.66877, 122.67892])
		self.images = collections.defaultdict(list) # 用于存储image的路径，存为字典形式，字典的key是mode(train or test)
		self.labels = collections.defaultdict(list) # 暂时没用到，因为label是直接生成的，不存在数据集里
		self.row_gap = 1
		self.col_gap = 1
		# self.bfreq = bfreq
		# self.hpf = hpf
		datasets = ['validate', 'train']
		self.deform_type_list=['fold', 'curve']
		self.scan_root=os.path.join(self.root, 'digital') # './dataset/WarpDoc/digital'
		self.wild_root=os.path.join(self.root, 'image')   # './dataset/WarpDoc/image'
		self.bg_path = './dataset/background/'
		self.save_path = './output/' # 用于将合成图像打包，在训练时保留即可，一般并不会被用到


		if self.mode == 'test':
			img_file_list = os.listdir(pjoin(self.root))
			self.images[self.mode] = img_file_list
			self.images[self.mode] = sorted(img_file_list, key=lambda num: (
			int(re.match(r'(\d+)_(\d+)( copy.png)', num, re.IGNORECASE).group(1)), int(re.match(r'(\d+)_(\d+)( copy.png)', num, re.IGNORECASE).group(2))))
		elif self.mode in datasets:
			img_file_list = []	
			for type in os.listdir(self.scan_root):
				for file_idex in os.listdir(pjoin(self.scan_root, type)):
					img_file_list.append(pjoin(type, file_idex))

			self.images[self.mode] = img_file_list # key-value pair
		else:
			raise Exception('load data error')
		# self.checkimg_availability()



	def __len__(self):
		return len(self.images[self.mode])

	def __getitem__(self, item):
		if self.mode == 'test':
			im_name = self.images[self.mode][item]
			digital_im_path = pjoin(self.root, im_name)
			
			im = cv2.imread(digital_im_path, flags=cv2.IMREAD_COLOR)
			# h,w,c=im.shape
			# if h<1024:
			# 	im=np.pad(im,(((1024-h)//2,(1024-h)//2),(0,0),(0,0)),'constant')
			# if w<960:
			# 	im=np.pad(im,((0,0),((960-w)//2,(960-w)//2),(0,0)),'constant')
			# print(im.shape)
			im = cv2.resize(im, (1020, 1020), interpolation=cv2.INTER_LINEAR)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
			im = self.transform_im(im) # HWC -> CHW

			return im, im_name

		else: # train
			'''load path'''			
			im_name = self.images[self.mode][item] # 'rotate/0151.jpg'
			digital_im_path = pjoin(self.scan_root, im_name) # './dataset/WarpDoc/digital/rotate/0151.jpg'
			wild_im_path = pjoin(self.wild_root, im_name) # './dataset/WarpDoc/image/rotate/0151.jpg'
			
			'''choose two deform type randomly'''
			deform_type1=np.random.choice(self.deform_type_list,p=[0.5,0.5])
			print('the first deformation type is:', deform_type1)
			deform_type2=np.random.choice(self.deform_type_list,p=[0.5,0.5])
			print('the second deformation type is:', deform_type2)

			# '''get two deformation document images'''
			# d1,lbl1=get_syn_image(path=digital_im_path, bg_path=self.bg_path,save_path=self.save_path,deform_type=deform_type1)
			# d2,lbl2=get_syn_image(path=digital_im_path, bg_path=self.bg_path,save_path=self.save_path,deform_type=deform_type2)
			# # shape of d1,d2: (1024, 768, 3)
			# # shape of lbl1,lbl2: (61, 61, 2)
			# d1=d1[:, :, ::-1] # d1和d2为cv2生成的合成图像，这里需要转换为RGB通道顺序
			# d2=d2[:, :, ::-1] # d1和d2为cv2生成的合成图像，这里需要转换为RGB通道顺序
			# print('finished two deformation document images')

			# '''visualization point 1 for synthesis result'''
			# self.check_item_vis(d1, lbl1, 1)
			# self.check_item_vis(d2, lbl2, 2)

			''' load digital and wild images pair'''
			digital_im = cv2.imread(digital_im_path, flags=cv2.IMREAD_COLOR)
			wild_im = cv2.imread(wild_im_path, flags=cv2.IMREAD_COLOR)
			digital_im = cv2.resize(digital_im, (1020, 1020), interpolation=cv2.INTER_LINEAR)
			wild_im = cv2.resize(wild_im, (1020, 1020), interpolation=cv2.INTER_LINEAR)
			# digital_im = cv2.cvtColor(digital_im, cv2.COLOR_BGR2RGB)
			# wild_im = cv2.cvtColor(wild_im, cv2.COLOR_BGR2RGB)
			digital_im = self.transform_im(digital_im) # HWC -> BCHW
			wild_im = self.transform_im(wild_im)       # HWC -> BCHW
			wild=wild_im.int().numpy()

			'''参考点生成'''
			xs = torch.linspace(0, 1020, steps=61)
			ys = torch.linspace(0, 1020, steps=61)
			x, y = torch.meshgrid(xs, ys, indexing='xy')
			reference_point = torch.dstack([x, y])
			reference_point = reference_point.permute(2, 0, 1)



			# im = self.resize_im1(im, self.bfreq, self.hpf)
			'''resize images, labels and tansform to tensor'''
			lbl1 = self.resize_lbl(lbl1,d1)
			lbl2 = self.resize_lbl(lbl2,d2)
			lbl1 = self.fiducal_points_lbl(lbl1)
			lbl2 = self.fiducal_points_lbl(lbl2)

			# 两张合成图像，都resize到 (1020,1020)
			d1=self.resize_im0(d1)
			d2=self.resize_im0(d2)

			# '''visualization point 2 for resized synthesized image and sampled control point'''
			# self.check_item_vis(d1, lbl1, 1)
			# self.check_item_vis(d2, lbl2, 2)
			
			d1 = d1.transpose(2, 0, 1)
			d2 = d2.transpose(2, 0, 1)
			lbl1 = lbl1.transpose(2, 0, 1)
			lbl2 = lbl2.transpose(2, 0, 1)
			d1 = torch.from_numpy(d1).float()
			lbl1 = torch.from_numpy(lbl1).float()

			d2 = torch.from_numpy(d2).float()
			lbl2 = torch.from_numpy(lbl2).float()

			print('finished dataset preparation')
			return d1, lbl1, d2, lbl2, wild_im, digital_im, reference_point

	def transform_im(self, im):
		im = im.transpose(2, 0, 1)
		im = torch.from_numpy(im).float()

		return im

	def resize_im0(self, im):
		try:
			im = cv2.resize(im, (1020, 1020), interpolation=cv2.INTER_LINEAR)
			# im = cv2.resize(im, (496, 496), interpolation=cv2.INTER_LINEAR)
		except:
			pass

		return im


	def resize_lbl(self, lbl, image):
		h=image.shape[0]
		w=image.shape[1]
		lbl = lbl/[w, h]*[1020, 1020]
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










	def check_item_vis(self, im, lbl, idx):
		'''
		im : distorted image   # HWC 
		lbl : fiducial_points  # 61*61*2 
		'''
		# im=np.uint8(im)
		h=im.shape[0]*0.01
		w=im.shape[1]*0.01
		im = Image.fromarray(im)
		im.convert('RGB').save("./data_vis/img_vis{}.png".format(idx))
		
		fig, ax = plt.subplots(figsize = (w,h),facecolor='white')
		ax.imshow(im)
		ax.scatter(lbl[:,:,0].flatten(),lbl[:,:,1].flatten(),s=1.2,c='red',alpha=1)
		ax.axis('off')
		plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
		# plt.tight_layout()
		plt.savefig('./data_vis/point_vis{}.png'.format(idx))
		plt.close()

	# def resize_im1(self, im, bfreq, hpf):
	# 	try:
	# 		im = cv2.resize(im, (1020, 1020), interpolation=cv2.INTER_LINEAR)
	# 		# im = cv2.resize(im, (496, 496), interpolation=cv2.INTER_LINEAR)
	# 	except:
	# 		pass
	
	# 	# freq = np.fft.fft2(im,axes=(0,1))
	# 	# freq = np.fft.fftshift(freq)

	# 	# rhpf = bfreq + hpf * freq
	# 	# img_rhpf = np.abs(np.fft.ifft2(rhpf,axes=(0,1)))
	# 	# img_rhpf = np.clip(img_rhpf,0,255) #会产生一些过大值需要截断
	# 	# img_rhpf = img_rhpf.astype('uint8')

	# 	# return img_rhpf
	# 	return im
	
	def checkimg_availability(self):
		if self.mode == 'train' or self.mode == 'validate':
			for im_name in self.images[self.mode]:
				digital_im_path = pjoin(self.scan_root, im_name)
				try:
					img = Image.open(digital_im_path)
				except:
					print("bug image:", im_name)
					# os.remove(digital_im_path)
		print('all images is well prepared')

