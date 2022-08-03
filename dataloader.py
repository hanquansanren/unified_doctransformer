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


class PerturbedDatastsForFiducialPoints_pickle_color_v2_v2(data.Dataset):
	def __init__(self, root, mode='train', img_shrink=None, is_return_img_name=False, bfreq=None,hpf= None):
		self.root = os.path.expanduser(root) # './dataset/WarpDoc/'
		self.mode = mode
		self.img_shrink = img_shrink
		self.is_return_img_name = is_return_img_name
		# self.mean = np.array([104.00699, 116.66877, 122.67892])
		self.images = collections.defaultdict(list)
		self.labels = collections.defaultdict(list)
		self.row_gap = 1
		self.col_gap = 1
		# self.bfreq = bfreq
		# self.hpf = hpf
		datasets = ['validate', 'train']
		self.deform_type_list=['fold', 'curve']

		self.scan_root=os.path.join(self.root, 'digital')
		self.wild_root=os.path.join(self.root, 'image')
		self.scan_root=os.path.join(self.root, 'rotate')

		self.bg_path = './dataset/background/'
		self.save_path = './output/'


		if self.mode == 'test' or self.mode == 'eval':
			img_file_list = os.listdir(pjoin(self.root))
			self.images[self.mode] = img_file_list
			# self.images[self.mode] = sorted(img_file_list, key=lambda num: (
			# int(re.match(r'(\d+)_(\d+)( copy.png)', num, re.IGNORECASE).group(1)), int(re.match(r'(\d+)_(\d+)( copy.png)', num, re.IGNORECASE).group(2))))
		elif self.mode in datasets:
			img_file_list = []		
			for type in os.listdir(self.scan_root):
				for file_idex in os.listdir(pjoin(self.scan_root, type)):
					img_file_list.append(pjoin(type, file_idex))

			self.images[self.mode] = img_file_list
		else:
			raise Exception('load data error')
		# self.checkimg()

	def checkimg(self):
		if self.mode == 'train' or self.mode == 'validate':
			for im_name in self.images[self.mode]:
				im_path = pjoin(self.scan_root, im_name)
				try:
					img = Image.open(im_path)
				except:
					print("bug image:", im_name)
					# os.remove(im_path)
		print('all images is well prepared')

	def __len__(self):
		return len(self.images[self.mode])

	def __getitem__(self, item):
		if self.mode == 'test':
			im_name = self.images[self.mode][item]
			im_path = pjoin(self.root, im_name)

			im = cv2.imread(im_path, flags=cv2.IMREAD_COLOR)
			h,w,c=im.shape
			# if h<1024:
			# 	im=np.pad(im,(((1024-h)//2,(1024-h)//2),(0,0),(0,0)),'constant')
			# if w<960:
			# 	im=np.pad(im,((0,0),((960-w)//2,(960-w)//2),(0,0)),'constant')
			# print(im.shape)
			im = self.resize_im0(im)
			im = self.transform_im(im)

			if self.is_return_img_name:
				return im, im_name
			return im
		elif self.mode == 'eval':
			im_name = self.images[self.mode][item]
			im_path = pjoin(self.root, im_name)

			img = cv2.imread(im_path, flags=cv2.IMREAD_COLOR)

			im = self.resize_im0(img)
			im = self.transform_im(im)

			if self.is_return_img_name:
				return im, im_name
			return im, img
			# return im, img, im_name

		else: # train
			# img = Image.open(im_path)	
			'''load path'''			
			im_name = self.images[self.mode][item] # 'rotate/0151.jpg'
			im_path = pjoin(self.scan_root, im_name) # './dataset/WarpDoc/digital/rotate/0151.jpg'
			
			'''choose two deform type randomly'''
			deform_type1=np.random.choice(self.deform_type_list,p=[0.5,0.5])
			print(deform_type1)
			deform_type2=np.random.choice(self.deform_type_list,p=[0.5,0.5])
			print(deform_type2)

			'''get two deformation document images'''
			d1,lbl1,itv1=get_syn_image(path=im_path, bg_path=self.bg_path,save_path=self.save_path,deform_type=deform_type1)
			d2,lbl2,itv2=get_syn_image(path=im_path, bg_path=self.bg_path,save_path=self.save_path,deform_type=deform_type2)
			d1=d1[:, :, ::-1]
			d2=d2[:, :, ::-1]
			print('finished two deformation document images')

			'''visualization point'''
			self.check_item_vis(d1, lbl1, 1)
			self.check_item_vis(d2, lbl2, 2)


			# im = self.resize_im1(im, self.bfreq, self.hpf)
			'''resize and tansform to tensor'''
			d1=self.resize_im0(d1)
			d2=self.resize_im0(d2)

			lbl1 = self.resize_lbl(lbl1)
			lbl2 = self.resize_lbl(lbl2)
			lbl1, itv1 = self.fiducal_points_lbl(lbl1, itv1)
			lbl2, itv2 = self.fiducal_points_lbl(lbl2, itv2)

			'''visualization point'''
			self.check_item_vis(d1, lbl1, 1)
			self.check_item_vis(d2, lbl2, 2)
			
			d1 = d1.transpose(2, 0, 1)
			d2 = d2.transpose(2, 0, 1)
			lbl1 = lbl1.transpose(2, 0, 1)
			lbl2 = lbl2.transpose(2, 0, 1)
			d1 = torch.from_numpy(d1).float()
			lbl1 = torch.from_numpy(lbl1).float()
			itv1 = torch.from_numpy(itv1).float()
			d2 = torch.from_numpy(d2).float()
			lbl2 = torch.from_numpy(lbl2).float()
			itv2 = torch.from_numpy(itv2).float()
			print('finished dataset preparation')
			return d1, lbl1, itv2, d2, lbl2, itv2

	def transform_im(self, im):
		im = im.transpose(2, 0, 1)
		im = torch.from_numpy(im).float()

		return im

	def resize_im0(self, im):
		try:
			im = cv2.resize(im, (992, 992), interpolation=cv2.INTER_LINEAR)
			# im = cv2.resize(im, (496, 496), interpolation=cv2.INTER_LINEAR)
		except:
			pass

		return im


	def resize_lbl(self, lbl):
		lbl = lbl/[1024, 1024]*[992, 992]
		# lbl = lbl/[960, 1024]*[496, 496]
		return lbl

	def fiducal_points_lbl(self, fiducial_points, interval):

		fiducial_point_gaps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]  # POINTS NUM: 61, 31, 21, 16, 13, 11, 7, 6, 5, 4, 3, 2
		fiducial_points = fiducial_points[::fiducial_point_gaps[self.row_gap], ::fiducial_point_gaps[self.col_gap], :]
		interval = interval * [fiducial_point_gaps[self.col_gap], fiducial_point_gaps[self.row_gap]]
		return fiducial_points, interval










	def check_item_vis(self, im, lbl, idx):
		'''
		im : distorted image   # HWC 
		lbl : fiducial_points  # 61*61*2 
		'''
		# im=np.uint8(im)
		im = Image.fromarray(im)
		im.convert('RGB').save("./data_vis/img_vis{}.png".format(idx))
		
		fig, ax = plt.subplots(figsize = (10.24,10.24),facecolor='white')
		ax.imshow(im)
		ax.scatter(lbl[:,:,0].flatten(),lbl[:,:,1].flatten(),s=1.2,c='red',alpha=1)
		ax.axis('off')
		plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
		# plt.tight_layout()
		plt.savefig('./data_vis/point_vis{}.png'.format(idx))
		plt.close()

	def resize_im1(self, im, bfreq, hpf):
		try:
			im = cv2.resize(im, (992, 992), interpolation=cv2.INTER_LINEAR)
			# im = cv2.resize(im, (496, 496), interpolation=cv2.INTER_LINEAR)
		except:
			pass
	
		# freq = np.fft.fft2(im,axes=(0,1))
		# freq = np.fft.fftshift(freq)

		# rhpf = bfreq + hpf * freq
		# img_rhpf = np.abs(np.fft.ifft2(rhpf,axes=(0,1)))
		# img_rhpf = np.clip(img_rhpf,0,255) #会产生一些过大值需要截断
		# img_rhpf = img_rhpf.astype('uint8')

		# return img_rhpf
		return im
	


