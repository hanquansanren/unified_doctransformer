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



# def get_data_path(name):
# 	"""Extract path to data from config file.

# 	Args:
# 		name (str): The name of the dataset.

# 	Returns:
# 		(str): The path to the root directory containing the dataset.
# 	"""
# 	with open('../xgw/segmentation/config.json') as f:
# 		js = f.read()
# 	# js = open('config.json').read()
# 	data = json.loads(js)
# 	return os.path.expanduser(data[name]['data_path'])

# def getDatasets(dir):
# 	return os.listdir(dir)
'''
Resize the input image into 1024x960 (zooming in or out along the longest side and keeping the aspect ration, then filling zero for padding. )
'''
# def resize_image(origin_img, long_edge=1024, short_edge=960):
# 	# long_edge, short_edge = 2048, 1920
# 	# long_edge, short_edge = 1024, 960
# 	# long_edge, short_edge = 512, 480

# 	im_lr = origin_img.shape[0]
# 	im_ud = origin_img.shape[1]
# 	new_img = np.zeros([long_edge, short_edge, 3], dtype=np.uint8)
# 	new_shape = new_img.shape[:2]
# 	if im_lr > im_ud:
# 		img_shrink, base_img_shrink = long_edge, long_edge
# 		im_ud = int(im_ud / im_lr * base_img_shrink)
# 		im_ud += 32-im_ud%32
# 		im_ud = min(im_ud, short_edge)
# 		im_lr = img_shrink
# 		origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)
# 		new_img[:, (new_shape[1]-im_ud)//2:new_shape[1]-(new_shape[1]-im_ud)//2] = origin_img
# 		# mask = np.full(new_shape, 255, dtype='uint8')
# 		# mask[:, (new_shape[1] - im_ud) // 2:new_shape[1] - (new_shape[1] - im_ud) // 2] = 0
# 	else:
# 		img_shrink, base_img_shrink = short_edge, short_edge
# 		im_lr = int(im_lr / im_ud * base_img_shrink)
# 		im_lr += 32-im_lr%32
# 		im_lr = min(im_lr, long_edge)
# 		im_ud = img_shrink
# 		origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)
# 		new_img[(new_shape[0] - im_lr) // 2:new_shape[0] - (new_shape[0] - im_lr) // 2, :] = origin_img
# 	return new_img

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
		# self.rotate_root=os.path.join(self.root, 'digital', 'rotate/')
		# self.perspective_root=os.path.join(self.root, 'digital', 'perspective/')
		# self.random_root=os.path.join(self.root, 'digital', 'random/')
		# self.curved_root=os.path.join(self.root, 'digital', 'curved/')
		# self.fold_root=os.path.join(self.root, 'digital', 'fold/')
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
			im_name = self.images[self.mode][item]
			im_path = pjoin(self.scan_root, im_name)
			deform_type1=np.random.choice(self.deform_type_list,p=[0.5,0.5])
			print(deform_type1)
			# deform_type2=np.random.choice(self.deform_type_list,p=[0.5,0.5])
			# print(deform_type2)
			# im = np.uint8(im)
			d1,lbl1,itv1=get_syn_image(path=im_path, bg_path=self.bg_path,save_path=self.save_path,deform_type=deform_type1)
			# d2,lbl2,itv2=get_syn_image(path=im_path, bg_path=self.bg_path,save_path=self.save_path,deform_type=deform_type2)
			print('finish')
			d1 = Image.fromarray(d1)
			d1.convert('RGB').save("./data_vis/img111.png")
			# d2 = Image.fromarray(d2)
			# d2.convert('RGB').save("./data_vis/img222_{}.png".format(00))

			# im = self.resize_im1(im, self.bfreq, self.hpf)
			d1=self.resize_im0(d1)
			# d1 = d1.transpose(2, 0, 1)

			# lbl1 = self.resize_lbl(lbl1)
			# lbl1, itv1 = self.fiducal_points_lbl(lbl1, itv1)
			# lbl1 = lbl1.transpose(2, 0, 1)

			d1 = torch.from_numpy(d1).float()
			lbl1 = torch.from_numpy(lbl1).float()
			itv1 = torch.from_numpy(itv1).float()

			# if self.is_return_img_name:
			# 	return im, lbl, interval, im_name

			return d1, lbl1, itv1

	def transform_im(self, im):
		im = im.transpose(2, 0, 1)
		im = torch.from_numpy(im).float()

		return im

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

	def resize_im0(self, im):
		try:
			im = cv2.resize(im, (992, 992), interpolation=cv2.INTER_LINEAR)
			# im = cv2.resize(im, (496, 496), interpolation=cv2.INTER_LINEAR)
		except:
			pass

		return im


	def resize_lbl(self, lbl):
		lbl = lbl/[960, 1024]*[992, 992]
		# lbl = lbl/[960, 1024]*[496, 496]
		return lbl

	def fiducal_points_lbl(self, fiducial_points, segment):

		fiducial_point_gaps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]  # POINTS NUM: 61, 31, 21, 16, 13, 11, 7, 6, 5, 4, 3, 2
		fiducial_points = fiducial_points[::fiducial_point_gaps[self.row_gap], ::fiducial_point_gaps[self.col_gap], :]
		segment = segment * [fiducial_point_gaps[self.col_gap], fiducial_point_gaps[self.row_gap]]
		return fiducial_points, segment


	# def get_syn_image(self, path, bg_path, save_path, deform_type):

	# 	save_suffix = str.split(path, '/')[-2] # 'new'

	# 	all_bgImg_idx = os.listdir(bg_path)
	# 	global begin_train
	# 	begin_train = time.time()
	# 	fiducial_points = 61	# 31
	# 	process_pool = Pool(4)

	# 	save_suffix = str.split(path, '/')[-2]+str.split(path, '/')[-1][0:4] # 'new'
	# 	try:
	# 		if deform_type=='fold':
	# 			saveFold = perturbed(path, all_bgImg_idx, save_path, save_suffix)
	# 			repeat_time = min(max(round(np.random.normal(12, 4)), 1), 18) #随机折叠次数
	# 			# repeat_time = min(max(round(np.random.normal(8, 4)), 1), 12)	# random.randint(1, 2)		# min(max(round(np.random.normal(8, 4)), 1), 12)
	# 			process_pool.apply_async(func=saveFold.save_img, args=('fold', repeat_time, fiducial_points, 'relativeShift_v2'))
	# 		elif deform_type=='curve':
	# 			saveCurve = perturbed(path, all_bgImg_idx, save_path, save_suffix)	
	# 			repeat_time = min(max(round(np.random.normal(8, 4)), 1), 13) #随机弯曲次数
	# 			# repeat_time = min(max(round(np.random.normal(6, 4)), 1), 10)
	# 			process_pool.apply_async(func=saveCurve.save_img, args=('curve', repeat_time, fiducial_points, 'relativeShift_v2'))		
	# 		else:
	# 			print('error type')
	# 	except BaseException as err:
	# 		print(err)
	# 	# print('end')

	# 	process_pool.close()
	# 	process_pool.join()
	


