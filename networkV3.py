'''
2022/9/30
Weiguang Zhang
V3 means DDCP+FDRNet
'''

import math
import pickle
import torch.nn as nn
import torch
# from xgw.dewarp.fiducial_points.networks.resnet import *
import torch.nn.init as tinit
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3,  stride=stride, padding=1)

def dilation_conv_bn_act(in_channels, out_dim, act_fn, BatchNorm, dilation=4):
	model = nn.Sequential(
		nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
		BatchNorm(out_dim),
		# nn.BatchNorm2d(out_dim),
		act_fn,
	)
	return model

def dilation_conv(in_channels, out_dim, stride=1, dilation=4, groups=1):
	model = nn.Sequential(
		nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups),
	)
	return model

class ResidualBlockWithDilatedV1(nn.Module):
	def __init__(self, in_channels, out_channels, BatchNorm, stride=1, downsample=None, is_activation=True, is_top=False, is_dropout=False):
		super(ResidualBlockWithDilatedV1, self).__init__()
		self.stride = stride
		self.is_activation = is_activation
		self.downsample = downsample
		self.is_top = is_top
		self.relu = nn.ReLU(inplace=True)
		self.bn1 = BatchNorm(out_channels)
		self.bn2 = BatchNorm(out_channels)
		self.is_dropout = is_dropout
		self.drop_out = nn.Dropout2d(p=0.2)

		if self.stride != 1 or self.is_top:
			self.conv1 = conv3x3(in_channels, out_channels, self.stride)
		else:
			self.conv1 = dilation_conv(in_channels, out_channels, dilation=3)		# 3
		
		# self.bn1 = nn.BatchNorm2d(out_channels)
		if self.stride != 1 or self.is_top:
			self.conv2 = conv3x3(out_channels, out_channels)
		else:
			self.conv2 = dilation_conv(out_channels, out_channels, dilation=3)		# 1

	def forward(self, x):
		residual = x

		out1 = self.relu(self.bn1(self.conv1(x)))
		# if self.is_dropout:
		# 	out1 = self.drop_out(out1)
		out = self.bn2(self.conv2(out1))
		if self.downsample is not None:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)

		return out

class ResNetV2StraightV2(nn.Module):
	def __init__(self, num_filter, map_num, BatchNorm, block_nums=[3, 4, 6, 3], block=ResidualBlockWithDilatedV1, stride=[1, 2, 2, 2], dropRate=[0.2, 0.2, 0.2, 0.2], is_sub_dropout=False):
		super(ResNetV2StraightV2, self).__init__()
		self.in_channels = num_filter * map_num[0]
		self.dropRate = dropRate
		self.stride = stride
		self.is_sub_dropout = is_sub_dropout
		# self.is_dropout = is_dropout
		self.drop_out = nn.Dropout2d(p=dropRate[0])
		self.drop_out_2 = nn.Dropout2d(p=dropRate[1])
		self.drop_out_3 = nn.Dropout2d(p=dropRate[2])
		self.drop_out_4 = nn.Dropout2d(p=dropRate[3]) 		# add
		self.relu = nn.ReLU(inplace=True)

		self.block_nums = block_nums
		self.layer1 = self.blocklayer(block, num_filter * map_num[0], self.block_nums[0], BatchNorm, stride=self.stride[0])
		self.layer2 = self.blocklayer(block, num_filter * map_num[1], self.block_nums[1], BatchNorm, stride=self.stride[1])
		self.layer3 = self.blocklayer(block, num_filter * map_num[2], self.block_nums[2], BatchNorm, stride=self.stride[2])
		self.layer4 = self.blocklayer(block, num_filter * map_num[3], self.block_nums[3], BatchNorm, stride=self.stride[3])

	def blocklayer(self, block, out_channels, block_nums, BatchNorm, stride):
		downsample = None
		if (stride != 1) or (self.in_channels != out_channels): # 只有layer234才会用下采样，layer1不需要下采样
			downsample = nn.Sequential(
				conv3x3(self.in_channels, out_channels, stride=stride), # default, s=2, p=1
				BatchNorm(out_channels)) 

		layers = []
		layers.append(block(self.in_channels, out_channels, BatchNorm, stride, downsample=downsample, is_top=True))
		self.in_channels = out_channels
		for i in range(1, block_nums):
			layers.append(block(out_channels, out_channels, BatchNorm, is_top=False))#后几层默认采用S=1，且不需要下采样
		return nn.Sequential(*layers)

	def forward(self, x, is_skip=False):

		out1 = self.layer1(x)

		out2 = self.layer2(out1)

		out3 = self.layer3(out2)

		out4 = self.layer4(out3)

		return out4

class model_handlebar(nn.Module):
	'''
	主模型入口
	'''
	def __init__(self, n_classes, num_filter, architecture, BatchNorm='GN', in_channels=3):
		# in_channels=3=RGB,输入图像的通道数。
		# n_classes=2, num_filter=32, in_channels=3
		# architecture=DilatedResnet # 传入了类名
		super(model_handlebar, self).__init__()
		self.in_channels = in_channels # 3
		self.n_classes = n_classes
		self.num_filter = num_filter
		if BatchNorm == 'IN':
			BatchNorm = nn.InstanceNorm2d
		elif BatchNorm == 'BN':
			BatchNorm = nn.BatchNorm2d
		elif BatchNorm == 'GN':
			BatchNorm = nn.GroupNorm

		# 这里实现了类DilatedResnet的实例化
		self.dilated_unet = architecture(self.n_classes, self.num_filter, BatchNorm, in_channels=self.in_channels)

	def forward(self, images1, images2=None, w_im=None):
		return self.dilated_unet(images1, images2, w_im)
		# 参数传入forward之中


class DilatedResnet(nn.Module):
	def __init__(self, n_classes, num_filter, BatchNorm, in_channels=3):
		super(DilatedResnet, self).__init__()
		self.in_channels = in_channels # 图片的输入通道
		self.n_classes = n_classes # 标准控制点的通道数（维度），因为只有xy，所以为2
		self.num_filter = num_filter #卷积核个数=下一层的通道数
		# act_fn = nn.PReLU()
		act_fn = nn.ReLU(inplace=True)
		# act_fn = nn.LeakyReLU(0.2)

		map_num = [1, 2, 4, 8, 16] # 通道数控制器，乘以32后分别是：
		# [32, 64, 128, 256, 512]


		print("\n------load DilatedResnet------\n")

		self.resnet_head = nn.Sequential(

			nn.Conv2d(self.in_channels, self.num_filter * map_num[0], kernel_size=3, stride=2, padding=1), # 3 -> 32
			# nn.InstanceNorm2d(self.num_filter * map_num[0]),
			# BatchNorm(1, self.num_filter * map_num[0]),
			BatchNorm(self.num_filter * map_num[0]),
			# nn.BatchNorm2d(self.num_filter * map_num[0]),
			act_fn,
			# nn.Dropout(p=0.2),
			# nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
			nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=3, stride=2, padding=1), # 32 -> 32
			BatchNorm(self.num_filter * map_num[0]),
			act_fn,
			# nn.Dropout(p=0.2),
		)

		self.resnet_down = ResNetV2StraightV2(num_filter, map_num, BatchNorm, block_nums=[3, 4, 6, 3], block=ResidualBlockWithDilatedV1, dropRate=[0, 0, 0, 0], is_sub_dropout=False)

		map_num_i = 3
		self.bridge_1 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								 act_fn, BatchNorm, dilation=1),
			# conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn),
		)
		self.bridge_2 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=2),
		)
		self.bridge_3 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=5),
		)
		self.bridge_4 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=8),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=3),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=2),
		)
		self.bridge_5 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=12),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=7),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=4),
		)
		self.bridge_6 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=18),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=12),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=6),
		)

		self.bridge_concate = nn.Sequential(
			nn.Conv2d(self.num_filter * map_num[map_num_i] * 6, self.num_filter * map_num[2], kernel_size=1, stride=1, padding=0),
			# BatchNorm(GN_num, self.num_filter * map_num[4]),
			BatchNorm(self.num_filter * map_num[2]),
			# nn.BatchNorm2d(self.num_filter * map_num[4]),
			act_fn,
		)

		self.out_regress = nn.Sequential(
			nn.Conv2d(self.num_filter * map_num[2], self.num_filter * map_num[0], kernel_size=3, stride=1, padding=1),
			# input 128, output 32
			BatchNorm(self.num_filter * map_num[0]),
			nn.PReLU(),
			nn.Conv2d(self.num_filter * map_num[0], n_classes, kernel_size=3, stride=1, padding=1),
			# input 32, output 2
		)


		self.segment_regress = nn.Linear(self.num_filter * map_num[2]*31*31, 2)

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules(): # 遍历所有的network结构
			if isinstance(m, nn.Conv2d):
				tinit.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				# tinit.xavier_normal_(m.weight, gain=0.2)
				if m.bias is not None:
					tinit.constant_(m.bias, 0)
			elif isinstance(m, nn.ConvTranspose2d):
				assert m.kernel_size[0] == m.kernel_size[1]
				tinit.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				# tinit.xavier_normal_(m.weight, gain=0.2)
			elif isinstance(m, nn.Linear):
				tinit.normal_(m.weight, 0, 0.01)
				# m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					tinit.constant_(m.bias, 0)

	# def cat(self, trans, down):
	# 	return torch.cat([trans, down], dim=1)

	def forward(self, images1, images2=None, w_im=None):
		'''part 1'''
		resnet_head1 = self.resnet_head(images1) # 图示的第一层
		resnet_down1 = self.resnet_down(resnet_head1) # 图示的中四层

		bridge_11 = self.bridge_1(resnet_down1)
		bridge_12 = self.bridge_2(resnet_down1)
		bridge_13 = self.bridge_3(resnet_down1)
		bridge_14 = self.bridge_4(resnet_down1)
		bridge_15 = self.bridge_5(resnet_down1)
		bridge_16 = self.bridge_6(resnet_down1)
		bridge_concate1 = torch.cat([bridge_11, bridge_12, bridge_13, bridge_14, bridge_15, bridge_16], dim=1) 
		# output: torch.Size([1, 1536, 32, 32])
		bridge1 = self.bridge_concate(bridge_concate1)# torch.Size([1, 128, 31, 31]) # 第六层，输出层 256->128
		out_regress1 = self.out_regress(bridge1) # torch.Size([1, 2, 31, 31]) #第七八层，包含两次卷积 128->32->2 
		
		# '''part 2'''
		# resnet_head2 = self.resnet_head(images2) # 图示的第一层
		# resnet_down2 = self.resnet_down(resnet_head2) # 图示的中四层

		# bridge_21 = self.bridge_1(resnet_down2)
		# bridge_22 = self.bridge_2(resnet_down2)
		# bridge_23 = self.bridge_3(resnet_down2)
		# bridge_24 = self.bridge_4(resnet_down2)
		# bridge_25 = self.bridge_5(resnet_down2)
		# bridge_26 = self.bridge_6(resnet_down2)
		# bridge_concate2 = torch.cat([bridge_21, bridge_22, bridge_23, bridge_24, bridge_25, bridge_26], dim=1)
		# bridge2 = self.bridge_concate(bridge_concate2)
		# out_regress2 = self.out_regress(bridge2)

		# '''part 3'''
		# resnet_head3 = self.resnet_head(w_im) # 图示的第一层
		# resnet_down3 = self.resnet_down(resnet_head3) # 图示的中四层

		# bridge_31 = self.bridge_1(resnet_down3)
		# bridge_32 = self.bridge_2(resnet_down3)
		# bridge_33 = self.bridge_3(resnet_down3)
		# bridge_34 = self.bridge_4(resnet_down3)
		# bridge_35 = self.bridge_5(resnet_down3)
		# bridge_36 = self.bridge_6(resnet_down3)
		# bridge_concate3 = torch.cat([bridge_31, bridge_32, bridge_33, bridge_34, bridge_35, bridge_36], dim=1)
		# bridge3 = self.bridge_concate(bridge_concate3) # torch.Size([1, 128, 31, 31]) # 第六层，输出层 256->128
		# out_regress3 = self.out_regress(bridge3)	# torch.Size([1, 2, 31, 31]) #第七八层，包含两次卷积 128->32->2
		
		# '''part 6'''
		# resnet_head4 = self.resnet_head(w_im) # 图示的第一层
		# resnet_down4 = self.resnet_down(resnet_head4) # 图示的中四层

		# bridge_41 = self.bridge_1(resnet_down4)
		# bridge_42 = self.bridge_2(resnet_down4)
		# bridge_43 = self.bridge_3(resnet_down4)
		# bridge_44 = self.bridge_4(resnet_down4)
		# bridge_45 = self.bridge_5(resnet_down4)
		# bridge_46 = self.bridge_6(resnet_down4)
		# bridge_concate4 = torch.cat([bridge_41, bridge_42, bridge_43, bridge_44, bridge_45, bridge_46], dim=1)
		# bridge4 = self.bridge_concate(bridge_concate4) # torch.Size([1, 128, 31, 31]) # 第六层，输出层 256->128
		# out_regress4 = self.out_regress(bridge4)	# torch.Size([1, 2, 31, 31]) #第七八层，包含两次卷积 128->32->2		

		# '''part 7'''
		# resnet_head5 = self.resnet_head(w_im) # 图示的第一层
		# resnet_down5 = self.resnet_down(resnet_head5) # 图示的中四层

		# bridge_51 = self.bridge_1(resnet_down5)
		# bridge_52 = self.bridge_2(resnet_down5)
		# bridge_53 = self.bridge_3(resnet_down5)
		# bridge_54 = self.bridge_4(resnet_down5)
		# bridge_55 = self.bridge_5(resnet_down5)
		# bridge_56 = self.bridge_6(resnet_down5)
		# bridge_concate5 = torch.cat([bridge_51, bridge_52, bridge_53, bridge_54, bridge_55, bridge_56], dim=1)
		# bridge5 = self.bridge_concate(bridge_concate5) # torch.Size([1, 128, 31, 31]) # 第六层，输出层 256->128
		# out_regress5 = self.out_regress(bridge5)	# torch.Size([1, 2, 31, 31]) #第七八层，包含两次卷积 128->32->2		
		
		
		
		# return out_regress1, out_regress2, out_regress3
		# return out_regress1, out_regress2
		return out_regress1





























class DilatedResnet_for_test_single_image(nn.Module):

	def __init__(self, n_classes, num_filter, BatchNorm, in_channels=3):
		super(DilatedResnet_for_test_single_image, self).__init__()
		self.in_channels = in_channels # 图片的输入通道
		self.n_classes = n_classes # 标准控制点的通道数（维度），因为只有xy，所以为2
		self.num_filter = num_filter #卷积核个数=下一层的通道数
		# act_fn = nn.PReLU()
		act_fn = nn.ReLU(inplace=True)
		# act_fn = nn.LeakyReLU(0.2)

		map_num = [1, 2, 4, 8, 16] # 通道数控制器，乘以32后分别是：
		# [32, 64, 128, 256, 512]


		print("\n------load DilatedResnet------\n")

		self.resnet_head = nn.Sequential(

			nn.Conv2d(self.in_channels, self.num_filter * map_num[0], kernel_size=3, stride=2, padding=1), # 3 -> 32
			# nn.InstanceNorm2d(self.num_filter * map_num[0]),
			# BatchNorm(1, self.num_filter * map_num[0]),
			BatchNorm(self.num_filter * map_num[0]),
			# nn.BatchNorm2d(self.num_filter * map_num[0]),
			act_fn,
			# nn.Dropout(p=0.2),
			# nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
			nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=3, stride=2, padding=1), # 32 -> 32
			BatchNorm(self.num_filter * map_num[0]),
			act_fn,
			# nn.Dropout(p=0.2),
		)

		self.resnet_down = ResNetV2StraightV2(num_filter, map_num, BatchNorm, block_nums=[3, 4, 6, 3], block=ResidualBlockWithDilatedV1, dropRate=[0, 0, 0, 0], is_sub_dropout=False)

		map_num_i = 3
		self.bridge_1 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								 act_fn, BatchNorm, dilation=1),
			# conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn),
		)
		self.bridge_2 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=2),
		)
		self.bridge_3 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=5),
		)
		self.bridge_4 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=8),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=3),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=2),
		)
		self.bridge_5 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=12),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=7),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=4),
		)
		self.bridge_6 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=18),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=12),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=6),
		)

		self.bridge_concate = nn.Sequential(
			nn.Conv2d(self.num_filter * map_num[map_num_i] * 6, self.num_filter * map_num[2], kernel_size=1, stride=1, padding=0),
			# BatchNorm(GN_num, self.num_filter * map_num[4]),
			BatchNorm(self.num_filter * map_num[2]),
			# nn.BatchNorm2d(self.num_filter * map_num[4]),
			act_fn,
		)

		self.out_regress = nn.Sequential(
			nn.Conv2d(self.num_filter * map_num[2], self.num_filter * map_num[0], kernel_size=3, stride=1, padding=1),
			# input 128, output 32
			BatchNorm(self.num_filter * map_num[0]),
			nn.PReLU(),
			nn.Conv2d(self.num_filter * map_num[0], n_classes, kernel_size=3, stride=1, padding=1),
			# input 32, output 2
		)


		self.segment_regress = nn.Linear(self.num_filter * map_num[2]*31*31, 2)

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules(): # 遍历所有的network结构
			if isinstance(m, nn.Conv2d):
				tinit.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				# tinit.xavier_normal_(m.weight, gain=0.2)
				if m.bias is not None:
					tinit.constant_(m.bias, 0)
			elif isinstance(m, nn.ConvTranspose2d):
				assert m.kernel_size[0] == m.kernel_size[1]
				tinit.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				# tinit.xavier_normal_(m.weight, gain=0.2)
			elif isinstance(m, nn.Linear):
				tinit.normal_(m.weight, 0, 0.01)
				# m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					tinit.constant_(m.bias, 0)

	# def cat(self, trans, down):
	# 	return torch.cat([trans, down], dim=1)

	def forward(self, images1, images2, w_im):
		resnet_head1 = self.resnet_head(images1) # 图示的第一层
		resnet_down1 = self.resnet_down(resnet_head1) # 图示的中四层

		bridge_11 = self.bridge_1(resnet_down1)
		bridge_12 = self.bridge_2(resnet_down1)
		bridge_13 = self.bridge_3(resnet_down1)
		bridge_14 = self.bridge_4(resnet_down1)
		bridge_15 = self.bridge_5(resnet_down1)
		bridge_16 = self.bridge_6(resnet_down1)
		bridge_concate1 = torch.cat([bridge_11, bridge_12, bridge_13, bridge_14, bridge_15, bridge_16], dim=1) 
		# output: torch.Size([1, 1536, 32, 32])
		bridge1 = self.bridge_concate(bridge_concate1)# torch.Size([1, 128, 31, 31]) # 第六层，输出层 256->128
		out_regress1 = self.out_regress(bridge1) # torch.Size([1, 2, 31, 31]) #第七八层，包含两次卷积 128->32->2 
		
		return out_regress1
