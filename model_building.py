import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T
import scipy.io as sio

# All data parameters import
from utils.params import ParamsPack
param_pack = ParamsPack()

from backbone_nets import mobilenetv2_backbone
from backbone_nets.pointnet_backbone import MLP_for, MLP_rev
from loss_definition import ParamLoss, WingLoss

import time

# Image-to-parameter
class I2P(nn.Module):
	def __init__(self, args):
		super(I2P, self).__init__()
		self.args = args
		# backbone definition
		self.backbone = getattr(mobilenetv2_backbone, args.arch)()

	def forward(self,input, target):
		"""Training time forward"""
		_3D_attr, avgpool = self.backbone(input)
		_3D_attr_GT = target.type(torch.cuda.FloatTensor)
		return _3D_attr, _3D_attr_GT, avgpool

	def forward_test(self, input):
		""" Testing time forward."""
		_3D_attr, avgpool = self.backbone(input)
		return _3D_attr, avgpool

# Main model SynergyNet definition
class SynergyNet(nn.Module):
	def __init__(self, args):
		super(SynergyNet, self).__init__()
		self.triangles = sio.loadmat('./3dmm_data/tri.mat')['tri'] -1
		self.triangles = torch.Tensor(self.triangles.astype(np.int)).long().cuda()
		self.img_size = args.img_size
		# Image-to-parameter
		self.I2P = I2P(args)
		# Forward
		self.forwardDirection = MLP_for(68)
		# Reverse
		self.reverseDirection = MLP_rev(68)
		self.LMKLoss_3D = WingLoss()
		self.ParamLoss = ParamLoss()
		
		self.loss = {'loss_LMK_f0':0.0,
					'loss_LMK_pointNet': 0.0,
					'loss_Param_In':0.0,
					'loss_PointResidual':0.0,
					'loss_Param_S2': 0.0,
					'loss_Param_S1S2': 0.0,
					}

		self.register_buffer('param_mean', torch.Tensor(param_pack.param_mean).cuda(non_blocking=True))
		self.register_buffer('param_std', torch.Tensor(param_pack.param_std).cuda(non_blocking=True))
		self.register_buffer('w_shp', torch.Tensor(param_pack.w_shp).cuda(non_blocking=True))
		self.register_buffer('u', torch.Tensor(param_pack.u).cuda(non_blocking=True))
		self.register_buffer('w_exp', torch.Tensor(param_pack.w_exp).cuda(non_blocking=True))
		# unsued texture parameters
		self.register_buffer('u_tex', torch.Tensor(param_pack.u_tex).cuda(non_blocking=True))
		self.register_buffer('w_tex', torch.Tensor(param_pack.w_tex).cuda(non_blocking=True))

		# If doing only offline evaluation, use these
		# self.u_base = torch.Tensor(param_pack.u_base).cuda(non_blocking=True)
		# self.w_shp_base = torch.Tensor(param_pack.w_shp_base).cuda(non_blocking=True)
		# self.w_exp_base = torch.Tensor(param_pack.w_exp_base).cuda(non_blocking=True)

		# Online training needs these to parallel
		self.register_buffer('u_base', torch.Tensor(param_pack.u_base).cuda(non_blocking=True))
		self.register_buffer('w_shp_base', torch.Tensor(param_pack.w_shp_base).cuda(non_blocking=True))
		self.register_buffer('w_exp_base', torch.Tensor(param_pack.w_exp_base).cuda(non_blocking=True))
		self.keypoints = torch.Tensor(param_pack.keypoints).long()
 
		self.data_param = [self.param_mean, self.param_std, self.w_shp_base, self.u_base, self.w_exp_base]

	def forward(self):
		# TODO
		pass

	def forward_test(self, input):
		"""test time forward"""
		_3D_attr, _ = self.I2P.forward_test(input)
		return _3D_attr

	def get_losses(self):
		return self.loss.keys()


if __name__ == '__main__':
	pass