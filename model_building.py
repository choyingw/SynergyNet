import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T
import scipy.io as sio

# All data parameters import
from utils.params import ParamsPack
param_pack = ParamsPack()

from backbone_nets import resnet_backbone
from backbone_nets import mobilenetv1_backbone
from backbone_nets import mobilenetv2_backbone
from backbone_nets import ghostnet_backbone
from backbone_nets.pointnet_backbone import MLP_for, MLP_rev
from loss_definition import ParamLoss, WingLoss

from backbone_nets.ResNeSt import resnest50, resnest101
import time
from utils.inference import predict_sparseVert, predict_denseVert, predict_pose, crop_img
from FaceBoxes import FaceBoxes
import cv2
import types

def parse_param_62(param):
	"""Work for only tensor"""
	p_ = param[:, :12].reshape(-1, 3, 4)
	p = p_[:, :, :3]
	offset = p_[:, :, -1].reshape(-1, 3, 1)
	alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
	alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
	return p, offset, alpha_shp, alpha_exp

# Image-to-parameter
class I2P(nn.Module):
	def __init__(self, args):
		super(I2P, self).__init__()
		self.args = args
		# backbone definition
		if 'mobilenet_v2' in self.args.arch:
			self.backbone = getattr(mobilenetv2_backbone, args.arch)(pretrained=False)
		elif 'mobilenet' in self.args.arch:
			self.backbone = getattr(mobilenetv1_backbone, args.arch)()		
		elif 'resnet' in self.args.arch:
			self.backbone = getattr(resnet_backbone, args.arch)(pretrained=False)
		elif 'ghostnet' in self.args.arch:
			self.backbone = getattr(ghostnet_backbone, args.arch)()
		elif 'resnest' in self.args.arch:
			self.backbone = resnest50()
		else:
			raise RuntimeError("Please choose [mobilenet_v2, mobilenet_1, resnet50, or ghostnet]")

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
		self.triangles = torch.Tensor(self.triangles.astype(np.int64)).long().cuda()
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
					'loss_Param_S2': 0.0,
					'loss_Param_S1S2': 0.0,
					}

		self.register_buffer('param_mean', torch.Tensor(param_pack.param_mean).cuda(non_blocking=True))
		self.register_buffer('param_std', torch.Tensor(param_pack.param_std).cuda(non_blocking=True))
		self.register_buffer('w_shp', torch.Tensor(param_pack.w_shp).cuda(non_blocking=True))
		self.register_buffer('u', torch.Tensor(param_pack.u).cuda(non_blocking=True))
		self.register_buffer('w_exp', torch.Tensor(param_pack.w_exp).cuda(non_blocking=True))

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

	def reconstruct_vertex_62(self, param, whitening=True, dense=False, transform=True, lmk_pts=68):
		"""
		Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
		dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
		image coordinate space, but without alignment caused by face cropping.
		transform: whether transform to image space
		Working with batched tensors. Using Fortan-type reshape.
		"""

		if whitening:
			if param.shape[1] == 62:
				param_ = param * self.param_std[:62] + self.param_mean[:62]
			else:
				raise RuntimeError('length of params mismatch')

		p, offset, alpha_shp, alpha_exp = parse_param_62(param_)

		if dense:
			
			vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).contiguous().view(-1, 53215, 3).transpose(1,2) + offset
			
			if transform: 
				# transform to image coordinate space
				vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

		else:
			"""For 68 pts"""
			vertex = p @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp).contiguous().view(-1, lmk_pts, 3).transpose(1,2) + offset

			if transform: 
				# transform to image coordinate space
				vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

		return vertex

	def forward(self, input, target):
		_3D_attr, _3D_attr_GT, avgpool = self.I2P(input, target)
			
		vertex_lmk = self.reconstruct_vertex_62(_3D_attr, dense=False)
		vertex_GT_lmk = self.reconstruct_vertex_62(_3D_attr_GT, dense=False)
		self.loss['loss_LMK_f0'] = 0.05 *self.LMKLoss_3D(vertex_lmk, vertex_GT_lmk, kp=True)		
		self.loss['loss_Param_In'] = 0.02 * self.ParamLoss(_3D_attr, _3D_attr_GT)

		point_residual = self.forwardDirection(vertex_lmk, avgpool, _3D_attr[:,12:52], _3D_attr[:,52:62])
		vertex_lmk = vertex_lmk + 0.05 * point_residual
		self.loss['loss_LMK_pointNet'] = 0.05 * self.LMKLoss_3D(vertex_lmk, vertex_GT_lmk, kp=True)

		_3D_attr_S2 = self.reverseDirection(vertex_lmk)
		self.loss['loss_Param_S2'] = 0.02 * self.ParamLoss(_3D_attr_S2, _3D_attr_GT, mode='only_3dmm')
		self.loss['loss_Param_S1S2'] = 0.001 * self.ParamLoss(_3D_attr_S2, _3D_attr, mode='only_3dmm')

		return self.loss

	def forward_test(self, input):
		"""test time forward"""
		_3D_attr, _ = self.I2P.forward_test(input)
		return _3D_attr

	def get_losses(self):
		return self.loss.keys()


# Main model SynergyNet definition
class WrapUpSynergyNet(nn.Module):
	def __init__(self):
		super(WrapUpSynergyNet, self).__init__()
		self.triangles = sio.loadmat('./3dmm_data/tri.mat')['tri'] -1
		self.triangles = torch.Tensor(self.triangles.astype(np.int64)).long()
		args = types.SimpleNamespace()
		args.arch = 'mobilenet_v2'
		args.checkpoint_fp = 'pretrained/best.pth.tar'

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
					'loss_Param_S2': 0.0,
					'loss_Param_S1S2': 0.0,
					}

		self.register_buffer('param_mean', torch.Tensor(param_pack.param_mean))
		self.register_buffer('param_std', torch.Tensor(param_pack.param_std))
		self.register_buffer('w_shp', torch.Tensor(param_pack.w_shp))
		self.register_buffer('u', torch.Tensor(param_pack.u))
		self.register_buffer('w_exp', torch.Tensor(param_pack.w_exp))

		# Online training needs these to parallel
		self.register_buffer('u_base', torch.Tensor(param_pack.u_base))
		self.register_buffer('w_shp_base', torch.Tensor(param_pack.w_shp_base))
		self.register_buffer('w_exp_base', torch.Tensor(param_pack.w_exp_base))
		self.keypoints = torch.Tensor(param_pack.keypoints).long()
 
		self.data_param = [self.param_mean, self.param_std, self.w_shp_base, self.u_base, self.w_exp_base]

		try:
			print("loading weights from ", args.checkpoint_fp)
			self.load_weights(args.checkpoint_fp)
		except:
			pass
		self.eval()

	def reconstruct_vertex_62(self, param, whitening=True, dense=False, transform=True, lmk_pts=68):
		"""
		Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
		dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
		image coordinate space, but without alignment caused by face cropping.
		transform: whether transform to image space
		Working with batched tensors. Using Fortan-type reshape.
		"""

		if whitening:
			if param.shape[1] == 62:
				param_ = param * self.param_std[:62] + self.param_mean[:62]
			else:
				raise RuntimeError('length of params mismatch')

		p, offset, alpha_shp, alpha_exp = parse_param_62(param_)

		if dense:
			
			vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).contiguous().view(-1, 53215, 3).transpose(1,2) + offset
			
			if transform: 
				# transform to image coordinate space
				vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

		else:
			"""For 68 pts"""
			vertex = p @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp).contiguous().view(-1, lmk_pts, 3).transpose(1,2) + offset

			if transform: 
				# transform to image coordinate space
				vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

		return vertex

	def forward_test(self, input):
		"""test time forward"""
		_3D_attr, _ = self.I2P.forward_test(input)
		return _3D_attr

	def load_weights(self, path):
		model_dict = self.state_dict()
		checkpoint = torch.load(path, map_location=lambda storage, loc: storage)['state_dict']

		# because the model is trained by multiple gpus, prefix 'module' should be removed
		for k in checkpoint.keys():
			model_dict[k.replace('module.', '')] = checkpoint[k]

		self.load_state_dict(model_dict, strict=False)


	def get_all_outputs(self, input):
		"""convenient api to get 3d landmarks, face pose, 3d faces"""

		face_boxes = FaceBoxes()
		rects = face_boxes(input)

		# storage
		pts_res = []
		poses = []
		vertices_lst = []
		for idx, rect in enumerate(rects):
			roi_box = rect

			# enlarge the bbox a little and do a square crop
			HCenter = (rect[1] + rect[3])/2
			WCenter = (rect[0] + rect[2])/2
			side_len = roi_box[3]-roi_box[1]
			margin = side_len * 1.2 // 2
			roi_box[0], roi_box[1], roi_box[2], roi_box[3] = WCenter-margin, HCenter-margin, WCenter+margin, HCenter+margin

			img = crop_img(input, roi_box)
			img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LANCZOS4)
			img = torch.from_numpy(img)
			img = img.permute(2,0,1)
			img = img.unsqueeze(0)
			img = (img - 127.5)/ 128.0

			with torch.no_grad():
				param = self.forward_test(img)
			
			param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

			lmks = predict_sparseVert(param, roi_box, transform=True)
			vertices = predict_denseVert(param, roi_box,  transform=True)
			angles, translation = predict_pose(param, roi_box)

			pts_res.append(lmks)
			vertices_lst.append(vertices)
			poses.append([angles, translation])

		return pts_res, vertices_lst, poses

if __name__ == '__main__':
	pass
