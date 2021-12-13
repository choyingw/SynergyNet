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
from backbone_nets import dcnv2
from backbone_nets import dcnv1
from backbone_nets.pointnet_backbone import MLP_for, MLP_rev
from loss_definition import ParamLoss, WingLoss

import time

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
        elif 'dcnv2' in self.args.arch:
            self.backbone = getattr(dcnv2, args.arch)(pretrained=False)
        elif 'dcnv1' in self.args.arch:
            self.backbone = getattr(dcnv1, args.arch)(pretrained=False)
        else:
            raise RuntimeError("Please choose [mobilenet_v2, mobilenet_1, resnet50, ghostnet, dcnv1, or dcnv2]")

    def forward(self,input, target):
        """Training time forward"""
        _3D_attr, avgpool = self.backbone(input)
        _3D_attr_GT = target.type(torch.FloatTensor)#.cuda.FloatTensor)
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
        self.triangles = torch.Tensor(self.triangles.astype(np.int)).long()#.cuda()
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
                    # 'loss_LMK_pointNet': 0.0,
                    'loss_Param_In':0.0,
                    # 'loss_Param_S2': 0.0,
                    # 'loss_Param_S1S2': 0.0,
                    }

        self.register_buffer('param_mean', torch.Tensor(param_pack.param_mean))#.cuda(non_blocking=True))
        self.register_buffer('param_std', torch.Tensor(param_pack.param_std))#.cuda(non_blocking=True))
        self.register_buffer('w_shp', torch.Tensor(param_pack.w_shp))#.cuda(non_blocking=True))
        self.register_buffer('u', torch.Tensor(param_pack.u))#.cuda(non_blocking=True))
        self.register_buffer('w_exp', torch.Tensor(param_pack.w_exp))#.cuda(non_blocking=True))

        # If doing only offline evaluation, use these
        # self.u_base = torch.Tensor(param_pack.u_base).cuda(non_blocking=True)
        # self.w_shp_base = torch.Tensor(param_pack.w_shp_base).cuda(non_blocking=True)
        # self.w_exp_base = torch.Tensor(param_pack.w_exp_base).cuda(non_blocking=True)

        # Online training needs these to parallel
        self.register_buffer('u_base', torch.Tensor(param_pack.u_base))#.cuda(non_blocking=True))
        self.register_buffer('w_shp_base', torch.Tensor(param_pack.w_shp_base))#.cuda(non_blocking=True))
        self.register_buffer('w_exp_base', torch.Tensor(param_pack.w_exp_base))#.cuda(non_blocking=True))
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

        """
        point_residual = self.forwardDirection(vertex_lmk, avgpool, _3D_attr[:,12:52], _3D_attr[:,52:62])
        vertex_lmk = vertex_lmk + 0.05 * point_residual
        self.loss['loss_LMK_pointNet'] = 0.05 * self.LMKLoss_3D(vertex_lmk, vertex_GT_lmk, kp=True)

        _3D_attr_S2 = self.reverseDirection(vertex_lmk)
        self.loss['loss_Param_S2'] = 0.02 * self.ParamLoss(_3D_attr_S2, _3D_attr_GT, mode='only_3dmm')
        self.loss['loss_Param_S1S2'] = 0.001 * self.ParamLoss(_3D_attr_S2, _3D_attr, mode='only_3dmm')
        """

        return self.loss

    def forward_test(self, input):
        """test time forward"""
        _3D_attr, _ = self.I2P.forward_test(input)
        return _3D_attr

    def get_losses(self):
        return self.loss.keys()


if __name__ == '__main__':
    pass
