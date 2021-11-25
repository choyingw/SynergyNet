#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time
import numpy as np

from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
from benchmark_aflw2000 import ana_msg as ana_alfw2000

from utils.ddfa import ToTensor, Normalize, DDFATestDataset, CenterCrop
import argparse

import logging
import os
from utils.params import ParamsPack
param_pack = ParamsPack()
import glob
import scipy.io as sio
import math
from math import cos, sin, atan2, asin, sqrt

# Only work with numpy without batch
def parse_pose(param):
    """
    Parse the parameters into 3x4 affine matrix and pose angles
    """
    param = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    Ps = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle_corr(R)  # yaw, pitch, roll
    return P, pose

def P2sRt(P):
    ''' 
    Decompositing camera matrix P.
    '''
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d

# def matrix2angle(R):
#     ''' 
#     Compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
#     '''

#     if R[2, 0] != 1 and R[2, 0] != -1:
#         x = asin(R[2, 0])
#         y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
#         z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

#     else:  # Gimbal lock
#         z = 0  # can be anything
#         if R[2, 0] == -1:
#             x = np.pi / 2
#             y = z + atan2(R[0, 1], R[0, 2])
#         else:
#             x = -np.pi / 2
#             y = -z + atan2(-R[0, 1], -R[0, 2])
    
#     rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi

#     return [rx, ry, rz]

#numpy
def matrix2angle_corr(R):
    ''' 
    Compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    '''

    if R[2, 0] != 1 and R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[1, 2] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[0, 1] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])
    
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi

    return [rx, ry, rz]



def parse_param_62_batch(param):
    """batch styler"""
    p_ = param[:, :12].reshape(-1, 3, 4)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].reshape(-1, 3, 1)
    alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
    alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
    return p, offset, alpha_shp, alpha_exp


# 62-with-false-rot
def reconstruct_vertex(param, data_param, whitening=True, dense=False, transform=True):
    """
    Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    Working with Tensor with batch. Using Fortan-type reshape.
    """
    param_mean, param_std, w_shp_base, u_base, w_exp_base = data_param

    if whitening:
        if param.shape[1] == 62:
            param = param * param_std[:62] + param_mean[:62]

    p, offset, alpha_shp, alpha_exp = parse_param_62_batch(param)

    """For 68 pts"""
    vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).contiguous().view(-1, 68, 3).transpose(1,2) + offset

    if transform:
        # transform to image coordinate space
        vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :] ## corrected


    return vertex


def extract_param(model, root='', filelists=None,
                  batch_size=128, num_workers=4):
    
    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensor(), CenterCrop(5, mode='test'), Normalize(mean=127.5, std=130)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True
    model.eval()

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, inputs in enumerate(data_loader):
            inputs = inputs.cuda()
            output = model.module.forward_test(inputs)

            for i in range(output.shape[0]):
                param_prediction = output[i].cpu().numpy().flatten()

                outputs.append(param_prediction)
        outputs = np.array(outputs, dtype=np.float32)

    logging.info('Extracting params take {: .3f}s\n'.format(time.time() - end))
    return outputs


def _benchmark_aflw2000(outputs):
    """
    Calculate the error statistics.
    """
    return ana_alfw2000(calc_nme_alfw2000(outputs, option='ori'))


def benchmark_aflw2000_params(params, data_param):
    """
    Reconstruct the landmark points and calculate the statistics
    """
    outputs = []
    params = torch.Tensor(params).cuda()

    batch_size = 50
    num_samples = params.shape[0]
    iter_num = math.floor(num_samples / batch_size)
    residual = num_samples % batch_size
    for i in range(iter_num+1):
        if i == iter_num:
            if residual == 0:
                break
            batch_data = params[i*batch_size: i*batch_size + residual]
            lm = reconstruct_vertex(batch_data, data_param)
            lm = lm.cpu().numpy()
            for j in range(residual):
                outputs.append(lm[j, :2, :])
        else:
            batch_data = params[i*batch_size: (i+1)*batch_size]
            lm = reconstruct_vertex(batch_data, data_param)
            lm = lm.cpu().numpy()
            for j in range(batch_size):
                outputs.append(lm[j, :2, :])
    return _benchmark_aflw2000(outputs)


def benchmark_FOE(params):
    """
    FOE benchmark validation. Only calculate the groundtruth of angles within [-99, 99]
    """

    # Define the data path for AFLW200 groundturh and skip indices, where the data and structure lie on S3 buckets (fixed structure)
    groundtruth_excl = './aflw2000_data/eval/ALFW2000-3D_pose_3ANG_excl.npy'
    skip_aflw2000 = './aflw2000_data/eval/ALFW2000-3D_pose_3ANG_skip.npy'

    if not os.path.isfile(groundtruth_excl) or not os.path.isfile(skip_aflw2000):
        raise RuntimeError('The data is not properly downloaded from the S3 bucket. Please check your S3 bucket access permission')


    pose_GT = np.load(groundtruth_excl) # groundtruth load
    skip_indices = np.load(skip_aflw2000) # load the skip indices in AFLW2000
    pose_mat = np.ones((pose_GT.shape[0],3))

    total = 0
    for i in range(params.shape[0]):
        if i in skip_indices:
            continue
        P, angles = parse_pose(params[i]) # original per-sample decode
        angles[0], angles[1], angles[2] = angles[1], angles[0], angles[2]
        pose_mat[total,:] = np.array(angles)
        total += 1

    pose_analy = np.mean(np.abs(pose_mat-pose_GT),axis=0)
    MAE = np.mean(pose_analy)
    yaw = pose_analy[1]
    pitch = pose_analy[0]
    roll = pose_analy[2]
    msg = 'MAE = %3.3f, [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]'%(MAE, yaw, pitch, roll)
    print('\n--------------------------------------------------------------------------------')
    print(msg)
    print('--------------------------------------------------------------------------------')
    return msg


# 102
def benchmark_pipeline(model):
    """
    Run the benchmark validation pipeline for Facial Alignments: AFLW and AFLW2000, FOE: AFLW2000.
    """

    def aflw2000(data_param):
        root = './aflw2000_data/AFLW2000-3D_crop'
        filelists = './aflw2000_data/AFLW2000-3D_crop.list'

        if not os.path.isdir(root) or not os.path.isfile(filelists):
            raise RuntimeError('The data is not properly downloaded from the S3 bucket. Please check your S3 bucket access permission')

        params = extract_param(
            model=model,
            root=root,
            filelists=filelists,
            batch_size=128)

        s2 = benchmark_aflw2000_params(params, data_param)
        logging.info(s2)
        # s3 = benchmark_FOE(params)
        # logging.info(s3)

    aflw2000(model.module.data_param)


def main():
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    parser.add_argument('--arch', default='mobilenet_1', type=str)
    parser.add_argument('-c', '--checkpoint-fp', default='models/phase1_wpdc.pth.tar', type=str)
    args = parser.parse_args()

    benchmark_pipeline(args.arch, args.checkpoint_fp)


if __name__ == '__main__':
    main()
