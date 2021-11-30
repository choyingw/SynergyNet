import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time
import numpy as np

from utils.ddfa import ToTensor, Normalize, CenterCrop, DDFATestDataset
from model_building import SynergyNet
from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
from benchmark_aflw2000 import ana_msg as ana_alfw2000

import argparse
import os
import glob
import math
from math import cos, atan2, asin
import cv2

from utils.params import ParamsPack
param_pack = ParamsPack()

def parse_pose(param):
    '''parse parameters into pose'''
    if len(param)==62:
        param = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    else:
        param = param * param_pack.param_std + param_pack.param_mean
    Ps = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)  # yaw, pitch, roll
    return P, pose

def P2sRt(P):
    '''decomposing camera matrix P'''   
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d

def matrix2angle(R):
    '''convert matrix to angle'''
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

def parsing(param):
    p_ = param[:, :12].reshape(-1, 3, 4)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].reshape(-1, 3, 1)
    alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
    alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
    return p, offset, alpha_shp, alpha_exp

def reconstruct_vertex(param, data_param, whitening=True, transform=True, lmk_pts=68):
    """
    This function includes parameter de-whitening, reconstruction of landmarks, and transform from coordinate space (x,y) to image space (u,v)
    """
    param_mean, param_std, w_shp_base, u_base, w_exp_base = data_param

    if whitening:
        if param.shape[1] == 62:
            param = param * param_std[:62] + param_mean[:62]
        else:
            raise NotImplementedError("Parameter length must be 62")

    if param.shape[1] == 62:
        p, offset, alpha_shp, alpha_exp = parsing(param)
    else:
        raise NotImplementedError("Parameter length must be 62")

    vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).contiguous().view(-1, lmk_pts, 3).transpose(1,2) + offset
    if transform:
        vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

    return vertex

def extract_param(checkpoint_fp, root='', args=None, filelists=None, device_ids=[0],
                  batch_size=128, num_workers=4):
    map_location = {'cuda:{}'.format(i): 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
    
    # Need to take off these for different numbers of base landmark points
    # del checkpoint['module.u_base']
    # del checkpoint['module.w_shp_base']
    # del checkpoint['module.w_exp_base']

    torch.cuda.set_device(device_ids[0])

    model = SynergyNet(args)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(checkpoint, strict=False)

    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensor(), CenterCrop(5, mode='test') , Normalize(mean=127.5, std=128)  ]))
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

    print('Extracting params take {: .3f}s'.format(time.time() - end))
    return outputs, model.module.data_param


def _benchmark_aflw2000(outputs):
    '''Calculate the error statistics.'''
    return ana_alfw2000(calc_nme_alfw2000(outputs,option='ori'))

# AFLW2000 facial alignment
img_list = sorted(glob.glob('./aflw2000_data/AFLW2000-3D_crop/*.jpg'))
def benchmark_aflw2000_params(params, data_param):
    '''Reconstruct the landmark points and calculate the statistics'''
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
            lm = reconstruct_vertex(batch_data, data_param, lmk_pts=68)
            lm = lm.cpu().numpy()
            for j in range(residual):
                outputs.append(lm[j, :2, :])
        else:
            batch_data = params[i*batch_size: (i+1)*batch_size]
            lm = reconstruct_vertex(batch_data, data_param, lmk_pts=68)
            lm = lm.cpu().numpy()
            for j in range(batch_size):
                if i == 0:
                    #plot the first 50 samples for validation
                    bkg = cv2.imread(img_list[i*batch_size+j],-1)
                    lm_sample = lm[j]
                    c0 = np.clip((lm_sample[1,:]).astype(np.int), 0, 119)
                    c1 = np.clip((lm_sample[0,:]).astype(np.int), 0, 119)
                    for y, x, in zip([c0,c0,c0-1,c0-1],[c1,c1-1,c1,c1-1]):
                        bkg[y, x, :] = np.array([233,193,133])
                    cv2.imwrite(f'./results/{i*batch_size+j}.png', bkg)

                outputs.append(lm[j, :2, :])
    return _benchmark_aflw2000(outputs)


# AFLW2000 face orientation estimation
def benchmark_FOE(params):
    """
    FOE benchmark validation. Only calculate the groundtruth of angles within [-99, 99] (following FSA-Net https://github.com/shamangary/FSA-Net)
    """

    # AFLW200 groundturh and indices for skipping, whose yaw angle lies outside [-99, 99]
    exclude_aflw2000 = './aflw2000_data/eval/ALFW2000-3D_pose_3ANG_excl.npy'
    skip_aflw2000 = './aflw2000_data/eval/ALFW2000-3D_pose_3ANG_skip.npy'

    if not os.path.isfile(exclude_aflw2000) or not os.path.isfile(skip_aflw2000):
        raise RuntimeError('Missing data')

    pose_GT = np.load(exclude_aflw2000) 
    skip_indices = np.load(skip_aflw2000)
    pose_mat = np.ones((pose_GT.shape[0],3))

    idx = 0
    for i in range(params.shape[0]):
        if i in skip_indices:
            continue
        P, angles = parse_pose(params[i])
        angles[0], angles[1], angles[2] = angles[1], angles[0], angles[2] # we decode raw-ptich-yaw order
        pose_mat[idx,:] = np.array(angles)
        idx += 1

    pose_analyis = np.mean(np.abs(pose_mat-pose_GT),axis=0) # pose GT uses [pitch-yaw-roll] order
    MAE = np.mean(pose_analyis)
    yaw = pose_analyis[1]
    pitch = pose_analyis[0]
    roll = pose_analyis[2]
    msg = 'Mean MAE = %3.3f (in deg), [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]'%(MAE, yaw, pitch, roll)
    print('\nFace orientation estimation:')
    print(msg)
    return msg

def benchmark(checkpoint_fp, args):
    '''benchmark validation pipeline'''
    device_ids = [0]

    def aflw2000():
        root = './aflw2000_data/AFLW2000-3D_crop'
        filelists = './aflw2000_data/AFLW2000-3D_crop.list'

        if not os.path.isdir(root) or not os.path.isfile(filelists):
            raise RuntimeError('check if the testing data exist')

        params, data_param = extract_param(
            checkpoint_fp=checkpoint_fp,
            root=root,
            args= args,
            filelists=filelists,
            device_ids=device_ids,
            batch_size=128)

        info_out_fal = benchmark_aflw2000_params(params, data_param)
        print(info_out_fal)
        info_out_foe = benchmark_FOE(params)

    aflw2000()

def main():
    parser = argparse.ArgumentParser(description='SynergyNet benchmark on AFLW2000-3D')
    parser.add_argument('-a', '--arch', default='mobilenet_v2', type=str)
    parser.add_argument('-w', '--weights', default='models/best.pth.tar', type=str)
    parser.add_argument('-d', '--device', default='0', type=str)
    parser.add_argument('--img_size', default='120', type=int)
    args = parser.parse_args()
    args.device = [int(d) for d in args.device.split(',')]

    benchmark(args.weights, args)


if __name__ == '__main__':
    main()