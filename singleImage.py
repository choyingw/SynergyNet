import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from utils.ddfa import ToTensor, Normalize
from model_building import SynergyNet
from utils.inference import crop_img, predict_sparseVert, draw_landmarks, predict_denseVert, predict_pose, draw_axis
import argparse
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import os
import os.path as osp
import glob
from FaceBoxes import FaceBoxes
from utils.render import render

# Following 3DDFA-V2, we also use 120x120 resolution
IMG_SIZE = 120

def main(args):
    # load pre-tained model
    checkpoint_fp = 'pretrained/best.pth.tar' 
    args.arch = 'mobilenet_v2'
    args.devices_id = [0]

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    
    model = SynergyNet(args)
    model_dict = model.state_dict()

    # because the model is trained by multiple gpus, prefix 'module' should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]

    model.load_state_dict(model_dict, strict=False)
    model = model.cuda()
    model.eval()

    # face detector
    face_boxes = FaceBoxes()

    # preparation
    transform = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])
    if osp.isdir(args.files):
        if not args.files[-1] == '/':
            args.files = args.files + '/'
        if not args.png:
            files = sorted(glob.glob(args.files+'*.jpg'))
        else:
            files = sorted(glob.glob(args.files+'*.png'))
    else:
        files = [args.files]

    for img_fp in files:
        print("Process the image: ", img_fp)

        img_ori = cv2.imread(img_fp)

        # crop faces
        rects = face_boxes(img_ori)

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

            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite(f'validate_{idx}.png', img)
            
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                input = input.cuda()
                param = model.forward_test(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # inferences
            lmks = predict_sparseVert(param, roi_box, transform=True)
            vertices = predict_denseVert(param, roi_box, transform=True)
            angles, translation = predict_pose(param, roi_box)

            pts_res.append(lmks)
            vertices_lst.append(vertices)
            poses.append([angles, translation, lmks])

        if not osp.exists(f'inference_output/rendering_overlay/'):
            os.makedirs(f'inference_output/rendering_overlay/')
        if not osp.exists(f'inference_output/landmarks/'):
            os.makedirs(f'inference_output/landmarks/')
        if not osp.exists(f'inference_output/poses/'):
            os.makedirs(f'inference_output/poses/')
        
        name = img_fp.rsplit('/',1)[-1][:-4]
        img_ori_copy = img_ori.copy()

        # mesh
        render(img_ori, vertices_lst, alpha=0.6, wfp=f'inference_output/rendering_overlay/{name}.jpg')
        
        # landmarks
        draw_landmarks(img_ori_copy, pts_res, wfp=f'inference_output/landmarks/{name}.jpg')
        
        # face orientation
        img_axis_plot = img_ori_copy
        for angles, translation, lmks in poses:
            img_axis_plot = draw_axis(img_axis_plot, angles[0], angles[1],
                angles[2], translation[0], translation[1], size = 50, pts68=lmks)
        wfp = f'inference_output/poses/{name}.jpg'
        cv2.imwrite(wfp, img_axis_plot)
        print(f'Save pose result to {wfp}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', default='', help='path to a single image or path to a folder containing multiple images')
    parser.add_argument("--png", action="store_true", help="if images are with .png extension")
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int)

    args = parser.parse_args()
    main(args)