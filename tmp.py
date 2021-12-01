import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from utils.ddfa import ToTensor, Normalize, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors, predict_pose, draw_axis, dump_to_xyz, write_obj, cal_nosetip_dis, filter_from_data
from utils.cv_plot import plot_pose_box
import argparse
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from face3d.face3d import mesh
from face3d.face3d.morphable_model import MorphabelModel
import os
import os.path as osp
import glob
#from FaceBoxes import FaceBoxes
from utils.render import render

STD_SIZE = 120
OCD_DIST = np.load('3D_eval/Florence_OCD.npy')
OCD_DIST[1]=91.0

#size_

def normalize_vertices(vertices):
    """
    Normalize mesh vertices into a unit cube centered at zero.
    """
    vertices = vertices - vertices.min(1)[:, None]
    vertices /= np.abs(vertices).max()
    vertices *= 2
    vertices -= vertices.max(1)[:, None]/ 2
    return vertices

def mean_shift(vertices):
    """
    Normalize mesh vertices into a unit cube centered at zero.
    """
    vertices = vertices - vertices.mean(0)[None, :]
    #vertices /= np.abs(vertices).max()
    vertices *= 2
    vertices -= vertices.max(0)[None,:]/ 2
    return vertices

def get_colors(img, ver):
    h, w, _ = img.shape
    ver[0, :] = np.minimum(np.maximum(ver[0, :], 0), w - 1)  # x
    ver[1, :] = np.minimum(np.maximum(ver[1, :], 0), h - 1)  # y
    ind = np.round(ver).astype(np.int32)
    colors = img[ind[1, :], ind[0, :], :] / 255.  # n x 3

    return colors.copy()

def normalize_vertices_InputSpace(vertices):
    """
    Normalize mesh vertices which is consistent to input image.
    """
    vertices[:, 2, :] = vertices[:, 2, :] - vertices.min(2)[0][:, 2, None] # unify the z-axis
    vertices[:, 0, :] -= size_/2 # center image origin of x and y axis to O
    vertices[:, 1, :] -= size_/2 # center image origin of x and y axis to O

    vertices[:, 0, :] /= size_ # reduce the frame size from 120 to 1
    vertices[:, 1, :] /= size_ # reduce the frame size from 120 to 1
    vertices[:,  2, :] /= 100 # approx. reduce the z-axis to 1
    vertices[:, :3, :] *= 2
    vertices[:, 2, :] -= vertices.max(2)[0][:, 2, None] / 2 # center z 
    return vertices

def to_render_inputs(vertices, colors, triangles):
    """
    prepare the input (normalized vertices, triangles, and textures) to neural mesh renderer (NMR) from vertices and colors 
    """
    vertices = normalize_vertices_InputSpace(vertices)

    # mean color of the triangles and expand the dims to make it comform to NMR input
    textures = torch.mean(colors[:,:,triangles], dim=2)
    textures = textures.transpose(1,2).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    textures = textures.expand(-1, -1, 1, 1, 1, -1)
    return (vertices.transpose(1,2), triangles.transpose(0,1).unsqueeze(0).expand(vertices.shape[0],-1,-1), textures)

def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'models/**S1S2_woGeo_best.pth.tar' #models/**S1S2_woGeo_best.pth.tar'
    args.arch = 'mobilenet_v2' #'mobilenet_v2'#
    args.devices_id = [0]

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    
    from model_building import MultiConsisNet
    model = MultiConsisNet(args)
    model_dict = model.state_dict()
    uv_vert=np.load('test.configs/BFM_UV.npy')
    c1 = (uv_vert[:,1]*255.0).astype(np.int32)
    c2 = (uv_vert[:,0]*255.0).astype(np.int32)

    keep_ind = np.load('test.configs/Texture_mapping/keep_ind.npy')
    tri_deletion = np.load('test.configs/Texture_mapping/tri_deletion.npy')

    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. Initialize FaceBoxes
    #face_boxes = FaceBoxes()

    # 3. forward
    tri = sio.loadmat('train.configs/tri.mat')['tri']
    transform = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])

    if osp.isdir(args.files):
        if not args.files[-1] == '/':
            args.files = args.files + '/'
        files = sorted(glob.glob(args.files+'*.jpg')) # check the extension
    else:
        files = [args.files]

    for num_iter, img_fp in enumerate(files):
        print("Processing: ", img_fp)

        # read image and uniformly resize to 256x256
        img_ori = cv2.imread(img_fp)
        img_ori = cv2.resize(img_ori, [256,256])
        size_h, size_w = img_ori.shape[0], img_ori.shape[1]
        global size_ 
        size_ = max(size_h, size_w)

        # suppose only single pre-cropped face exists with uniform size
        rects = [[0, 0, 255, 255]]

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        for num_rect, rect in enumerate(rects):
            
            roi_box = rect            
            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model.forward_test(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68, _ = predict_68pts(param, roi_box, transform=True)
            vertices, colors = predict_dense(param, roi_box, transform=True)
            angles, t3d = predict_pose(param, roi_box)

            pts_res.append(pts68)
            vertices_lst.append([vertices, colors])
            poses.append([angles, t3d, pts68])


            # textured obj file output
            if args.dump_obj:
                if not osp.exists(f'demo_sequences/300VW-{args.serial}/obj/'):
                    os.makedirs(f'demo_sequences/300VW-{args.serial}/obj/')
                
                name = img_fp.rsplit('/',1)[-1][:-4] # dropping off the extension
                colors_temp = cv2.imread(f'test.configs/Texture_mapping_2/uv_art/{name}_fake_B.png',-1)
                colors_temp = np.flip(colors_temp,axis=0)

                colors_uv = (colors_temp[c1,c2,:])
                colors = colors.transpose(1,0)[:,[2,1,0]]

                wfp2 = f'demo_sequences/300VW-{args.serial}/obj/{name}_sim.obj'
                write_obj_with_colors(wfp2, vertices[:,keep_ind], tri_deletion, colors_uv[keep_ind,:].astype(np.float32))

            ind += 1


        if not osp.exists(f'demo_sequences/300VW-{args.serial}/fitted_image/'):
            os.makedirs(f'demo_sequences/300VW-{args.serial}/fitted_image/')
        if not osp.exists(f'demo_sequences/300VW-{args.serial}/overlayed_image/'):
            os.makedirs(f'demo_sequences/300VW-{args.serial}/overlayed_image/')
        if not osp.exists(f'demo_sequences/300VW-{args.serial}/overlayed_image_solid/'):
            os.makedirs(f'demo_sequences/300VW-{args.serial}/overlayed_image_solid/')    
        if not osp.exists(f'demo_sequences/300VW-{args.serial}/original_image/'):
            os.makedirs(f'demo_sequences/300VW-{args.serial}/original_image/')
        if not osp.exists(f'demo_sequences/300VW-{args.serial}/landmarks/'):
            os.makedirs(f'demo_sequences/300VW-{args.serial}/landmarks/')
        if not osp.exists(f'demo_sequences/300VW-{args.serial}/poses/'):
            os.makedirs(f'demo_sequences/300VW-{args.serial}/poses/')
        
        fitted_image = 0
        name = img_fp.rsplit('/',1)[-1][:-4]
        cv2.imwrite(f'demo_sequences/300VW-{args.serial}/original_image/{name}.jpg', img_ori)
        img_ori_copy = img_ori.copy()


        for k in range(len(vertices_lst)):
            vert = vertices_lst[k][0].transpose(1,0)
            clrs = get_colors(img_ori, vert.transpose(1,0)) * 255.0
            fitted_image += mesh.render.render_colors(vert, tri.T-1, clrs, img_ori.shape[0], img_ori.shape[1]) # in BGR

        fitted_image = fitted_image[:,:,[2,1,0]] # To RGB
        gray = cv2.cvtColor(fitted_image, cv2.COLOR_RGB2GRAY)
        area_x, area_y = np.where(gray>0)

        fitted_image = np.asarray(fitted_image, np.uint8)
        fitted_image = cv2.cvtColor(fitted_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'demo_sequences/300VW-{args.serial}/fitted_image/{name}.jpg', fitted_image)
        
        img_ori_copy = img_ori.copy()
        img_ori[area_x,area_y,:] = fitted_image[area_x,area_y,:]*0.6 + img_ori[area_x,area_y,:] *0.4
        cv2.imwrite(f'demo_sequences/300VW-{args.serial}/overlayed_image/{name}.jpg', img_ori)

        wfp = f'demo_sequences/300VW-{args.serial}/overlayed_image_solid/{name}.jpg'
        render(img_ori, vertices_lst, alpha=0.6, wfp=wfp)

        draw_landmarks(img_ori_copy, pts_res, wfp=f'demo_sequences/300VW-{args.serial}/landmarks/{name}.jpg', show_flg=args.show_flg)
        
        img_axis_plot = img_ori_copy
        for ang, translation, pts68 in poses:
            img_axis_plot = draw_axis(img_axis_plot, angles[0], angles[1],
                angles[2], translation[0], translation[1], size = 50, pts68=pts68)

        cv2.imwrite(f'demo_sequences/300VW-{args.serial}/poses/{name}.jpg', img_axis_plot)
        print(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', default='',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='False', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='False', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='False', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='False', type=str2bool)
    parser.add_argument('--dump_xyz', default='False', type=str2bool)
    parser.add_argument('--dump_pts', default='False', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='False', type=str2bool)
    parser.add_argument('--dump_depth', default='False', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='False', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')
    parser.add_argument('-p', '--params', default='102', type=str)
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--w1', default='0.15', type=str)
    parser.add_argument('--w2', default='', type=str)
    parser.add_argument('--w3', default='', type=str)
    parser.add_argument('-s', '--serial', default='', type=str)
    parser.add_argument('--video', default='False', type=str2bool)

    args = parser.parse_args()
    main(args)

## Large image visualization
# min_x, max_x, min_y, max_y = np.min(area_x), np.max(area_x), np.min(area_y), np.max(area_y)
# offset_pix = 100
# fitted_image_2 = fitted_image[min_x-5: max_x+5, min_y-5: max_y+5, :]
# fitted_side = np.zeros((350,350, 3))
# fitted_side[offset_pix: offset_pix+ (max_x-min_x)+10, offset_pix: offset_pix+ (max_y-min_y)+10, :] = fitted_image_2
# fitted_side = cv2.resize(fitted_side, dsize=(700,700))
# print(fitted_side.shape)

# img_ori_copy = img_ori.copy()

# h,w = img_ori.shape[0], img_ori.shape[1]
# img_ori[0:700, 0:700, :] = fitted_side
# cv2.imwrite(f'demo_sequences/300VW-{args.serial}/overlayed_image/{name}.jpg', img_ori)


## Old code using face3d
# for k in range(len(vertices_lst)):
#             vert = vertices_lst[k][0].transpose(1,0)
#             clrs = vertices_lst[k][1].transpose(1,0)
#             fitted_image += mesh.render.render_colors(vert, tri.T-1, clrs, img_ori.shape[0], img_ori.shape[1])

#         gray = cv2.cvtColor(fitted_image, cv2.COLOR_RGB2GRAY)
#         area_x, area_y = np.where(gray>0)

#         fitted_image = np.asarray(fitted_image, np.uint8)
#         fitted_image = cv2.cvtColor(fitted_image, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(f'demo_sequences/300VW-{args.serial}/fitted_image/{name}.jpg', fitted_image)
        
#         img_ori_copy = img_ori.copy()
#         img_ori[area_x,area_y,:] = fitted_image[area_x,area_y,:]*0.6 + img_ori[area_x,area_y,:] *0.4
#         cv2.imwrite(f'demo_sequences/300VW-{args.serial}/overlayed_image/{name}.jpg', img_ori)