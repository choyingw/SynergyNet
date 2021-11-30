#!/usr/bin/env python3
# coding: utf-8


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# from .benchmark import reconstruct_vertex_102
from utils.params import ParamsPack
param_pack = ParamsPack('v201')
from math import cos, sin, atan2, asin, sqrt
import cv2

keep_ind = np.load('test.configs/3DDFA/keep_ind.npy')
tri_deletion = np.load('test.configs/3DDFA/tri_deletion.npy')

def parse_param_102(param):
    """batch styler"""
    p_ = param[:12].reshape(3, 4)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(40, 1)
    alpha_exp = param[52:62].reshape(10, 1)
    alpha_tex = param[62:102].reshape(40, 1)
    return p, offset, alpha_shp, alpha_exp, alpha_tex

def reconstruct_vertex_102(param, whitening=True, dense=False, transform=True, lmk_pts=68):
    """
    Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    Working with batched tensors. Using Fortan-type reshape.
    """

    if whitening:
        if param.shape[0] == 102:
            param_ = param * param_pack.param_std + param_pack.param_mean
        else:
            raise RuntimeError('length of params mismatch')

    p, offset, alpha_shp, alpha_exp, alpha_tex = parse_param_102(param_)

    if dense:
        
        vertex = p @ (param_pack.u + param_pack.w_shp @ alpha_shp + param_pack.w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
        #vertex = (param_pack.u + param_pack.w_shp @ alpha_shp + param_pack.w_exp @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform: 
            # transform to image coordinate space
            vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]

        colors = param_pack.u_tex + param_pack.w_tex @ alpha_tex
        colors = colors.reshape(3, -1, order='F')

    else:
        """For 68 pts"""
        vertex = p @ (param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform: 
            # transform to image coordinate space
            vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]

        colors = None

    return vertex, colors

def parse_pose(param):
    """
    Parse the parameters into 3x4 affine matrix and pose angles
    """
    param = param * param_pack.param_std + param_pack.param_mean
    Ps = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle_corr(R)  # yaw, pitch, roll
    return P, pose, t3d

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


def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size
    return roi_box

NOSE_POINT=8191#8297#8170 #8191
def cal_nosetip_dis(pts, tri):
    ref = pts[:, NOSE_POINT][:, None]
    dis = np.linalg.norm(pts-ref, axis=0)
    keep_ind = np.sort(np.where(dis <= 90)[0])
    np.save('keep_ind.npy', keep_ind)
    pts_retain = pts[:, keep_ind]

    keep_ind_increment = keep_ind+1
    tri_new = []
    for l in range(tri.shape[1]):
        trilet = tri[:, l]
        if (trilet[0] in keep_ind_increment) and (trilet[1] in keep_ind_increment) and (trilet[2] in keep_ind_increment):
            #a=int(np.where(keep_ind_increment==trilet[0])[0])
            #print(a)
            new_indices = [int(np.where(keep_ind_increment==trilet[0])[0]), int(np.where(keep_ind_increment==trilet[1])[0]),
                int(np.where(keep_ind_increment==trilet[2])[0])]

            tri_new.append(new_indices)
    tri_new_arr = np.asarray(tri_new)
    #print(tri_new_arr.shape)
    tri_new_arr = tri_new_arr.transpose(1,0)
    tri_new = 1 + np.asarray(tri_new).transpose(1,0)
    np.save('tri_deletion.npy', tri_new)

    return pts_retain, tri_new

def filter_from_data(pts, tri):
    pts_retain = pts[:, keep_ind]
    tri_new = tri_deletion
    return pts_retain, tri_new


def dump_to_ply(vertex, tri, wfp):
    header = """ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    element face {}
    property list uchar int vertex_indices
    end_header"""

    n_vertex = vertex.shape[1]
    n_face = tri.shape[1]
    header = header.format(n_vertex, n_face)

    with open(wfp, 'w') as f:
        f.write(header + '\n')
        for i in range(n_vertex):
            x, y, z = vertex[:, i]
            f.write('{:.4f} {:.4f} {:.4f}\n'.format(x, y, z))
        for i in range(n_face):
            idx1, idx2, idx3 = tri[:, i]
            f.write('3 {} {} {}\n'.format(idx1 - 1, idx2 - 1, idx3 - 1))
    print('Dump tp {}'.format(wfp))

def dump_to_xyz(vertex, tri, wfp):
    n_vertex = vertex.shape[1]
    with open(wfp, 'w') as f:
        for i in range(n_vertex):
            x, y, z = vertex[:, i]
            f.write('{:.6f} {:.6f} {:.6f}\n'.format(x, y, z))
    print('Dump tp {}'.format(wfp))


def dump_vertex(vertex, wfp):
    sio.savemat(wfp, {'vertex': vertex})
    print('Dump to {}'.format(wfp))


def _predict_vertices(param, roi_bbox, dense, transform=True):
    vertex, colors = reconstruct_vertex_102(param, dense=dense, transform=transform)
    sx, sy, ex, ey = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s

    return vertex, colors


def predict_68pts(param, roi_box, transform=False):
    return _predict_vertices(param, roi_box, dense=False, transform=transform)


def predict_dense(param, roi_box, transform=False):
    return _predict_vertices(param, roi_box, dense=True, transform=transform)

def predict_pose(param, roi_bbox, ret_mat=False):

    param_ = param * param_pack.param_std + param_pack.param_mean
    p, offset, alpha_shp, alpha_exp, alpha_tex = parse_param_102(param_)
    P, angles, t3d = parse_pose(param)

    #print(t3d.shape)

    sx, sy, ex, ey = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    t3d[0] = t3d[0] * scale_x + sx
    t3d[1] = t3d[1] * scale_y + sy

    if ret_mat:
        return P
    return angles, t3d

def draw_landmarks(img, pts, style='fancy', wfp=None, show_flg=False, **kwargs):
    """Draw landmarks using matplotlib"""
    height, width = img.shape[:2]
    base = 6.4 #6.4
    plt.figure(figsize=(base, height / width * base))
    plt.imshow(img[:, :, ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        if style == 'simple':
            plt.plot(pts[i][0, :], pts[i][1, :], 'o', markersize=4, color='g')

        elif style == 'fancy':
            alpha = 0.8
            markersize = 1.5 # change this to 1.5# change this 2.5 # change to 10.5 for AF 
            lw = 0.7 # change this to 0.7 # change this 1.2 # change to 7.5 for AF
            color = kwargs.get('color', 'g')
            markeredgecolor = kwargs.get('markeredgecolor', 'green')

            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

            # close eyes and mouths
            plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                 color=color, lw=lw, alpha=alpha - 0.1)
            plot_close(41, 36)
            plot_close(47, 42)
            plot_close(59, 48)
            plot_close(67, 60)

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

                plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                         color=color,
                         markeredgecolor=markeredgecolor, alpha=alpha)

    if wfp is not None:
        plt.savefig(wfp, dpi=200)
        print('Save visualization result to {}'.format(wfp))
        plt.close()
    if show_flg:
        plt.show()

    plt.close()


def get_colors(image, vertices):
    [h, w, _] = image.shape
    vertices[0, :] = np.minimum(np.maximum(vertices[0, :], 0), w - 1)  # x
    vertices[1, :] = np.minimum(np.maximum(vertices[1, :], 0), h - 1)  # y
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[1, :], ind[0, :], :]  # n x 3

    return colors


def write_obj_with_colors(obj_name, vertices, triangles, colors):
    triangles = triangles.copy() # meshlab start with 1

    # print("1",vertices.shape)
    # print("2",colors.shape)
    # print("3",triangles.shape)

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        ### This api use color: (num_of_vertices, 3), vertices: (3, num_of_vertices)
        for i in range(vertices.shape[1]):
            s = 'v {:.4f} {:.4f} {:.4f} {} {} {}\n'.format(vertices[0, i], vertices[1, i], vertices[2, i], colors[i, 2],
                                               colors[i, 1], colors[i, 0])
            f.write(s)

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[1]):
            s = 'f {} {} {}\n'.format(triangles[0, i], triangles[1, i], triangles[2, i])
            f.write(s)

def write_obj(obj_name, vertices, triangles):
    triangles = triangles.copy() # meshlab start with 1

    # print("1",vertices.shape)
    # print("2",colors.shape)
    # print("3",triangles.shape)

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[1]):
            s = 'v {:.4f} {:.4f} {:.4f}\n'.format(vertices[0, i], vertices[1, i], vertices[2, i])
            f.write(s)
        # write f: ver ind/ uv ind
        for i in range(triangles.shape[1]):
            s = 'f {} {} {}\n'.format(triangles[2, i], triangles[1, i], triangles[0, i])
            f.write(s)



def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100, pts68=None):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    tdx = pts68[0,30]
    tdy = pts68[1,30]


    minx, maxx = np.min(pts68[0, :]), np.max(pts68[0, :])
    miny, maxy = np.min(pts68[1, :]), np.max(pts68[1, :])
    llength = sqrt((maxx - minx) * (maxy - miny))
    size = llength * 0.5

    # print(tdx)
    # print(tdy)
    # print(pts68.shape)
    # exit()

    # if pts8 != None:
    #     tdx = 

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    minus=0

    cv2.line(img, (int(tdx), int(tdy)-minus), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)-minus), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)-minus), (int(x3),int(y3)),(255,0,0),4)

    return img




def main():
    pass


if __name__ == '__main__':
    main()
