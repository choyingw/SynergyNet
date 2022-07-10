import numpy as np
import matplotlib.pyplot as plt
from utils.params import ParamsPack
param_pack = ParamsPack()
from math import cos, sin, atan2, asin, sqrt
import cv2

def write_obj(obj_name, vertices, triangles):
    triangles = triangles.copy() # meshlab start with 1

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

def parse_param(param):
    p_ = param[:12].reshape(3, 4)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(40, 1)
    alpha_exp = param[52:62].reshape(10, 1)
    return p, offset, alpha_shp, alpha_exp

def P2sRt(P):
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
    if R[2, 0] != 1 and R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[1, 2] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[0, 1] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])
    
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi

    return [rx, ry, rz]

def param2vert(param, dense=False, transform=True):
    if param.shape[0] == 62:
        param_ = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    else:
        raise RuntimeError('length of params mismatch')

    p, offset, alpha_shp, alpha_exp = parse_param(param_)

    if dense:
        vertex = p @ (param_pack.u + param_pack.w_shp @ alpha_shp + param_pack.w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
        if transform: 
            # transform to image coordinate space
            vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]

    else:
        vertex = p @ (param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset
        if transform: 
            # transform to image coordinate space
            vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]

    return vertex

def parse_pose(param):
    param = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    Ps = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle_corr(R)  # yaw, pitch, roll
    return P, pose, t3d


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey, _ = [int(round(_)) for _ in roi_box]
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

def _predict_vertices(param, roi_bbox, dense, transform=True):
    vertex = param2vert(param, dense=dense, transform=transform)
    sx, sy, ex, ey, _ = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s

    return vertex

def predict_sparseVert(param, roi_box, transform=False):
    return _predict_vertices(param, roi_box, dense=False, transform=transform)

def predict_denseVert(param, roi_box, transform=False):
    return _predict_vertices(param, roi_box, dense=True, transform=transform)

def predict_pose(param, roi_bbox, ret_mat=False):
    P, angles, t3d = parse_pose(param)

    sx, sy, ex, ey, _ = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    t3d[0] = t3d[0] * scale_x + sx
    t3d[1] = t3d[1] * scale_y + sy

    if ret_mat:
        return P
    return angles, t3d

def draw_landmarks(img, pts, wfp):
    height, width = img.shape[:2]
    base = 6.4 
    plt.figure(figsize=(base, height / width * base))
    plt.imshow(img[:, :, ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        alpha = 0.8
        markersize = 1.5
        lw = 0.7 
        color = 'g'
        markeredgecolor = 'green'

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

    plt.savefig(wfp, dpi=200)
    print('Save landmark result to {}'.format(wfp))
    plt.close()


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
