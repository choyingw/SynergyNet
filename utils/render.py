# modified from 3DDFA-V2

import sys

sys.path.append('..')

import cv2
import numpy as np
import scipy.io as sio

from Sim3DR import RenderPipeline

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr

cfg = {
    'intensity_ambient': 0.75,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.7,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.2,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)

def render(img, ver_lst, alpha=0.6, wfp=None, tex=None, connectivity=None):
    tri = sio.loadmat('./3dmm_data/tri.mat')['tri'] - 1
    tri = _to_ctype(tri.T).astype(np.int32)
    # save solid mesh rendering and alpha overlaying on images
    if not connectivity is None:
        tri = _to_ctype(connectivity.T).astype(np.int32)

    overlap = img.copy()
    for ver_ in ver_lst:
        ver_ = ver_.astype(np.float32)
        ver = _to_ctype(ver_.T)  # transpose
        overlap = render_app(ver, tri, overlap, texture=tex)
    cv2.imwrite(wfp[:-4]+'_solid'+'.png', overlap)

    res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save mesh result to {wfp}')

    return res
