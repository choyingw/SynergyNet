import os.path as osp
import numpy as np
from .io import _load

def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)

class ParamsPack():
	"""Parameter package"""
	def __init__(self):
		try:
			d = make_abs_path('../3dmm_data')
			self.keypoints = _load(osp.join(d, 'keypoints_sim.npy'))

			# PCA basis for shape, expression, texture
			self.w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
			self.w_exp = _load(osp.join(d, 'w_exp_sim.npy'))
			# param_mean and param_std are used for re-whitening
			meta = _load(osp.join(d, 'param_whitening.pkl'))
			self.param_mean = meta.get('param_mean')
			self.param_std = meta.get('param_std')
			# mean values
			self.u_shp = _load(osp.join(d, 'u_shp.npy'))
			self.u_exp = _load(osp.join(d, 'u_exp.npy'))
			self.u = self.u_shp + self.u_exp
			self.w = np.concatenate((self.w_shp, self.w_exp), axis=1)
			# base vector for landmarks
			self.w_base = self.w[self.keypoints]
			self.w_norm = np.linalg.norm(self.w, axis=0)
			self.w_base_norm = np.linalg.norm(self.w_base, axis=0)
			self.u_base = self.u[self.keypoints].reshape(-1, 1)
			self.w_shp_base = self.w_shp[self.keypoints]
			self.w_exp_base = self.w_exp[self.keypoints]
			self.std_size = 120
			self.dim = self.w_shp.shape[0] // 3
		except:
			raise RuntimeError('Missing data')