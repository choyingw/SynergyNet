import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['MLP_for', 'MLP_rev']

class MLP_for(nn.Module):
	def __init__(self, num_pts):
		super(MLP_for,self).__init__()
		self.conv1 = torch.nn.Conv1d(3,64,1)
		self.conv2 = torch.nn.Conv1d(64,64,1)
		self.conv3 = torch.nn.Conv1d(64,64,1)
		self.conv4 = torch.nn.Conv1d(64,128,1)
		self.conv5 = torch.nn.Conv1d(128,1024,1)
		self.conv6 = nn.Conv1d(2418, 512, 1) # 1024 + 64 + 1280 = 2368
		self.conv7 = nn.Conv1d(512, 256, 1)
		self.conv8 = nn.Conv1d(256, 128, 1)
		self.conv9 = nn.Conv1d(128, 3, 1)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(1024)
		self.bn6 = nn.BatchNorm1d(512)
		self.bn7 = nn.BatchNorm1d(256)
		self.bn8 = nn.BatchNorm1d(128)
		self.bn9 = nn.BatchNorm1d(3)
		self.num_pts = num_pts
		self.max_pool = nn.MaxPool1d(num_pts)
		
	def forward(self,x, other_input1=None, other_input2=None, other_input3=None):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		point_features = out
		out = F.relu(self.bn3(self.conv3(out)))
		out = F.relu(self.bn4(self.conv4(out)))
		out = F.relu(self.bn5(self.conv5(out)))
		global_features = self.max_pool(out)
		global_features_repeated = global_features.repeat(1,1, self.num_pts)

		#out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated],1))))
		
		# Avg_pool
		# avgpool = other_input1
		# avgpool = avgpool.unsqueeze(2).repeat(1,1,self.num_pts)
		# out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated, avgpool],1))))

		#3DMMImg
		avgpool = other_input1
		avgpool = avgpool.unsqueeze(2).repeat(1,1,self.num_pts)
		
		shape_code = other_input2
		shape_code = shape_code.unsqueeze(2).repeat(1,1,self.num_pts)

		expr_code = other_input3
		expr_code = expr_code.unsqueeze(2).repeat(1,1,self.num_pts)

		out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated, avgpool, shape_code, expr_code],1))))


		out = F.relu(self.bn7(self.conv7(out)))
		out = F.relu(self.bn8(self.conv8(out)))
		out = F.relu(self.bn9(self.conv9(out)))
		return out


class MLP_rev(nn.Module):
	def __init__(self, num_pts):
		super(MLP_rev,self).__init__()
		self.conv1 = torch.nn.Conv1d(3,64,1)
		self.conv2 = torch.nn.Conv1d(64,64,1)
		self.conv3 = torch.nn.Conv1d(64,64,1)
		self.conv4 = torch.nn.Conv1d(64,128,1)
		self.conv5 = torch.nn.Conv1d(128,1024,1)
		self.conv6_1 = nn.Conv1d(1024, 12, 1)
		self.conv6_2 = nn.Conv1d(1024, 40, 1)
		self.conv6_3 = nn.Conv1d(1024, 10, 1)

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(1024)
		self.bn6_1 = nn.BatchNorm1d(12)
		self.bn6_2 = nn.BatchNorm1d(40)
		self.bn6_3 = nn.BatchNorm1d(10)
		self.num_pts = num_pts
		self.max_pool = nn.MaxPool1d(num_pts)

	def forward(self,x, other_input1=None, other_input2=None, other_input3=None):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = F.relu(self.bn3(self.conv3(out)))
		out = F.relu(self.bn4(self.conv4(out)))
		out = F.relu(self.bn5(self.conv5(out)))
		global_features = self.max_pool(out)

		# Global point feature
		out_rot = F.relu(self.bn6_1(self.conv6_1(global_features)))
		out_shape = F.relu(self.bn6_2(self.conv6_2(global_features)))
		out_expr = F.relu(self.bn6_3(self.conv6_3(global_features)))


		out = torch.cat([out_rot, out_shape, out_expr], 1).squeeze(2)

		return out