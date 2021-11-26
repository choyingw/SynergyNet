import torch
import torch.nn as nn
from utils.params import ParamsPack
param_pack = ParamsPack()
import math


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.log_term = math.log(1 + self.omega / self.epsilon)

    def forward(self, pred, target, kp=False):
        n_points = pred.shape[2]
        pred = pred.transpose(1,2).contiguous().view(-1, 3*n_points)
        target = target.transpose(1,2).contiguous().view(-1, 3*n_points)
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * self.log_term
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

class ParamLoss(nn.Module):
    """Input and target are all 62-d param"""
    def __init__(self):
        super(ParamLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, input, target, mode = 'normal'):
        if mode == 'normal':
            loss = self.criterion(input[:,:12], target[:,:12]).mean(1) + self.criterion(input[:,12:], target[:,12:]).mean(1)
            return torch.sqrt(loss)
        elif mode == 'only_3dmm':
            loss = self.criterion(input[:,:50], target[:,12:62]).mean(1)
            return torch.sqrt(loss)
        return torch.sqrt(loss.mean(1))


if __name__ == "__main__":
    pass

