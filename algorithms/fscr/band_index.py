import torch.nn as nn
import torch
import torch.nn.functional as F


class BandIndex(nn.Module):
    def __init__(self, sigmoid=True):
        super().__init__()
        self.sigmoid = sigmoid
        self.net = nn.Sequential(
            nn.Linear(5, 1)
        )
        self.raw_weights = nn.Parameter(torch.rand(5))

    def forward(self, spline):
        index = self.index_value()
        outs = spline.evaluate(index)
        return outs

    def index_value(self):
        index = self.net(self.raw_weights)
        if self.sigmoid:
            return F.sigmoid(index)
        return index

    def range_loss(self):
        if self.sigmoid:
            return 0
        index = self.index_value()
        loss_l_lower = F.relu(-1 * index)
        loss_l_upper = F.relu(index - 1)
        return loss_l_lower + loss_l_upper
