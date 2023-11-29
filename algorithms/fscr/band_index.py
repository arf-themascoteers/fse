import torch.nn as nn
import torch
import torch.nn.functional as F


class BandIndex(nn.Module):
    def __init__(self, sigmoid=True):
        super().__init__()
        self.sigmoid = sigmoid
        self.input_weight = nn.Parameter(torch.rand(1,5))
        self.net = nn.Sequential(
            nn.Linear(5,4),
            nn.LeakyReLU(),
            nn.Linear(4,1),
            nn.Flatten(start_dim=0)
        )

    def forward(self, spline):
        outs = spline.evaluate(self.index_value())
        return outs

    def index_value(self):
        index = self.net(self.input_weight)
        if self.sigmoid:
            return F.sigmoid(index)
        return index

    def range_loss(self):
        if self.sigmoid:
            return 0
        loss_l_lower = F.relu(-1 * self.raw_index)
        loss_l_upper = F.relu(self.raw_index - 1)
        return loss_l_lower + loss_l_upper
