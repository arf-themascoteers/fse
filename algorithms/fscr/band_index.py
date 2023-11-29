import torch
import torch.nn as nn
import torch.nn.functional as F


class BandIndex(nn.Module):
    def __init__(self, original_feature_size, sigmoid=True):
        super().__init__()
        self.original_feature_size = original_feature_size
        self.sigmoid = sigmoid
        self.net = nn.Sequential(
            nn.Linear(original_feature_size, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1),
            nn.Flatten(start_dim=0)
        )
        self.X = None

    def forward(self, X, spline):
        self.X = X
        outs = spline.evaluate(self.index_value())
        return outs

    def index_value(self):
        index = self.net(self.X)
        index = torch.mean(index)
        if self.sigmoid:
            return F.sigmoid(index)
        return index

    def range_loss(self):
        index = self.net(self.X)
        if self.sigmoid:
            return 0
        loss_l_lower = F.relu(-1 * index)
        loss_l_upper = F.relu(index - 1)
        return loss_l_lower + loss_l_upper
