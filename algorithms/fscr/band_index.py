import torch.nn as nn
import torch
import torch.nn.functional as F


class BandIndex(nn.Module):
    def __init__(self, indexify="sigmoid"):
        super().__init__()
        self.raw_index = nn.Parameter((torch.rand(1)*10)-5)
        self.indexify = indexify

    def forward(self, spline):
        outs = spline.evaluate(self.index_value())
        return outs

    def index_value(self):
        if self.indexify == "sigmoid":
            return F.sigmoid(self.raw_index)
        return self.raw_index

    def range_loss(self):
        if self.indexify == "sigmoid":
            return 0
        loss_l_lower = F.relu(-1 * self.raw_index)
        loss_l_upper = F.relu(self.raw_index - 1)
        return loss_l_lower + loss_l_upper
