import torch.nn as nn
import torch
import torch.nn.functional as F


class BandIndex(nn.Module):
    def __init__(self, val=None):
        super().__init__()
        if val is None:
            val = torch.rand(1)
            val = (val*10)-5
        self.raw_index = nn.Parameter(val)

    def forward(self, spline):
        outs = spline.evaluate(self.index_value())
        return outs

    def index_value(self):
        return torch.sigmoid(self.raw_index)
