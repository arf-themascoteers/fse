import torch.nn as nn
import torch


class BSNetFCCW(nn.Module):
    def __init__(self, bands):
        super().__init__()
        torch.manual_seed(3)
        self.bands = bands
        self.raw_channel_weights = nn.Parameter(torch.randn(self.bands), requires_grad=True)
        self.encoder = nn.Sequential(
            nn.Linear(self.bands, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.bands),
            nn.BatchNorm1d(self.bands),
            nn.Sigmoid()
        )

    def get_channel_weights(self):
        return torch.sigmoid(self.raw_channel_weights)

    def forward(self, X):
        channel_weights = self.get_channel_weights()
        channel_weights_ = channel_weights.expand(X.shape[0],-1)
        reweight_out = X * channel_weights_
        output = self.encoder(reweight_out)
        return channel_weights, output

    def get_l1_loss(self):
        norms = torch.norm(self.get_channel_weights(), p=1)
        return torch.sum(norms)




