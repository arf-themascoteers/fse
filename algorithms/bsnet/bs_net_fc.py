import torch.nn as nn
import torch


class BSNetFC(nn.Module):
    def __init__(self, bands):
        super().__init__()
        torch.manual_seed(3)
        self.bands = bands
        self.weighter = nn.Sequential(
            nn.BatchNorm1d(self.bands),
            nn.Linear(self.bands, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.channel_weight_layer = nn.Sequential(
            nn.Linear(128, self.bands),
            nn.Sigmoid()
        )
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

    def forward(self, X):
        channel_weights = self.weighter(X)
        channel_weights = self.channel_weight_layer(channel_weights)
        channel_weights_ = torch.reshape(channel_weights, (-1, self.bands))
        reweight_out = X * channel_weights_
        output = self.encoder(reweight_out)
        return channel_weights, output

    def get_l1_loss(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in self.channel_weight_layer.parameters():
            l1_reg = l1_reg + torch.norm(param, p=1)
        return l1_reg*0.01




