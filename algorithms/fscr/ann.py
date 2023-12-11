import torch.nn as nn
import torch


class ANN(nn.Module):
    def __init__(self, target_feature_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_feature_size = target_feature_size
        self.indexer = nn.Sequential(
            nn.Linear(4200,10),
            nn.LeakyReLU(),
            nn.Linear(10,target_feature_size),
            nn.Sigmoid()
        )

        self.linear = nn.Sequential(
            nn.Linear(target_feature_size,10),
            nn.LeakyReLU(),
            nn.Linear(10,1)
        )

        self.x = None

    def forward(self, x, spline):
        x = self.indexer(x)
        self.x = torch.mean(x, dim=0)
        outputs = torch.cat([spline.evaluate(i).reshape(-1,1) for i in self.x], dim=1)
        soc_hat = self.linear(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_indices(self):
        if self.x is None:
            return [-1 for i in range(self.target_feature_size)]
        return self.x

