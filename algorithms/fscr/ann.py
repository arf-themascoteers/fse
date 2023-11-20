import torch.nn as nn
import torch
import torch.nn.functional as F
from algorithms.fscr.band_index import BandIndex
import configs


class ANN(nn.Module):
    def __init__(self, target_feature_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_feature_size = target_feature_size
        h1,h2 = configs.get_hidden(target_feature_size)
        print("ANN",target_feature_size, h1,h2)
        self.linear = nn.Sequential(
            nn.Linear(self.target_feature_size, h1),
            nn.LeakyReLU(),
            nn.Linear(h1, h2),
            nn.LeakyReLU(),
            nn.Linear(h2, 1)
        )
        modules = []
        for i in range(self.target_feature_size):
            modules.append(BandIndex())
        self.machines = nn.ModuleList(modules)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, spline, size):
        outputs = torch.zeros(size, self.target_feature_size, dtype=torch.float32).to(self.device)
        for i,machine in enumerate(self.machines):
            outputs[:,i] = machine(spline)
        soc_hat = self.linear(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def retention_loss(self):
        loss = None
        for i in range(1, len(self.machines)):
            later_band = self.machines[i].raw_index
            past_band = self.machines[i-1].raw_index
            this_loss = F.relu(past_band-later_band)
            if loss is None:
                loss = this_loss
            else:
                loss = loss + this_loss
        return loss

    def get_indices(self):
        return [machine.index_value() for machine in self.machines]

