import torch.nn as nn
import torch
import torch.nn.functional as F
from algorithms.fscr.band_index import BandIndex
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)


class ANN(nn.Module):
    def __init__(self, target_feature_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_feature_size = target_feature_size
        first_hidden = 50
        #[10, 100, 200, 1000, 2000]
        if target_feature_size == 100:
            first_hidden = 15
        elif target_feature_size == 200:
            first_hidden = 15
        elif target_feature_size == 1000:
            first_hidden = 8
        elif target_feature_size == 2000:
            first_hidden = 4
        print(f"ANN - {target_feature_size} - {first_hidden}")
        self.linear = nn.Sequential(
            nn.Linear(self.target_feature_size, first_hidden),
            nn.LeakyReLU(),
            nn.Linear(first_hidden, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )
        modules = []
        for i in range(self.target_feature_size):
            modules.append(BandIndex())
        self.machines = nn.ModuleList(modules)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, x):
        outputs = torch.zeros(x.shape[0], self.target_feature_size, dtype=torch.float32).to(self.device)
        x = x.permute(1,0)
        indices = torch.linspace(0, 1, x.shape[0]).to(self.device)
        coeffs = natural_cubic_spline_coeffs(indices, x)
        spline = NaturalCubicSpline(coeffs)
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

