import torch.nn as nn
import torch


class ANNL(nn.Module):
    def __init__(self, target_size, class_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.class_size = class_size
        self.linear = nn.Sequential(
            nn.Linear(self.target_size, 15),
            nn.LeakyReLU(),
            nn.Linear(15, 10),
            nn.LeakyReLU(),
            nn.Linear(10, self.class_size)
        )
        init_vals = torch.linspace(0.001,0.99, target_size+2)
        self.indices = nn.Parameter(torch.tensor([ANNL.inverse_sigmoid_torch(init_vals[i + 1]) for i in range(self.target_size)], requires_grad=True).to(self.device))
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, linterp):
        outputs = linterp(self.get_indices())
        soc_hat = self.linear(outputs)
        if self.class_size == 1:
            soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_indices(self):
        return torch.sigmoid(self.indices)

