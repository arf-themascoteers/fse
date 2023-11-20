import torch.nn as nn
import torch
import configs


class ModelANN(nn.Module):
    def __init__(self, target_feature_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_feature_size = target_feature_size
        h1,h2 = configs.get_hidden(target_feature_size)
        print("ANN-Model",h1,h2)
        self.linear = nn.Sequential(
            nn.Linear(self.target_feature_size, h1),
            nn.LeakyReLU(),
            nn.Linear(h1, h2),
            nn.LeakyReLU(),
            nn.Linear(h2, 1)
        )
        self.epoch = 1500
        self.lr = 0.001

    def forward(self, X):
        return self.linear(X).reshape(-1)

    def create_optimizer(self):
        weight_decay = self.lr/10
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

    def fit(self, X, y):
        self.train()
        optimizer = self.create_optimizer()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        for epoch in range(self.epochs):
            y_hat = self.model(X)
            loss = self.criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def predict(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        return self.model(X)

