import math
from sklearn.metrics import mean_squared_error, r2_score
from algorithms.fscr.spectral_dataset import SpectralDataset
import torch
from torch.utils.data import DataLoader
from algorithms.fscr.ann import ANN
from datetime import datetime
import os


class FSCR:
    def __init__(self, target_feature_size):
        self.target_feature_size = target_feature_size
        self.lr = 0.001
        self.model = ANN(self.target_feature_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.batch_size = 1000
        self.epochs = 50
        if self.target_feature_size > 1000:
            self.epochs = 50
        self.csv_file = os.path.join("results", "fscr.csv")
        self.original_feature_size = None
        self.start_time = datetime.now()

    def get_elapsed_time(self):
        return (datetime.now() - self.start_time).total_seconds()

    def create_optimizer(self):
        weight_decay = self.lr/10
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def fit(self, X_train, y_train):
        self.original_feature_size = X_train.shape[1]
        self.train_dataset = SpectralDataset(X_train, y_train)
        self.write_columns()
        self.model.train()
        optimizer = self.create_optimizer()
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            rows = []
            for batch_number,(x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                for machine in self.model.machines:
                    loss = loss + machine.range_loss()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            row = self.dump_row(epoch)
            rows.append(row)
            print("".join([str(i).ljust(20) for i in row]))
        return self.model

    def evaluate(self,dataset):
        batch_size = 30000
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x)
            y_hat = y_hat.reshape(-1)
            y_hat = y_hat.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            r2 = r2_score(y, y_hat)
            rmse = math.sqrt(mean_squared_error(y, y_hat))
            self.model.train()
            return max(r2,0), rmse

    def train_results(self):
        return self.evaluate(self.train_dataset)

    def write_columns(self):
        columns = ["epoch","train_r2","train_rmse","time"]
        for index,p in enumerate(self.model.get_indices()):
            columns.append(f"band_{index+1}")
        print("".join([c.ljust(20) for c in columns]))
        with open(self.csv_file, 'w') as file:
            file.write(",".join(columns))
            file.write("\n")

    def dump_row(self, epoch):
        train_r2, train_rmse = self.train_results()
        row = [train_r2, train_rmse]
        row = [round(r,5) for r in row]
        row = [epoch] + row + [self.get_elapsed_time()]
        for p in self.model.get_indices():
            row.append(self.indexify_raw_index(p))
        with open(self.csv_file, 'a') as file:
            file.write(",".join([f"{x}" for x in row]))
            file.write("\n")
        return row

    def indexify_raw_index(self, raw_index):
        return round(raw_index.item() * self.original_feature_size)

    def get_indices(self):
        return [self.indexify_raw_index(p) for p in self.model.get_indices()]

    def transform(self, X):
        return X[:,self.get_indices()]
