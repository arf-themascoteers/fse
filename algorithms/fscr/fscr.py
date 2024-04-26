import math
from sklearn.metrics import mean_squared_error, r2_score
from approximator import get_splines
import torch
from algorithms.fscr.ann import ANN
from datetime import datetime
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import numpy as np


class FSCR:
    def __init__(self, target_size, class_size=1):
        self.target_size = target_size
        self.class_size = class_size
        self.lr = 0.001
        self.model = ANN(self.target_size, self.class_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = self.get_criterion()
        self.epochs = 4000
        self.csv_file = os.path.join("results", f"fscr-{target_size}-{str(datetime.now().timestamp()).replace('.','')}.csv")
        self.original_feature_size = None
        self.start_time = datetime.now()

    def get_criterion(self):
        if self.is_regression():
            return torch.nn.MSELoss(reduction='mean')
        return torch.nn.CrossEntropyLoss()

    def is_regression(self):
        return self.class_size == 1

    def get_elapsed_time(self):
        elapsed_time = round((datetime.now() - self.start_time).total_seconds(),2)
        return float(elapsed_time)

    def create_optimizer(self):
        weight_decay = self.lr/10
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def fit(self, X, y, X_validation, y_validation):
        row_size = X.shape[0]
        row_test_size = X_validation.shape[0]
        self.original_feature_size = X.shape[1]
        self.write_columns()
        self.model.train()
        optimizer = self.create_optimizer()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        spline = get_splines(X, self.device)
        X_validation = torch.tensor(X_validation, dtype=torch.float32).to(self.device)
        spline_validation = get_splines(X_validation, self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        y_validation = torch.tensor(y_validation, dtype=torch.float32).to(self.device)
        if not self.is_regression():
            y = y.type(torch.LongTensor).to(self.device)
            y_validation = y_validation.type(torch.LongTensor).to(self.device)
        for epoch in range(self.epochs):
            y_hat = self.model(spline, row_size)
            loss = self.criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            row = self.dump_row(epoch, spline, y, spline_validation, y_validation, row_size, row_test_size)
            if epoch%50 == 0:
                print("".join([str(i).ljust(20) for i in row]))
        return self.get_indices()

    def evaluate(self,spline,y,size):
        self.model.eval()
        y_hat = self.model(spline, size)
        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        if self.is_regression():
            y_hat = y_hat.reshape(-1)
            r2 = r2_score(y, y_hat)
            rmse = math.sqrt(mean_squared_error(y, y_hat))
            self.model.train()
            return max(r2,0), rmse
        y_hat = np.argmax(y_hat, axis=1)
        accuracy = accuracy_score(y, y_hat)
        kappa = cohen_kappa_score(y, y_hat)
        self.model.train()
        return accuracy, kappa

    def get_metric1(self):
        if self.is_regression():
            return "r2"
        return "accuracy"

    def get_metric2(self):
        if self.is_regression():
            return "rmse"
        return "kappa"

    def write_columns(self):
        columns = ["epoch",
                   f"train_{self.get_metric1()}",f"validation_{self.get_metric1()}",
                   f"train_{self.get_metric2()}",f"validation_{self.get_metric2()}",
                   "time"]
        for index,p in enumerate(self.model.get_indices()):
            columns.append(f"band_{index+1}")
        print("".join([c.ljust(20) for c in columns]))
        with open(self.csv_file, 'w') as file:
            file.write(",".join(columns))
            file.write("\n")

    def dump_row(self, epoch, spline, y, spline_test, y_test, row_size, row_test_size):
        train_metric1, train_metric2 = self.evaluate(spline, y, row_size)
        test_metric1, test_metric2 = self.evaluate(spline_test, y_test, row_test_size)
        row = [train_metric1, test_metric1, train_metric2, test_metric2]
        row = [round(r,5) for r in row]
        elapsed_time = self.get_elapsed_time()
        row = [epoch] + row + [elapsed_time]
        for p in self.model.get_indices():
            row.append(self.indexify_raw_index(p))
        with open(self.csv_file, 'a') as file:
            file.write(",".join([f"{x}" for x in row]))
            file.write("\n")
        return row

    def indexify_raw_index(self, raw_index):
        multiplier = self.original_feature_size
        return round(raw_index.item() * multiplier)

    def get_indices(self):
        indices = sorted([self.indexify_raw_index(p) for p in self.model.get_indices()])
        return list(dict.fromkeys(indices))

    def transform(self, X):
        return X[:,self.get_indices()]
