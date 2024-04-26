from algorithm import Algorithm
from algorithms.bsnetig2.bs_net_fc2 import BSNetFC2
import torch
from torch.utils.data import TensorDataset, DataLoader


class AlgorithmBSNetIG2(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def get_selected_indices(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bsnet = BSNetFC2(self.train_x.shape[1]).to(device)
        optimizer = torch.optim.Adam(bsnet.parameters(), lr=0.00002)
        X_train = torch.tensor(self.train_x, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_train, X_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        channel_weights = None
        for epoch in range(100):
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                channel_weights, y_hat = bsnet(X)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = bsnet.get_l1_loss()
                loss = mse_loss + l1_loss
                print(f"Epoch={epoch} Batch={batch_idx} - MSE={round(mse_loss.item(),5)}, L1={round(l1_loss.item(),5)}, LOSS={round(loss.item(),5)}")
                loss.backward()
                optimizer.step()

        gradients = self.input_gradient(bsnet)
        gradients = torch.mean(torch.abs(gradients), dim=0)
        band_indx = (torch.argsort(gradients, descending=True)[:self.target_size]).tolist()
        return bsnet, band_indx

    def input_gradient(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = torch.tensor(self.train_x, dtype=torch.float32).to(device)
        X_train2 = torch.tensor(self.train_x, dtype=torch.float32).to(device)
        y_train = torch.tensor(self.train_x, dtype=torch.float32).to(device)
        X_train2.requires_grad_()

        channel_weights = model.bam(X_train)
        channel_weights, output = model.recnet(channel_weights, X_train2)
        mse_loss = self.criterion(output, y_train)
        mse_loss.backward()
        return X_train2.grad

    def get_name(self):
        return "bsnetig2"