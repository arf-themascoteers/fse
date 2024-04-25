from algorithm import Algorithm
from algorithms.bsnet.bs_net_fc import BSNetFC
import torch
from torch.utils.data import TensorDataset, DataLoader


class AlgorithmBSNet(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bsnet = BSNetFC(self.X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(bsnet.parameters(), lr=0.00002)
        criterion = torch.nn.MSELoss(reduction='sum')
        X_train = torch.tensor(self.X_train, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_train, X_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        channel_weights = None
        for epoch in range(100):
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                channel_weights, y_hat = bsnet(X)
                mse_loss = criterion(y_hat, y)
                l1_loss = bsnet.get_l1_loss()
                loss = mse_loss + 0.01*l1_loss
                print(f"Epoch={epoch} Batch={batch_idx} - MSE={round(mse_loss.item(),5)}, L1={round(l1_loss.item(),5)}, LOSS={round(loss.item(),5)}")
                loss.backward()
                optimizer.step()
        mean_weight = torch.mean(channel_weights, dim=0)
        band_indx = (torch.argsort(mean_weight, descending=True)[:self.target_feature_size]).tolist()
        return bsnet, band_indx