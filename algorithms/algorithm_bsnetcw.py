from algorithm import Algorithm
from algorithms.bsnetcw.bs_net_fc_cw import BSNetFCCW
import torch
from torch.utils.data import TensorDataset, DataLoader


class AlgorithmBSNetCW(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def get_selected_indices(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bsnet = BSNetFCCW(self.splits.train_x.shape[1]).to(device)
        optimizer = torch.optim.Adam(bsnet.parameters(), lr=0.00002)
        X_train = torch.tensor(self.splits.train_x, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_train, X_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0
        for epoch in range(100):
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                channel_weights, y_hat = bsnet(X)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = bsnet.get_l1_loss()
                loss = mse_loss + l1_loss * 0.01
                loss.backward()
                optimizer.step()
            print(f"Epoch={epoch} MSE={round(mse_loss.item(), 5)}, L1={round(l1_loss.item(), 5)}, LOSS={round(loss.item(), 5)}")
            print(f"Channel Weights: {bsnet.get_channel_weights()[0:5]}")
        band_indx = (torch.argsort(bsnet.get_channel_weights(), descending=True)).tolist()
        super()._set_all_indices(band_indx)
        selected_indices = band_indx[: self.target_size]
        print(bsnet.get_channel_weights())
        return bsnet, selected_indices

    def get_name(self):
        return "bsnetcw"