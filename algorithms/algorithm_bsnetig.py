from algorithms.algorithm_bsnet import AlgorithmBSNet
import torch


class AlgorithmBSNetIG(AlgorithmBSNet):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)

    def get_selected_indices(self):
        model, indices = super().get_selected_indices()
        gradients = self.input_gradient(model)
        gradients = torch.mean(torch.abs(gradients), dim=0)
        band_indx = (torch.argsort(gradients, descending=True)).tolist()
        super()._set_all_indices(band_indx)
        selected_indices = band_indx[: self.target_size]
        return model, selected_indices

    def input_gradient(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = torch.tensor(self.splits.train_x, dtype=torch.float32).to(device)
        y_train = torch.tensor(self.splits.train_x, dtype=torch.float32).to(device)
        X_train.requires_grad_()
        _,output = model(X_train)
        mse_loss = self.criterion(output, y_train)
        l1_loss = model.get_l1_loss()
        loss = mse_loss + l1_loss
        loss.backward()
        return X_train.grad

    def get_name(self):
        return "bsnetig"