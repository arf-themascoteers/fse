from algorithms.algorithm_bsnet import AlgorithmBSNet
import torch


class AlgorithmBSNetIG(AlgorithmBSNet):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        model, indices = super().get_selected_indices()
        gradients = self.input_gradient(model)
        gradients = torch.mean(torch.abs(gradients), dim=0)
        band_indx = (torch.argsort(gradients, descending=True)[:self.target_feature_size]).tolist()
        return model, band_indx

    def input_gradient(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = torch.tensor(self.X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(self.X_train, dtype=torch.float32).to(device)
        X_train.requires_grad_()
        _,output = model(X_train)
        mse_loss = self.criterion(output, y_train)
        l1_loss = model.get_l1_loss()
        loss = mse_loss + l1_loss
        loss.backward()
        return X_train.grad
