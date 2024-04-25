from algorithm import Algorithm
from algorithms.fscr.fscr import FSCR
import torch


class AlgorithmFSCRD(Algorithm):
    def __init__(self, task, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(task, target_feature_size, X_train, y_train, X_validation, y_validation)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

    def get_selected_indices(self):
        fscr = FSCR(self.X_train.shape[0], self.target_feature_size)
        fscr.fit(self.X_train, self.y_train, self.X_validation, self.y_validation)
        return fscr, fscr.get_indices()