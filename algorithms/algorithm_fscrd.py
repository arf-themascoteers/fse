from algorithms.algorithm_fscr import AlgorithmFSCR
import torch


class AlgorithmFSCRD(AlgorithmFSCR):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

    def get_name(self):
        return "fsdrs"