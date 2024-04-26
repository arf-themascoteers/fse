from algorithms.algorithm_fscr import AlgorithmFSCR
import torch


class AlgorithmFSCRD(AlgorithmFSCR):
    def __init__(self, target_feature_size, splits):
        super().__init__(target_feature_size, splits)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

    def get_name(self):
        return "fsdrs"