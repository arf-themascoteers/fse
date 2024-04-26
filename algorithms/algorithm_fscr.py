from algorithm import Algorithm
from algorithms.fscr.fscr import FSCR
import numpy as np
from ds_manager import DSManager


class AlgorithmFSCR(Algorithm):
    def __init__(self, target_feature_size, splits):
        super().__init__(target_feature_size, splits)
        self.task = DSManager.get_task_by_name(splits.get_name())

    def get_selected_indices(self):
        class_size = 1
        if self.task == "classification":
            class_size = len(np.unique(self.y_train))
        fscr = FSCR(self.target_feature_size, class_size)
        fscr.fit(self.X_train, self.y_train, self.X_validation, self.y_validation)
        return fscr, fscr.get_indices()

    def get_name(self):
        return "fsdr"