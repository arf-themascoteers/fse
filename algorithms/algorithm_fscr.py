from algorithm import Algorithm
from algorithms.fscr.fscr import FSCR
import numpy as np


class AlgorithmFSCR(Algorithm):
    def __init__(self, target_feature_size, task, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)
        self.task = task

    def get_selected_indices(self):
        class_size = 1
        if self.task == "classification":
            class_size = len(np.unique(self.y_train))
        fscr = FSCR(self.target_feature_size, class_size)
        fscr.fit(self.X_train, self.y_train, self.X_validation, self.y_validation)
        return fscr, fscr.get_indices()

    def get_name(self):
        return "fsdr"