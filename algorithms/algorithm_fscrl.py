from algorithm import Algorithm
from algorithms.fscrl.fscrl import FSCRL
import numpy as np
from ds_manager import DSManager


class AlgorithmFSCRL(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)
        self.task = DSManager.get_task_by_name(splits.get_name())

    def get_selected_indices(self):
        class_size = 1
        if self.task == "classification":
            class_size = len(np.unique(self.splits.train_y))
        fscr = FSCRL(self.target_size, class_size)
        fscr.fit(self.splits.train_x, self.splits.train_y, self.splits.validation_x, self.splits.validation_y)
        return fscr, fscr.get_indices()

    def get_name(self):
        return "fsdr"