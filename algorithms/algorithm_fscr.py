from algorithm import Algorithm
from algorithms.fscr.fscr import FSCR


class AlgorithmFSCR(Algorithm):
    def __init__(self, task, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(task, target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        fscr = FSCR(self.target_feature_size)
        fscr.fit(self.X_train, self.y_train, self.X_validation, self.y_validation)
        return fscr, fscr.get_indices()