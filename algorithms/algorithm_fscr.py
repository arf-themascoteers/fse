from algorithm import Algorithm
from algorithms.fscr.fscr import FSCR


class AlgorithmFSCR(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selector(self):
        fscr = FSCR(self.target_feature_size)
        fscr.fit(self.X_train, self.y_train)
        return fscr, fscr.get_indices()