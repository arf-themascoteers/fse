from algorithm import Algorithm
from sklearn.linear_model import LogisticRegression
import numpy as np


class AlgorithmLogistic(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)
        self.indices = None

    def get_selected_indices(self):
        logistic = LogisticRegression(penalty="l1")
        logistic.fit(self.train_x, self.train_y)
        self.indices = np.argsort(np.abs(logistic.coef_))[::-1][:self.target_size]
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]

    def get_name(self):
        return "logistic"