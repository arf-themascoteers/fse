from algorithm import Algorithm
from sklearn.linear_model import LogisticRegression
import numpy as np


class AlgorithmLogistic(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)
        self.indices = None

    def get_selected_indices(self):
        logistic = LogisticRegression(penalty="l1")
        logistic.fit(self.X_train, self.y_train)
        self.indices = np.argsort(np.abs(logistic.coef_))[::-1][:self.target_feature_size]
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]

    def get_name(self):
        return "logistic"