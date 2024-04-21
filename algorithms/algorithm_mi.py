from algorithm import Algorithm
from sklearn.feature_selection import mutual_info_regression
import numpy as np


class AlgorithmMI(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)
        self.indices = None

    def get_selected_indices(self):
        mi_scores = mutual_info_regression(self.X_train, self.y_train)
        self.indices = np.argsort(mi_scores)[::-1][:self.target_feature_size]
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]