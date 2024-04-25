from algorithm import Algorithm
from sklearn.linear_model import Lasso
import numpy as np


class AlgorithmLasso(Algorithm):
    def __init__(self, task, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(task, target_feature_size, X_train, y_train, X_validation, y_validation)
        self.indices = None

    def get_selected_indices(self):
        lasso = Lasso(alpha=0.001)
        lasso.fit(self.X_train, self.y_train)
        self.indices = np.argsort(np.abs(lasso.coef_))[::-1][:self.target_feature_size]
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]