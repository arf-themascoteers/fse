from algorithm import Algorithm
from sklearn.linear_model import Lasso
import numpy as np


class AlgorithmLasso(Algorithm):
    def __init__(self, target_feature_size, splits):
        super().__init__(target_feature_size, splits)
        self.indices = None

    def get_selected_indices(self):
        lasso = Lasso(alpha=0.001)
        lasso.fit(self.X_train, self.y_train)
        self.indices = np.argsort(np.abs(lasso.coef_))[::-1][:self.target_feature_size]
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]

    def get_name(self):
        return "lasso"