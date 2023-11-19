from algorithm import Algorithm
import configs
from sklearn.feature_selection import SelectFromModel
import numpy as np

class AlgorithmFM(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selector(self):
        selector = SelectFromModel(configs.get_internal_model(),
                                   threshold='median', max_features=self.target_feature_size)
        selector.fit(self.X_train, self.y_train)
        indices = np.where(selector.get_support())[0]
        return selector, indices