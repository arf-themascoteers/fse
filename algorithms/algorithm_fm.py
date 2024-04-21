from algorithm import Algorithm
import my_utils
from sklearn.feature_selection import SelectFromModel
import numpy as np


class AlgorithmFM(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        model = my_utils.get_internal_model()
        sfm = SelectFromModel(model, threshold=-np.inf, max_features=5)
        sfm.fit(self.X_train, self.y_train)
        selected_feature_indices = np.where(sfm.get_support())[0]
        return sfm, selected_feature_indices
