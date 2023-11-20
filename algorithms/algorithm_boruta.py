from algorithm import Algorithm
import configs
from boruta import BorutaPy
import numpy as np


class AlgorithmBoruta(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selector(self):
        model = configs.get_internal_model()
        boruta_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=42)
        boruta_selector.fit(self.X_train, self.y_train)
        selected_feature_indices = np.where(boruta_selector.support_)[0]
        return boruta_selector, selected_feature_indices
