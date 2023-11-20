from algorithm import Algorithm
from sklearn.feature_selection import RFE
import configs


class AlgorithmRFE(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        rfe = RFE(estimator=configs.get_internal_model(), n_features_to_select= self.target_feature_size)
        rfe.fit(self.X_train, self.y_train)
        indices = [i for i, selected in enumerate(rfe.support_) if selected]
        return indices