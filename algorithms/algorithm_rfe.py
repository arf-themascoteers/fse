from algorithm import Algorithm
from sklearn.feature_selection import RFE
import my_utils


class AlgorithmRFE(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        rfe = RFE(estimator=my_utils.get_internal_model(), n_features_to_select= self.target_feature_size)
        rfe.fit(self.X_train, self.y_train)
        indices = [i for i, selected in enumerate(rfe.support_) if selected]
        return rfe, indices