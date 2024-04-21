from algorithm import Algorithm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import my_utils


class AlgorithmSFS(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        sfs = SFS(my_utils.get_internal_model(),
                  k_features=self.target_feature_size,
                  forward=True, floating=False, scoring='r2', cv=5)
        sfs.fit(self.X_train, self.y_train)
        return sfs, sfs.k_feature_idx_