from algorithm import Algorithm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import configs


class AlgorithmSFS(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selector(self):
        sfs = SFS(configs.get_internal_model(),
                  k_features=self.target_feature_size,
                  forward=True, floating=False, scoring='r2', cv=5)
        sfs.fit(self.X_train, self.y_train)
        return sfs, sfs.k_feature_idx_