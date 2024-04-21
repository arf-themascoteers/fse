from algorithm import Algorithm
from sklearn.cross_decomposition import PLSRegression


class AlgorithmPLS(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        pls = PLSRegression(n_components=self.target_feature_size)
        pls.fit(self.X_train, self.y_train)
        return pls,[]