from algorithm import Algorithm
from sklearn.decomposition import PCA


class AlgorithmPCA(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        pca = PCA(n_components=self.target_feature_size)
        pca.fit(self.X_train)
        return pca,[]