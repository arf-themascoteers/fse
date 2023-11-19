from algorithm import Algorithm
from sklearn.decomposition import PCA


class AlgorithmPCA(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selector(self):
        pca = PCA(n_components=self.target_feature_size)
        pca.fit(self.X_train)
        return pca, []