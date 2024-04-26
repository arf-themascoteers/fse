from algorithm import Algorithm
from sklearn.decomposition import PCA
import numpy as np


class AlgorithmPCALoading(Algorithm):
    def __init__(self, target_feature_size, splits):
        super().__init__(target_feature_size, splits)
        self.indices = None

    def get_selected_indices(self):
        pca = PCA(n_components=self.target_feature_size)
        pca.fit(self.X_train)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        feature_importance = np.sum(np.abs(loadings), axis=1)
        feature_ranking = np.argsort(feature_importance)[::-1]
        self.indices = feature_ranking[:self.target_feature_size]
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]

    def get_name(self):
        return "pcal"