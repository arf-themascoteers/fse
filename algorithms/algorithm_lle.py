from algorithm import Algorithm
from sklearn.manifold import LocallyLinearEmbedding


class AlgorithmLLE(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        lle = LocallyLinearEmbedding(n_neighbors=3, n_components=self.target_feature_size)
        lle.fit(self.X_train)
        return lle,[]