from algorithm import Algorithm
from sklearn.decomposition import PCA


class AlgorithmPCA(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)

    def get_selected_indices(self):
        pca = PCA(n_components=self.target_size)
        pca.fit(self.splits.train_x)
        return pca,[]

    def get_name(self):
        return "pca"