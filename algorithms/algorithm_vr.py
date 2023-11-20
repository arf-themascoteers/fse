from algorithm import Algorithm
from sklearn.feature_selection import VarianceThreshold


class AlgorithmVT(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selector(self):
        selector = VarianceThreshold(threshold=0.5)
        selector.fit(self.X_train)
        selected_features = selector.get_support(indices=True)
        return selector, selected_features