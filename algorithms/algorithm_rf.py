from algorithm import Algorithm
from auswahl import RandomFrog


class AlgorithmRF(Algorithm):
    def __init__(self, X_train, y_train, X_validation, y_validation, target_feature_size):
        super().__init__(X_train, y_train, X_validation, y_validation, target_feature_size)

    def get_selected_indices(self):
        selector = RandomFrog(n_features_to_select=self.target_feature_size)
        selector.fit(self.X_train, self.y_train)
        return selector, selector.get_support(indices=True)
