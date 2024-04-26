from algorithm import Algorithm
from auswahl import MCUVE


class AlgorithmMCUVE(Algorithm):
    def __init__(self, target_feature_size, splits):
        super().__init__(target_feature_size, splits)

    def get_selected_indices(self):
        selector = MCUVE(n_features_to_select=self.target_feature_size)
        selector.fit(self.X_train, self.y_train)
        return selector, selector.get_support(indices=True)

    def get_name(self):
        return "mcuve"