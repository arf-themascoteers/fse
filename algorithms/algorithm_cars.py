from algorithm import Algorithm
from auswahl import CARS


class AlgorithmCARS(Algorithm):
    def __init__(self, target_feature_size, splits):
        super().__init__(target_feature_size, splits)

    def get_selected_indices(self):
        selector = CARS(n_features_to_select=self.target_feature_size)
        selector.fit(self.X_train, self.y_train)
        return selector, selector.get_support(indices=True)

    def get_name(self):
        return "cars"