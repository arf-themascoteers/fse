from algorithm import Algorithm
from auswahl import CARS


class AlgorithmCARS(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)

    def get_selected_indices(self):
        selector = CARS(n_features_to_select=self.target_size)
        selector.fit(self.train_x, self.train_y)
        return selector, selector.get_support(indices=True)

    def get_name(self):
        return "cars"