from algorithm import Algorithm
from mlxtend.feature_selection import ExhaustiveFeatureSelector
import my_utils


class AlgorithmEx(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        selector = ExhaustiveFeatureSelector(my_utils.get_internal_model(),
                                             min_features=1,
                                             max_features=self.target_feature_size,
                                             scoring='neg_mean_squared_error', print_progress=True)

        selector.fit(self.X_train, self.y_train)
        return selector, selector.best_idx_