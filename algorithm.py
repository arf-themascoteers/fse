from abc import ABC, abstractmethod


class Algorithm(ABC):
    def __init__(self, target_feature_size, splits):
        self.X_train = splits.train_x
        self.y_train = splits.train_y
        self.X_validation = splits.validation_x
        self.y_validation = splits.validation_y
        self.target_feature_size = target_feature_size
        self.selected_indices = []
        self.model = None
        self.all_indices = None

    def fit(self):
        self.model, self.selected_indices = self.get_selected_indices()
        return sorted(self.selected_indices)

    def transform(self, X):
        if len(self.selected_indices) != 0:
            return X[:,self.selected_indices]
        return self.model.transform(X)

    @abstractmethod
    def get_selected_indices(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_all_indices(self):
        return self.all_indices

    def _set_all_indices(self, all_indices):
        self.all_indices = all_indices

    def is_independent_of_target_size(self):
        return self.get_all_indices() is not None