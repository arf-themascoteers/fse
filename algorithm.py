from abc import ABC, abstractmethod


class Algorithm(ABC):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        self.X_train, self.y_train, self.X_validation, self.y_validation = X_train, y_train, X_validation, y_validation
        self.target_feature_size = target_feature_size
        self.selected_indices = []
        self.model = None

    def fit(self):
        self.model, self.selected_indices = self.get_selected_indices()
        return self.selected_indices

    def transform(self, X):
        if len(self.selected_indices) != 0:
            return X[:,self.selected_indices]
        return self.model.transform(X)

    @abstractmethod
    def get_selected_indices(self):
        pass
