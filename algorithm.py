from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class Algorithm(ABC):
    def __init__(self, X_train, y_train, target_feature_size):
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=40)
        self.target_feature_size = target_feature_size

    @abstractmethod
    def get_selected_indices(self):
        pass
