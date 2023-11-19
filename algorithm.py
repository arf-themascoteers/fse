from abc import ABC, abstractmethod


class Algorithm(ABC):
    def __init__(self, X_train, y_train, target_feature_size):
        self.X_train = X_train
        self.y_train = y_train
        self.target_feature_size = target_feature_size

    @abstractmethod
    def get_selector(self):
        pass
