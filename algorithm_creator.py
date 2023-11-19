from algorithms.algorithm_pca import AlgorithmPCA
from algorithms.algorithm_pcat95 import AlgorithmPCAT95
from algorithms.algorithm_rfe import AlgorithmRFE


class AlgorithmCreator:
    @staticmethod
    def create(name, X_train, y_train, target_feature_size):
        if name == "pca":
            return AlgorithmPCA(X_train, y_train, target_feature_size)
        if name == "pcat95":
            return AlgorithmPCAT95(X_train, y_train, target_feature_size)
        elif name == "rfe":
            return AlgorithmRFE(X_train, y_train, target_feature_size)