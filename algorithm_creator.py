from algorithms.algorithm_fscr import AlgorithmFSCR
from algorithms.algorithm_fscrd import AlgorithmFSCRD
from algorithms.algorithm_lasso import AlgorithmLasso
from algorithms.algorithm_logistic import AlgorithmLogistic
from algorithms.algorithm_cars import AlgorithmCARS
from algorithms.algorithm_spa import AlgorithmSPA
from algorithms.algorithm_mcuve import AlgorithmMCUVE


class AlgorithmCreator:
    @staticmethod
    def create(name, task, target_feature_size, X_train, y_train, X_validation, y_validation):
        if name == "fsdr":
            return AlgorithmFSCR(task, target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "lasso":
            return AlgorithmLasso(task, target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "logistic":
            return AlgorithmLogistic(task, target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "cars":
            return AlgorithmCARS(task, target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "spa":
            return AlgorithmSPA(task, target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "mcuve":
            return AlgorithmMCUVE(task, target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "fsdrd":
            return AlgorithmFSCRD(task, target_feature_size, X_train, y_train, X_validation, y_validation)

