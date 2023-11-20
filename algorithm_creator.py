from algorithms.algorithm_pca import AlgorithmPCA
from algorithms.algorithm_pcat95 import AlgorithmPCAT95
from algorithms.algorithm_rfe import AlgorithmRFE
from algorithms.algorithm_fscr import AlgorithmFSCR
from algorithms.algorithm_sfs import AlgorithmSFS
from algorithms.algorithm_sbs import AlgorithmSBS
from algorithms.algorithm_kbest import AlgorithmKBest
from algorithms.algorithm_fm import AlgorithmFM
from algorithms.algorithm_tbfi import AlgorithmTBFI


class AlgorithmCreator:
    @staticmethod
    def create(name, X_train, y_train, target_feature_size):
        if name == "pca":
            return AlgorithmPCA(X_train, y_train, target_feature_size)
        elif name == "pcat95":
            return AlgorithmPCAT95(X_train, y_train, target_feature_size)
        elif name == "rfe":
            return AlgorithmRFE(X_train, y_train, target_feature_size)
        elif name == "fscr":
            return AlgorithmFSCR(X_train, y_train, target_feature_size)
        elif name == "sfs":
            return AlgorithmSFS(X_train, y_train, target_feature_size)
        elif name == "sbs":
            return AlgorithmSBS(X_train, y_train, target_feature_size)
        elif name == "fm":
            return AlgorithmFM(X_train, y_train, target_feature_size)
        elif name == "kbest":
            return AlgorithmKBest(X_train, y_train, target_feature_size)
        elif name == "tbfi":
            return AlgorithmTBFI(X_train, y_train, target_feature_size)

