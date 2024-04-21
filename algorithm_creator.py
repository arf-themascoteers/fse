from algorithms.algorithm_pca import AlgorithmPCA
from algorithms.algorithm_lle import AlgorithmLLE
from algorithms.algorithm_pcat95 import AlgorithmPCAT95
from algorithms.algorithm_rfe import AlgorithmRFE
from algorithms.algorithm_fscr import AlgorithmFSCR
from algorithms.algorithm_fscrns import AlgorithmFSCRNS
from algorithms.algorithm_sfs import AlgorithmSFS
from algorithms.algorithm_sbs import AlgorithmSBS
from algorithms.algorithm_kbest import AlgorithmKBest
from algorithms.algorithm_fm import AlgorithmFM
from algorithms.algorithm_tbfi import AlgorithmTBFI
from algorithms.algorithm_pls import AlgorithmPLS
from algorithms.algorithm_ex import AlgorithmEx
from algorithms.algorithm_mi import AlgorithmMI
from algorithms.algorithm_lasso import AlgorithmLasso
from algorithms.algorithm_cars import AlgorithmCARS
from algorithms.algorithm_spa import AlgorithmSPA
from algorithms.algorithm_mcuve import AlgorithmMCUVE
from algorithms.algorithm_rf import AlgorithmRF


class AlgorithmCreator:
    @staticmethod
    def create(name, target_feature_size, X_train, y_train, X_validation, y_validation):
        if name == "pca":
            return AlgorithmPCA(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "lle":
            return AlgorithmLLE(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "pcat95":
            return AlgorithmPCAT95(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "rfe":
            return AlgorithmRFE(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "fsdr":
            return AlgorithmFSCR(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "fscrns":
            return AlgorithmFSCRNS(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "sfs":
            return AlgorithmSFS(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "sbs":
            return AlgorithmSBS(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "fm":
            return AlgorithmFM(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "kbest":
            return AlgorithmKBest(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "tbfi":
            return AlgorithmTBFI(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "pls":
            return AlgorithmPLS(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "ex":
            return AlgorithmEx(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "mi":
            return AlgorithmMI(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "lasso":
            return AlgorithmLasso(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "cars":
            return AlgorithmCARS(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "spa":
            return AlgorithmSPA(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "mcuve":
            return AlgorithmMCUVE(target_feature_size, X_train, y_train, X_validation, y_validation)
        elif name == "rf":
            return AlgorithmRF(target_feature_size, X_train, y_train, X_validation, y_validation)

