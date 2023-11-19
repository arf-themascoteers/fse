from algorithm import Algorithm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


class AlgorithmPCA(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selector(self):
        rfe = RFE(estimator=LinearRegression(), n_features_to_select= self.target_feature_size)
        rfe.fit(self.X_train, self.y_train)
        indices = [i for i, selected in enumerate(rfe.support_) if selected]
        return rfe, indices