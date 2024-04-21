from algorithm import Algorithm
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression


class AlgorithmPCA(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        rfecv = RFECV(estimator=LinearRegression(), step=1, cv=5, min_features_to_select= self.target_feature_size)
        rfecv.fit(self.X_train, self.y_train)
        indices = [i for i, selected in enumerate(rfecv.support_) if selected]
        return rfecv, indices