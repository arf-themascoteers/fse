from algorithm import Algorithm
from sklearn.ensemble import RandomForestRegressor


class AlgorithmTBFI(Algorithm):
    def __init__(self, target_feature_size, X_train, y_train, X_validation, y_validation):
        super().__init__(target_feature_size, X_train, y_train, X_validation, y_validation)

    def get_selected_indices(self):
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        selected_features_indices =  \
            model.feature_importances_.argsort()[-self.target_feature_size:][::-1]
        return model, selected_features_indices
