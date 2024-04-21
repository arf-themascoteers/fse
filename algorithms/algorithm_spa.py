from algorithm import Algorithm
from auswahl import SPA, VIP


class AlgorithmSPA(Algorithm):
    def __init__(self, X_train, y_train, X_validation, y_validation, target_feature_size):
        super().__init__(X_train, y_train, X_validation, y_validation, target_feature_size)

    def get_selected_indices(self):
        vip = VIP()
        selector = SPA(n_features_to_select=self.target_feature_size)
        vip.fit(self.X_train, self.y_train)
        mask = vip.vips_ > 0.3
        selector.fit(self.X_train, self.y_train, mask=mask)
        return selector, selector.get_support(indices=True)
