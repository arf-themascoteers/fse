from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from ds_manager import DSManager

d = DSManager(name="lucas_full")
X, y = d.get_all_X_y()
X = X[:,[397,1399,2022,2325,3461]]


param_grid = {
    'C': [1e5,1e6,1e7],
    'gamma': [1,2]
}

grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, verbose=3)
grid_search.fit(X, y)
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
print(best_estimator)