from sklearn.decomposition import PCA
import numpy as np
from ds_manager import DSManager
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ds = DSManager(dataset="brazilian")
train_X, train_y, test_X, test_y = ds.get_train_test_X_y()
print(train_X.shape[0])
print(test_X.shape[0])

pca = PCA(n_components=7)
X_pca = pca.fit_transform(train_X)
rf = RandomForestRegressor()
#rf = LinearRegression()
rf.fit(X_pca, train_y)
print(rf.score(X_pca, train_y))

X_pca_test = pca.fit_transform(test_X)
rf.fit(X_pca_test, test_y)
print(rf.score(X_pca_test, test_y))

