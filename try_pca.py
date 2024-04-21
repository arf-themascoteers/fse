from sklearn.decomposition import PCA
import numpy as np
from ds_manager import DSManager
from sklearn.linear_model import LinearRegression

ds = DSManager(name="demmin")
train_X, train_y, test_X, test_y = ds.get_all_set_X_y()


pca = PCA(n_components=7)
X_pca = pca.fit_transform(train_X)
mlr = LinearRegression()
mlr.fit(X_pca, train_y)
print(mlr.score(X_pca, train_y))

X_pca_test = pca.fit_transform(test_X)
mlr.fit(X_pca_test, test_y)
print(mlr.score(X_pca_test, test_y))

