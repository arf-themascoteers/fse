from sklearn.neural_network import MLPRegressor
from ds_manager import DSManager


def get_mlp():
    return MLPRegressor(hidden_layer_sizes=(), max_iter=300, random_state=42, learning_rate_init=0.01)


ds = DSManager(reduced_features=False,reduced_rows=False)
train_X, train_y, test_X, test_y = ds.get_train_test_X_y()

mlp = get_mlp()
mlp.fit(train_X, train_y)
print(mlp.score(test_X, test_y))

# s = [64, 6, 16, 25, 5, 4, 57, 60]
#
# train_X = train_X[:,s]
# test_X = test_X[:,s]
#
# mlp = get_mlp()
# mlp.fit(train_X, train_y)
# print(mlp.score(test_X, test_y))