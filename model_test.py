from sklearn.neural_network import MLPRegressor
from ds_manager import DSManager

ds = DSManager()
train_X, train_y, test_X, test_y = ds.get_train_test_X_y()

mlp = MLPRegressor(hidden_layer_sizes=(30), max_iter=300, random_state=42, learning_rate_init=0.001)
mlp.fit(train_X, train_y)
print(mlp.score(test_X, test_y))

s = [64, 6, 16, 25, 5, 4, 57, 60]

train_X = train_X[:,s]
test_X = test_X[:,s]

mlp = MLPRegressor(hidden_layer_sizes=(30), max_iter=300, random_state=42, learning_rate_init=0.001)
mlp.fit(train_X, train_y)
print(mlp.score(test_X, test_y))