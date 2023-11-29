from sklearn.neural_network import MLPRegressor
from ds_manager import DSManager


d = DSManager(reduced_features=True, reduced_rows=True)
mlp = MLPRegressor()
trainx, trainy, testx, testy = d.get_train_test_X_y()
mlp.fit(trainx, trainy)
print(mlp.score(testx, testy))

trainx = trainx[:,[40,50]]
testx = testx[:,[40,50]]
mlp = MLPRegressor(hidden_layer_sizes=(100,20))
mlp.fit(trainx, trainy)
print(mlp.score(testx, testy))