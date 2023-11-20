from sklearn.linear_model import LinearRegression
from ds_manager import DSManager


d = DSManager()
mlr = LinearRegression()
trainx, trainy, testx, testy = d.get_train_test_X_y()
mlr.fit(trainx, trainy)
print(mlr.score(testx, testy))