from sklearn.linear_model import LinearRegression
from ds_manager import DSManager


d = DSManager(dataset="brazilian")
mlr = LinearRegression()
trainx, trainy, testx, testy = d.get_train_test_X_y()
trainx = trainx[:,[10,100,400,800,1500,2000]]
testx = testx[:,[10,100,400,800,1500,2000]]
mlr.fit(trainx, trainy)
print(mlr.score(testx, testy))