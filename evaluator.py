import numpy as np
from sklearn.decomposition import PCA
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import math


class Evaluator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.filename = os.path.join("data",str(int(time.time())) + ".csv")

    def write_head_row(self):
        with open(self.filename, 'w') as file:
            file.write("algorithm,dataset,rows,columns,time,target_size,r2,rmse,features\n")

    def write_row(self,algorithm,dataset,time,target_size,r2,rmse,features):
        with open(self.filename, 'w') as file:
            file.write(f"{algorithm},{dataset.count_name()},{dataset.count_features()},"
                       f"{time},{target_size},{r2},{rmse},{';'.join(features)}\n")

    def evaluate(self):
        for task in self.tasks:
            dataset = task["dataset"]
            target_feature_size = task["target_feature_size"]
            algorithm = task["algorithm"]
            X_train, y_train, X_test, y_test, X_validation, y_validation = dataset.get_train_test_validation_X_y()
            self.get_metrics(X_train, y_train, X_test, y_test)
            pca = PCA(n_components=target_feature_size)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            print(X_train_pca)
            self.get_metrics(X_train_pca, y_train, X_test_pca, y_test)

    def get_metrics(self, X_train, y_train, X_test, y_test):
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = round(r2_score(y_test, y_pred),3)
        rmse = round(math.sqrt(mean_squared_error(y_test, y_pred)),3)
        print(r2, rmse)
        return r2, rmse