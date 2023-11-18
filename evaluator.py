import numpy as np
from sklearn.decomposition import PCA
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
from datetime import datetime


class Evaluator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.filename = os.path.join("results",str(int(time.time())) + ".csv")
        print(self.filename)
        with open(self.filename, 'w') as file:
            file.write("algorithm,dataset,rows,columns,time,target_size,r2_original,rmse_original,r2_reduced,rmse_reduced,selected_features\n")

    def write_row(self,algorithm,dataset,time,target_size,r2_original,rmse_original,r2_reduced,rmse_reduced,features):
        with open(self.filename, 'a') as file:
            file.write(f"{algorithm},{dataset},{dataset.count_rows()},{dataset.count_features()},"
                       f"{time},{target_size},{r2_original},{rmse_original},{r2_reduced},{rmse_reduced}"
                       f"{';'.join(features)}\n")

    def evaluate(self):
        for task in self.tasks:
            print(task)
            dataset = task["dataset"]
            target_feature_size = task["target_feature_size"]
            algorithm = task["algorithm"]
            start_time = datetime.now()
            r2_original = None
            rmse_original = None
            r2_reduced = None
            rmse_reduced = None
            selected_features = None
            if algorithm == "pca":
                r2_original, rmse_original, r2_reduced, rmse_reduced, selected_features = self.do_pca(dataset, target_feature_size)
            elif algorithm == "pls":
                r2_original, rmse_original, r2_reduced, rmse_reduced, selected_features = self.do_pls(dataset, target_feature_size)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.write_row(algorithm, dataset, elapsed_time, target_feature_size,
                           r2_original, rmse_original, r2_reduced, rmse_reduced,selected_features)


    def do_pca(self,dataset, target_feature_size):
        X_train, y_train, X_test, y_test = dataset.get_train_test_X_y()
        r2_original, rmse_original = self.get_metrics(X_train, y_train, X_test, y_test)
        pca = PCA(n_components=target_feature_size)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        r2_reduced, rmse_reduced = self.get_metrics(X_train_pca, y_train, X_test_pca, y_test)
        return  r2_original, rmse_original, r2_reduced, rmse_reduced, []

    def do_pls(self,dataset, target_feature_size):
        X_train, y_train, X_test, y_test = dataset.get_train_test_X_y()
        r2_original, rmse_original = self.get_metrics(X_train, y_train, X_test, y_test)
        pls = PLSRegression(n_components=target_feature_size)
        pls.fit(X_train, y_train)
        X_train_pls = pls.transform(X_train)
        X_test_pls = pls.transform(X_test)
        r2_reduced, rmse_reduced = self.get_metrics(X_train_pls, y_train, X_test_pls, y_test)
        return r2_original, rmse_original, r2_reduced, rmse_reduced, []

    def get_metrics(self, X_train, y_train, X_test, y_test):
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = round(r2_score(y_test, y_pred),3)
        rmse = round(math.sqrt(mean_squared_error(y_test, y_pred)),3)
        print(r2, rmse)
        return r2, rmse