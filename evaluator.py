import numpy as np
from sklearn.decomposition import PCA
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
import math
from datetime import datetime


class Evaluator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.filename = os.path.join("results",str(int(time.time())) + ".csv")
        print(self.filename)
        with open(self.filename, 'w') as file:
            file.write("algorithm,dataset,rows,columns,time,target_size,"
                       "r2_original,rmse_original,"
                       "r2_reduced_train,rmse_reduced_train,"
                       "r2_reduced_test,rmse_reduced_test,"
                       "selected_features\n")

    def evaluate(self):
        for task in self.tasks:
            print(task)
            dataset = task["dataset"]
            target_feature_size = task["target_feature_size"]
            algorithm = task["algorithm"]
            start_time = datetime.now()
            r2_original, rmse_original, \
                r2_reduced_train, rmse_reduced_train, \
                r2_reduced_test, rmse_reduced_test, \
                selected_features = \
                self.do_algorithm(algorithm, dataset, target_feature_size)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            with open(self.filename, 'a') as file:
                file.write(
                    f"{algorithm},{dataset},{dataset.count_rows()},{dataset.count_features()},{elapsed_time},{target_feature_size},"
                    f"{r2_original},{rmse_original},"
                    f"{r2_reduced_train},{rmse_reduced_train},"
                    f"{r2_reduced_test},{rmse_reduced_test},"
                    f"{';'.join(selected_features)}\n")

    def do_algorithm(self,algorithm, dataset, target_feature_size):
        X_train, y_train, X_test, y_test = dataset.get_train_test_X_y()
        _, _, r2_original, rmse_original = self.get_metrics(X_train, y_train, X_test, y_test)
        model = None
        selected_features = None
        if algorithm == "pca":
            model,selected_features = self.do_pca(X_train, y_train, target_feature_size)
        elif algorithm == "pls":
            model,selected_features = self.do_pls(X_train, y_train, target_feature_size)
        elif algorithm == "rfe":
            model,selected_features = self.do_rfe(X_train, y_train, target_feature_size)
        X_train_reduced = model.transform(X_train)
        X_test_reduced = model.transform(X_test)
        r2_reduced_train, rmse_reduced_train, r2_reduced_test, rmse_reduced_test = \
            self.get_metrics(X_train_reduced, y_train, X_test_reduced, y_test)
        return r2_original, rmse_original, \
            r2_reduced_train, rmse_reduced_train, \
            r2_reduced_test, rmse_reduced_test, selected_features

    def do_pca(self,X_train, y_train, target_feature_size):
        pca = PCA(n_components=target_feature_size)
        pca.fit(X_train)
        return pca,[]

    def do_pls(self,X_train, y_train, target_feature_size):
        pls = PLSRegression(n_components=target_feature_size)
        pls.fit(X_train, y_train)
        return pls,[]

    def do_rfe(self,X_train, y_train, target_feature_size):
        internal_model = RandomForestRegressor()
        rfe = RFE(internal_model, n_features_to_select=target_feature_size)
        rfe.fit(X_train, y_train)
        return rfe, X_train.columns[rfe.support_]

    def get_metrics(self, X_train, y_train, X_test, y_test):
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        r2_train = round(r2_score(y_train, y_pred),3)
        rmse_train = round(math.sqrt(mean_squared_error(y_train, y_pred)),3)
        
        y_pred = model.predict(X_test)
        r2_test = round(r2_score(y_test, y_pred),3)
        rmse_test = round(math.sqrt(mean_squared_error(y_test, y_pred)),3)

        return r2_train, rmse_train, r2_test, rmse_test
