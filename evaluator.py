from ds_manager import DSManager
from datetime import datetime
import os
from algorithm_creator import AlgorithmCreator
from sklearn.metrics import r2_score, mean_squared_error
import math
import pandas as pd
import my_utils


class Evaluator:
    def __init__(self, tasks, folds=1, filename="results.csv"):
        self.folds = folds
        self.tasks = tasks
        self.summary_filename = filename
        self.details_filename = f"details_{filename}.csv"
        self.summary_filename = os.path.join("results", self.summary_filename)
        if not os.path.exists(self.summary_filename):
            with open(self.summary_filename, 'w') as file:
                file.write("algorithm,rows,columns,time,target_size,final_size,"
                           "r2_train,r2_test,"
                           "rmse_train,rmse_test,"
                           "selected_features\n")

    def evaluate(self):
        for task in self.tasks:
            print(task)
            dataset = task["dataset"]
            target_feature_size = task["target_feature_size"]
            algorithm_name = task["algorithm"]
            dataset = DSManager(dataset=dataset, folds=self.folds)
            self.do_algorithm(algorithm_name, dataset, target_feature_size)

    def is_done(self,algorithm_name,dataset,target_feature_size):
        df = pd.read_csv(self.summary_filename)
        if len(df) == 0:
            return False
        rows = df.loc[
            (df['algorithm'] == algorithm_name) &
            (df['rows'] == dataset.count_rows()) &
            (df['columns'] == dataset.count_features()) &
            (df['target_size'] == target_feature_size)
        ]
        return len(rows) != 0

    def do_algorithm(self, algorithm_name, dataset, target_feature_size):
        for fold_number, (X_train, y_train, X_test, y_test) in enumerate(dataset.get_k_folds()):
            algorithm = AlgorithmCreator.create(algorithm_name, X_train, y_train, target_feature_size)
            start_time = datetime.now()
            selected_features = algorithm.fit()
            elapsed_time = (datetime.now() - start_time).total_seconds()
            X_train_reduced = algorithm.transform(X_train)
            X_test_reduced = algorithm.transform(X_test)
            r2_reduced_train, rmse_reduced_train, r2_reduced_test, rmse_reduced_test = \
                Evaluator.get_metrics(algorithm_name, X_train_reduced, y_train, X_test_reduced, y_test)

            with open(self.summary_filename, 'a') as file:
                file.write(
                    f"{algorithm_name},"
                    f"{dataset.count_rows()},"
                    f"{dataset.count_features()},"
                    f"{round(elapsed_time,2)},"
                    f"{len(selected_features)},"
                    f"{selected_features},"
                    f"{r2_reduced_train},"
                    f"{r2_reduced_test},"
                    f"{rmse_reduced_train},"
                    f"{rmse_reduced_test},"
                    f"{';'.join(str(i) for i in selected_features)}\n")


    @staticmethod
    def get_metrics(algorithm_name, X_train, y_train, X_test, y_test):
        metric_evaluator = my_utils.get_metric_evaluator(algorithm_name, X_train)
        metric_evaluator.fit(X_train, y_train)

        y_pred = metric_evaluator.predict(X_train)
        r2_train = round(r2_score(y_train, y_pred), 2)
        rmse_train = round(math.sqrt(mean_squared_error(y_train, y_pred)), 2)

        y_pred = metric_evaluator.predict(X_test)
        r2_test = round(r2_score(y_test, y_pred), 2)
        rmse_test = round(math.sqrt(mean_squared_error(y_test, y_pred)), 2)

        print(f"r2 train {r2_train}")
        print(f"r2 test {r2_test}")

        return r2_train, rmse_train, r2_test, rmse_test
