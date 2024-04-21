from ds_manager import DSManager
from datetime import datetime
import os
from algorithm_creator import AlgorithmCreator
from sklearn.metrics import r2_score, mean_squared_error
import math
import pandas as pd
import my_utils


class Evaluator:
    def __init__(self, task, folds=1, filename="results.csv"):
        self.task = task
        self.folds = folds
        self.summary_filename = filename
        self.details_filename = f"details_{self.summary_filename}.csv"
        self.all_features_details_filename = f"all_features_details_{self.summary_filename}.csv"
        self.all_features_summary_filename = f"all_features_summary_{self.summary_filename}.csv"

        self.summary_file = os.path.join("results", self.summary_filename)
        self.details_file = os.path.join("results", self.details_filename)
        self.all_features_details_file = os.path.join("results", self.all_features_details_filename)
        self.all_features_summary_file = os.path.join("results", self.all_features_summary_filename)

        if not os.path.exists(self.summary_file):
            with open(self.summary_file, 'w') as file:
                file.write("algorithm,dataset,time,target_size,final_size,"
                           "r2_train,r2_test,"
                           "rmse_train,rmse_test,"
                           "selected_features\n")

        if not os.path.exists(self.details_file):
            with open(self.summary_file, 'w') as file:
                file.write("fold,algorithm,dataset,time,target_size,final_size,"
                           "r2_train,r2_test,"
                           "rmse_train,rmse_test,"
                           "selected_features\n")

        if not os.path.exists(self.details_file):
            with open(self.summary_file, 'w') as file:
                file.write("fold,algorithm,dataset,time,target_size,final_size,"
                           "r2_train,r2_test,"
                           "rmse_train,rmse_test,"
                           "selected_features\n")

    def evaluate(self):
        for dataset in self.task.datasets:
            for target_feature_size in self.task.target_feature_sizes:
                self.evaluate_dataset_target_feature_size(dataset, target_feature_size)

    def evaluate_dataset_target_feature_size(self, dataset_name, target_feature_size):
        dataset = DSManager(name=dataset_name, folds=self.folds)
        Evaluator.get_metrics(dataset)
        self.do_algorithm(algorithm_name, dataset, target_feature_size)

    def is_done(self,algorithm_name,dataset,target_feature_size):
        df = pd.read_csv(self.summary_file)
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

            with open(self.summary_file, 'a') as file:
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

    def get_metrics(self, dataset):
        metric_evaluator = my_utils.get_metric_evaluator(dataset)

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
