from ds_manager import DSManager
from datetime import datetime
import os
from algorithm_creator import AlgorithmCreator
from sklearn.metrics import r2_score, mean_squared_error
import math
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
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
        self.all_features_summary_file = os.path.join("results", self.all_features_summary_filename)
        self.all_features_details_file = os.path.join("results", self.all_features_details_filename)

        if not os.path.exists(self.summary_file):
            with open(self.summary_file, 'w') as file:
                file.write("algorithm,dataset,target_size,final_size,time,metric1,metric2,selected_features\n")

        if not os.path.exists(self.details_file):
            with open(self.details_file, 'w') as file:
                file.write("fold,algorithm,dataset,target_size,final_size,time,metric1,metric2,selected_features\n")

        if not os.path.exists(self.all_features_summary_file):
            with open(self.all_features_summary_file, 'w') as file:
                file.write("dataset,metric1,metric2\n")

        if not os.path.exists(self.all_features_details_file):
            with open(self.all_features_details_file, 'w') as file:
                file.write("fold,dataset,metric1,metric2\n")

    def evaluate(self):
        for dataset_name in self.task.datasets:
            dataset = DSManager(name=dataset_name, folds=self.folds)
            self.evaluate_for_all_features(dataset)
            for target_feature_size in self.task.target_feature_sizes:
                for fold_number, (X_train, y_train, X_test_for_train, y_test_for_train, X_test_for_test, y_test_for_test) in enumerate(dataset.get_k_folds()):
                    for algorithm in self.task.algorithms:
                        self.evaluate_for_dataset_target_fold_algorithm(
                            dataset_name, target_feature_size, fold_number, algorithm, X_train, y_train, X_test_for_train, X_test_for_test, y_test_for_train, y_test_for_test)

    def evaluate_for_dataset_target_fold_algorithm(self, dataset, target_size, fold, algorithm_name, X_train, y_train, X_test_for_train, X_test_for_test, y_test_for_train, y_test_for_test):
        time, target_size, final_size, metric1_train,metric1_test,metric2_train,metric2_test,selected_features = (
            self.get_saved_metrics_dataset_target_fold_algorithm(dataset, target_size, fold, algorithm_name))
        if time is not None:
            print(f"Fold {fold} for {dataset} for {algorithm_name} for size {target_size} was done")
            return

        algorithm = AlgorithmCreator.create(algorithm_name, X_train, y_train, target_size)
        start_time = datetime.now()
        selected_features = algorithm.fit()
        elapsed_time = (datetime.now() - start_time).total_seconds()

        X_test_for_train = algorithm.transform(X_test_for_train)
        X_test_for_test = algorithm.transform(X_test_for_test)
        algorithm.fit(X_test_for_train, y_test_for_train)
        y_pred = algorithm.predict(X_test_for_test)
        metric1_train, metric2_train = Evaluator.calculate_metrics(dataset, y_test_for_test, y_pred)


        with open(self.summary_file, 'a') as file:
            file.write(
                f"{algorithm_name},"
                f"{dataset.count_rows()},"
                f"{dataset.count_features()},"
                f"{round(elapsed_time, 2)},"
                f"{len(selected_features)},"
                f"{selected_features},"
                f"{metric1_reduced_train},"
                f"{metric1_reduced_test},"
                f"{metric2_reduced_train},"
                f"{metric2_reduced_test},"
                f"{';'.join(str(i) for i in selected_features)}\n")

    def get_saved_metrics_dataset_target_fold_algorithm(self, dataset, target_size, fold, algorithm):
        df = pd.read_csv(self.details_file)
        if len(df) == 0:
            return None, None
        rows = df.loc[
            (df["dataset"] == dataset) &
            (df["target_size"] == target_size) &
            (df["fold"] == fold) &
            (df["algorithm"] == algorithm)
            ]
        if len(rows) == 0:
            return None, None
        row = rows.iloc[0]
        return row["time"], row["final_size"], row["metric1_train"], row["metric1_test"], row["metric2_train"], row["metric2_test"], row["selected_features"]

    def evaluate_for_all_features(self, dataset):
        for fold_number, (X_train, y_train, X_test_for_train, y_test_for_train, X_test_for_test, y_test_for_test) in enumerate(dataset.get_k_folds()):
            self.evaluate_for_all_features_fold(fold_number, dataset.name, X_test_for_train, y_test_for_train, X_test_for_test, y_test_for_test)
        self.evaluate_for_all_features_summary(dataset.name)

    def evaluate_for_all_features_summary(self, dataset):
        df = pd.read_csv(self.all_features_details_file)
        df = df[df["dataset"] == dataset]
        metric1 = round(df["metric1"].mean(),2)
        metric2 = round(df["metric2"].mean(),2)

        df2 = pd.read_csv(self.all_features_summary_file)
        df2.loc[df['dataset'] == dataset, 'metric1'] = metric1
        df2.loc[df['dataset'] == dataset, 'metric2'] = metric2

    def evaluate_for_all_features_fold(self, fold, dataset_name, X_train, y_train, X_test, y_test):
        metric1, metric2 = self.get_saved_metrics_for_all_feature_set_fold(fold, dataset_name)
        if metric1 is not None and metric2 is not None:
            print(f"Fold {fold} for {dataset_name} was done")
            return
        algorithm = my_utils.get_metric_evaluator(dataset_name)
        algorithm.fit(X_train, y_train)
        y_pred = algorithm.predict(X_test)
        metric1, metric2 = Evaluator.calculate_metrics(dataset_name, y_test, y_pred)
        with open(self.all_features_details_file, 'a') as file:
            file.write(f"{fold},{dataset_name},{metric1},{metric2}\n")

    def get_saved_metrics_for_all_feature_set_fold(self, fold, dataset):
        df = pd.read_csv(self.all_features_details_file)
        if len(df) == 0:
            return None, None
        rows = df.loc[(df['fold'] == fold) & (df['dataset'] == dataset)]
        if len(rows) == 0:
            return None, None
        row = rows.iloc[0]
        return row["metric1"], row["metric2"]

    @staticmethod
    def calculate_metrics(dataset_name, y_test, y_pred):
        if my_utils.get_task(dataset_name) == "classification":
            return Evaluator.calculate_metrics_for_classification(y_test, y_pred)
        return Evaluator.calculate_metrics_for_regression(y_test, y_pred)

    @staticmethod
    def calculate_metrics_for_classification(y_test, y_pred):
        accuracy = round(accuracy_score(y_test, y_pred),2)
        kappa = round(cohen_kappa_score(y_test, y_pred), 2)
        return accuracy, kappa

    @staticmethod
    def calculate_metrics_for_regression(y_test, y_pred):
        r2 = round(r2_score(y_test, y_pred), 2)
        rmse = round(math.sqrt(mean_squared_error(y_test, y_pred)), 2)
        return r2, rmse


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

