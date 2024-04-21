from ds_manager import DSManager
from datetime import datetime
import os
from algorithm_creator import AlgorithmCreator
from sklearn.metrics import r2_score, mean_squared_error
import math
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
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
                for fold, (train_x, train_y, validation_x, validation_y, test_for_train_x, test_for_train_y, test_for_test_x, test_for_test_y) in enumerate(dataset.get_k_folds()):
                    for algorithm in self.task.algorithms:
                        self.evaluate_for_dataset_target_fold_algorithm(
                            dataset_name, target_feature_size, fold, algorithm,
                            train_x, train_y,
                            validation_x, validation_y,
                            test_for_train_x, test_for_train_y,
                            test_for_test_x, test_for_test_y
                        )

    def evaluate_for_dataset_target_fold_algorithm(self,
                                                   dataset, target_size, fold, algorithm_name,
                                                   train_x, train_y,
                                                   validation_x, validation_y,
                                                   test_for_train_x, test_for_train_y,
                                                   test_for_test_x, test_for_test_y
                                                   ):
        final_size, time, metric1, metric2, selected_features = self.get_saved_metrics_dataset_target_fold_algorithm(dataset, target_size, fold, algorithm_name)
        if time is not None:
            print(f"Fold {fold} for {dataset} for {algorithm_name} for size {target_size} was done")
            return

        algorithm = AlgorithmCreator.create(algorithm_name, target_size, train_x, train_y, validation_x, validation_y)
        start_time = datetime.now()
        selected_features = algorithm.fit()
        elapsed_time = (datetime.now() - start_time).total_seconds()

        X_test_for_train = algorithm.transform(test_for_train_x)
        X_test_for_test = algorithm.transform(test_for_test_x)
        algorithm.fit(X_test_for_train, test_for_train_y)
        y_pred = algorithm.predict(X_test_for_test)
        metric1, metric2 = Evaluator.calculate_metrics(dataset, test_for_test_y, y_pred)

        with open(self.all_features_details_file, 'a') as file:
            file.write(f"{fold},{algorithm_name},{dataset},{target_size},"
                       f"{X_test_for_test.shape[1]},{elapsed_time},{metric1},{metric2},{selected_features}\n")

        self.update_summary_for_dataset_target_fold_algorithm(dataset, target_size, algorithm_name)

    def update_summary_for_dataset_target_fold_algorithm(self, dataset, target_size, algorithm_name):
        df = pd.read_csv(self.details_file)
        df = df[
            (df["dataset"] == dataset) &
            (df["target_size"] == target_size) &
            (df["algorithm"] == algorithm_name)
            ]
        if len(df) == 0:
            return

        final_size = round(df["final_size"].mean(), 2)
        time = round(df["time"].mean(), 2)
        metric1 = round(df["metric1"].mean(), 2)
        metric2 = round(df["metric2"].mean(), 2)
        selected_features = '||'.join(df['selected_features'])

        df2 = pd.read_csv(self.summary_file)
        mask = ((df2["dataset"] == dataset) & (df2["target_size"] == target_size) & (df2["algorithm"] == algorithm_name))
        if len(df2[mask]) == 0:
            df2.loc[len(df2)] = {
                "dataset":dataset, "target_size":target_size, "algorithm": algorithm_name,
                "final_size":final_size,"time":time,"metric1":metric1,"metric2":metric2
            }
        else:
            df2.loc[mask, 'final_size'] = final_size
            df2.loc[mask, 'time'] = time
            df2.loc[mask, 'metric1'] = metric1
            df2.loc[mask, 'metric2'] = metric2
            df2.loc[mask, 'selected_features'] = selected_features
        df2.to_csv(self.all_features_summary_file, index=False)

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
        return row["final_size"], row["time"], row["metric1"], row["metric2"], row["selected_features"]

    def evaluate_for_all_features(self, dataset):
        for fold, (_, _, _, _, test_for_train_x, test_for_train_y, test_for_test_x, test_for_test_y) in enumerate(dataset.get_k_folds()):
            self.evaluate_for_all_features_fold(fold, dataset.name, test_for_train_x, test_for_train_y, test_for_test_x, test_for_test_y)
        self.update_for_all_features_summary(dataset.name)

    def update_for_all_features_summary(self, dataset):
        df = pd.read_csv(self.all_features_details_file)
        df = df[df["dataset"] == dataset]
        if len(df) == 0:
            return

        metric1 = round(df["metric1"].mean(),2)
        metric2 = round(df["metric2"].mean(),2)

        df2 = pd.read_csv(self.all_features_summary_file)
        mask = (df2['dataset'] == dataset)
        if len(df[mask]) == 0:
            df2.loc[len(df2)] = {"dataset":dataset, "metric1":metric1, "metric2": metric2}
        else:
            df2.loc[mask, 'metric1'] = metric1
            df2.loc[mask, 'metric2'] = metric2
        df2.to_csv(self.all_features_summary_file, index=False)

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
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        return accuracy, kappa

    @staticmethod
    def calculate_metrics_for_regression(y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        return r2, rmse



