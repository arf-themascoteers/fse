from ds_manager import DSManager
from datetime import datetime
from algorithm_creator import AlgorithmCreator
from sklearn.metrics import r2_score, mean_squared_error
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from reporter import Reporter
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import pandas as pd


class Evaluator:
    def __init__(self, task, repeat=1, folds=1, filename="results.csv"):
        self.task = task
        self.repeat = repeat
        self.folds = folds
        self.reporter = Reporter(filename)
        self.cache = pd.DataFrame(columns=["dataset","fold","algorithm","repeat","final_size","metric1","metric2","time","selected_features"])

    def evaluate(self):
        for dataset_name in self.task["datasets"]:
            dataset = DSManager(name=dataset_name, folds=self.folds)
            self.evaluate_for_all_features(dataset)
            for target_size in self.task["target_sizes"]:
                for fold, splits in enumerate(dataset.get_k_folds()):
                    for algorithm in self.task["algorithms"]:
                        for repeat_no in range(self.repeat):
                            self.process_a_case(target_size, fold, algorithm, repeat_no, splits)

    def process_a_case(self, target_size, fold, algorithm_name, repeat_no, splits):
        final_size, time, metric1, metric2, selected_features = \
            self.reporter.get_saved_metrics(splits.get_name(), target_size, fold, algorithm_name, repeat_no)
        if time is not None:
            print(f"{splits.get_name()} for size {target_size} for fold {fold} for {algorithm_name} was done")
            return

        algorithm = AlgorithmCreator.create(algorithm_name, target_size, splits)

        final_size, elapsed_time, metric1, metric2, selected_features = \
            self.get_results_for_a_case(dataset, target_size, fold, algorithm, repeat_no, splits)

        self.reporter.write_details(dataset, target_size, fold, algorithm_name, repeat_no, final_size, elapsed_time, metric1, metric2, selected_features)

    def get_results_for_a_case(self, dataset, target_size, fold, algorithm, repeat_no, splits):
        final_size, elapsed_time, metric1, metric2, selected_features = self.get_from_cache(dataset, fold, algorithm, repeat_no)
        if elapsed_time is not None:
            print(f"{dataset} for size {target_size} for fold {fold} for {algorithm.get_name()} is got from cache")
            return final_size, elapsed_time, metric1, metric2, selected_features
        return self.compute_case(dataset, target_size, fold, algorithm, repeat_no, splits)

    def compute_case(self, dataset, target_size, fold, algorithm, repeat_no, splits):
        start_time = datetime.now()
        selected_features = algorithm.fit()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        test_for_train_x = algorithm.transform(splits.test_for_train_x)
        test_for_test_x = algorithm.transform(splits.test_for_test_x)
        metric1, metric2 = Evaluator.evaluate_train_test_pair(dataset, test_for_train_x, splits.test_for_train_y, test_for_test_x, splits.test_for_test_y)
        return test_for_test_x.shape[1], elapsed_time, metric1, metric2, selected_features

    def get_from_cache(self, dataset, fold, algorithm, repeat_no):
        if not algorithm.is_independent_of_target_size():
            return None, None, None, None, None
        if len(self.cache) == 0:
            return None, None, None, None, None
        rows = self.cache.loc[
            (self.cache["dataset"] == dataset) &
            (self.cache["fold"] == fold) &
            (self.cache["algorithm"] == algorithm.get_name()) &
            (self.cache["repeat"] == repeat_no)
        ]
        if len(rows) == 0:
            return None, None, None, None, None
        row = rows.iloc[0]
        return row["final_size"], row["time"], row["metric1"], row["metric2"], row["selected_features"]

    def evaluate_for_all_features(self, dataset):
        for fold, splits in enumerate(dataset.get_k_folds()):
            self.evaluate_for_all_features_fold(fold, dataset.name,
                splits.test_for_train_x, splits.test_for_train_y, splits.test_for_test_x, splits.test_for_test_y)

    def evaluate_for_all_features_fold(self, fold, dataset_name, X_train, y_train, X_test, y_test):
        metric1, metric2 = self.reporter.get_saved_metrics_for_all_feature(fold, dataset_name)
        if metric1 is not None and metric2 is not None:
            print(f"Fold {fold} for {dataset_name} was done")
            return
        metric1, metric2 = Evaluator.evaluate_train_test_pair(dataset_name, X_train, y_train, X_test, y_test)
        self.reporter.write_details_all_features(fold, dataset_name, metric1, metric2)

    @staticmethod
    def evaluate_train_test_pair(dataset_name, X_train, y_train, X_test, y_test):
        algorithm = Evaluator.get_metric_evaluator(DSManager.get_task_by_name(dataset_name))
        algorithm.fit(X_train, y_train)
        y_pred = algorithm.predict(X_test)
        return Evaluator.calculate_metrics(dataset_name, y_test, y_pred)

    @staticmethod
    def calculate_metrics(dataset_name, y_test, y_pred):
        if DSManager.get_task_by_name(dataset_name) == "classification":
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

    @staticmethod
    def get_metric_evaluator(task):
        gowith = "sv"

        if gowith == "rf":
            if task == "regression":
                return RandomForestRegressor()
            return RandomForestClassifier()
        elif gowith == "sv":
            if task == "regression":
                return SVR(C=1e5, kernel='rbf', gamma=1.)
            return SVC(C=1e5, kernel='rbf', gamma=1.)
        else:
            if task == "regression":
                return MLPRegressor(max_iter=2000)
            return MLPClassifier(max_iter=2000)

