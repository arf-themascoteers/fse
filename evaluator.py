from ds_manager import DSManager
from datetime import datetime
from algorithm_creator import AlgorithmCreator
from sklearn.metrics import r2_score, mean_squared_error
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import my_utils
from reporter import Reporter
from data_splits import DataSplits


class Evaluator:
    def __init__(self, task, repeat=1, folds=1, filename="results.csv"):
        self.task = task
        self.repeat = repeat
        self.folds = folds
        self.reporter = Reporter(filename)

    def evaluate(self):
        for dataset_name in self.task["datasets"]:
            dataset = DSManager(name=dataset_name, folds=self.folds)
            self.evaluate_for_all_features(dataset)
            for target_size in self.task["target_sizes"]:
                for fold, splits in enumerate(dataset.get_k_folds()):
                    for algorithm in self.task["algorithms"]:
                        for repeat_no in range(self.repeat):
                            self.evaluate_for_a_case(dataset_name, target_size, fold, algorithm, repeat_no, splits)

    def evaluate_for_a_case(self, dataset, target_size, fold, algorithm_name, repeat_no, splits):
        final_size, time, metric1, metric2, selected_features = \
            self.reporter.get_saved_metrics(dataset, target_size, fold, algorithm_name, repeat_no)
        if time is not None:
            print(f"{dataset} for size {target_size} for fold {fold} for {algorithm_name} was done")
            return

        algorithm = AlgorithmCreator.create(algorithm_name, target_size,
                                            splits.train_x, splits.train_y, splits.validation_x, splits.validation_y)
        start_time = datetime.now()
        selected_features = algorithm.fit()
        elapsed_time = (datetime.now() - start_time).total_seconds()

        test_for_train_x = algorithm.transform(splits.test_for_train_x)
        test_for_test_x = algorithm.transform(splits.test_for_test_x)
        metric1, metric2 = Evaluator.evaluate_train_test_pair(dataset, test_for_train_x, splits.test_for_train_y, test_for_test_x, splits.test_for_test_y)
        self.reporter.write_details(dataset, target_size, fold, algorithm_name, repeat_no, test_for_test_x.shape[1], elapsed_time, metric1, metric2, selected_features)

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
        algorithm = my_utils.get_metric_evaluator(DSManager.get_task_by_name(dataset_name))
        algorithm.fit(X_train, y_train)
        y_pred = algorithm.predict(X_test)
        return Evaluator.calculate_metrics(dataset_name, y_test, y_pred)

    @staticmethod
    def calculate_metrics(dataset_name, y_test, y_pred):
        if DSManager.get_task(dataset_name) == "classification":
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



