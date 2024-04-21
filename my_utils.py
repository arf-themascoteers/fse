from sklearn.linear_model import LinearRegression
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC


def get_metric_evaluator(dataset):
    task = get_task(dataset)
    if task == "regression":
        return SVR(kernel='rbf', C=1e3, gamma=0.1)
    return SVC(kernel='rbf', C=1.0, gamma='auto')


def get_internal_model():
    return LinearRegression()


def get_task(dataset):
    if dataset in ["ghsi","indian_pines"]:
        return "classification"
    return "regression"


def get_datasets():
    return [
        "demmin",
        "brazilian",
        "lucas_min",
        "lucas_skipped",
        "lucas_downsampled",
        "lucas_full",
        "ghsi",
        "indian_pines"
    ]


def is_dataset_low_sampled(dataset):
    if dataset in ["demmin", "brazilian", "lucas_min"]:
        return True