from sklearn.linear_model import LinearRegression
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC


def get_metric_evaluator(task):
    if task == "regression":
        return SVR(kernel='rbf', C=1e3, gamma=0.1)
    return SVC(kernel='rbf', C=1.0, gamma='auto')


def get_internal_model():
    return LinearRegression()
