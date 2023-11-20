from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from model_ann import ModelANN
import torch.nn as nn
import torch


def get_hidden_for_full(feature_size):
    h1 = 50
    h2 = 10
    if feature_size >=50 and feature_size < 100:
        h1 = 40
    elif feature_size >=100 and feature_size <= 200:
        h1 = 30
    elif feature_size >200 and feature_size < 1000:
        h1 = 10
    elif feature_size >= 1000 and feature_size < 2000:
        h1 = 8
    elif feature_size >= 2000 and feature_size < 3000:
        h1 = 4
    elif feature_size >= 3000:
        h1 = 2
        h2 = 1
    return h1, h2


def get_hidden_for_short(feature_size):
    h1 = 15
    h2 = 10
    if feature_size >=50 and feature_size < 100:
        h1 = 11
    elif feature_size >=100 and feature_size <= 200:
        h1 = 6
        h2 = 5
    elif feature_size > 200 and feature_size <= 250:
        h1 = 5
        h2 = 4
    elif feature_size > 300:
        h1 = 0
        h2 = 0
    return h1, h2

def get_hidden(rows, feature_size):
    if rows > 3000:
        return get_hidden_for_full(feature_size)
    return get_hidden_for_short(feature_size)

def get_ann(X):
    return ModelANN(X)


def get_metric_evaluator_for_traditional(X):
    return get_ann(X)


def get_metric_evaluator_for_fscr(X):
    return get_ann(X)


def get_metric_evaluator_for(algorithm_name,X):
    if algorithm_name == "fscr":
        return get_metric_evaluator_for_fscr(X)
    return get_metric_evaluator_for_traditional(X)


def get_internal_model():
    return LinearRegression()


def get_linear(rows, features):
    h1, h2 = get_hidden(rows, features)
    if h1 == 0 or h2 == 0:
        return nn.Sequential(
            nn.Linear(features, h1),
        )

    return nn.Sequential(
        nn.Linear(features, h1),
        nn.LeakyReLU(),
        nn.Linear(h1, h2),
        nn.LeakyReLU(),
        nn.Linear(h2, 1)
    )
