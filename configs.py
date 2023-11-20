from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from model_ann import ModelANN


def get_hidden(feature_size):
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


def get_mlp_model(feature_size):
    return ModelANN(feature_size)
    hidden = get_hidden(feature_size)
    print("Configs",hidden)
    return MLPRegressor(hidden_layer_sizes=hidden,
                        max_iter=1500, random_state=10,
                        learning_rate_init=0.001)


def get_metric_evaluator_for_traditional(feature_size):
    return get_mlp_model(feature_size)


def get_metric_evaluator_for_fscr(feature_size):
    return get_mlp_model(feature_size)


def get_metric_evaluator_for(algorithm_name,feature_size):
    if algorithm_name == "fscr":
        return get_metric_evaluator_for_fscr(feature_size)
    return get_metric_evaluator_for_traditional(feature_size)


def get_internal_model():
    return LinearRegression()
