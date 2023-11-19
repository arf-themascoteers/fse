from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def get_mlp_model(feature_size):
    hidden_layer = (30,)
    if feature_size > 100 and feature_size <= 200:
        hidden_layer = (20)
    elif feature_size > 200:
        hidden_layer = ()
    print(feature_size, hidden_layer)
    return MLPRegressor(hidden_layer_sizes=hidden_layer, max_iter=600, random_state=42, learning_rate_init=0.001)


def get_metric_evaluator_for_traditional(feature_size):
    return get_mlp_model(feature_size)


def get_metric_evaluator_for_fscr(feature_size):
    return get_mlp_model(feature_size)


def get_internal_model():
    return LinearRegression()
