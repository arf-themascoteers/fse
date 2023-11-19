from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def get_mlp_model():
    return MLPRegressor(hidden_layer_sizes=(30,), max_iter=600, random_state=42, learning_rate_init=0.001)

def get_metric_evaluator_for_traditional():
    return get_mlp_model()

def get_metric_evaluator_for_fscr():
    return get_mlp_model()

def get_internal_model():
    return LinearRegression()
