from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["mcuve", "cars", "lasso", "rf"],
        "datasets" : ["brazilian", "lucas_skipped", "lucas_full"],
        "target_feature_sizes" : [5, 10, 15, 20, 25, 30]
    }
    tasks = {
        "algorithms" : ["lasso", "mi"],
        "datasets" : ["brazilian", "lucas_skipped_min"],
        "target_feature_sizes" : [2, 5]
    }
    ev = Evaluator(tasks,5,"quick.csv")
    ev.evaluate()