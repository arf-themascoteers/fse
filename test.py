from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["mcuve", "cars", "lasso", "rf"],
        "datasets" : ["brazilian", "lucas_skipped", "lucas_full"],
        "target_feature_size" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,10,"quick.csv")
    ev.evaluate()