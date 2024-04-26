from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["mcuve", "cars", "lasso", "rf"],
        "datasets" : ["brazilian", "lucas_skipped", "lucas_full"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    tasks = {
        "algorithms" : ["lasso","bsnet","fsdr"],
        "datasets" : ["lucas_downsampled_min", "lucas_skipped_min"],
        "target_sizes" : [2, 5, 7]
    }
    ev = Evaluator(tasks,3,4,"quick.csv")
    ev.evaluate()