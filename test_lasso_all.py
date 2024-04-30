from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["lasso"],
        "datasets" : ["lucas_full", "lucas_skipped", "lucas_downsampled", "lucas_min", "indian_pines", "ghsi"],
        "target_sizes" : [5,10,15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,10,"lasso.csv")
    ev.evaluate()