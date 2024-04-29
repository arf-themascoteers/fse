from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdr","bsnet","spa","mcuve","pcal","lasso"],
        "datasets" : ["lucas_full", "indian_pines", "ghsi", "lucas_skipped", "lucas_downscaled"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,10,"fsdr.csv")
    ev.evaluate()