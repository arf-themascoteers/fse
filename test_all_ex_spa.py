from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdrl","bsnet","mcuve","pcal","lasso"],
        "datasets" : ["lucas_full", "indian_pines", "ghsi", "lucas_skipped", "lucas_downscaled", "lucas_min"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,10,"fsdr.csv")
    ev.evaluate()