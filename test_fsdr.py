from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdrl","bsnet","lasso","pcal"],
        "datasets" : ["lucas_full","lucas_min","lucas_downsampled","lucas_skipped"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,10,"fsdr-bsdr-c3.csv")
    ev.evaluate()