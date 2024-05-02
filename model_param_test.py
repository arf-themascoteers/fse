from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdrl","bsnet"],
        "datasets" : ["lucas_full","lucas_downsampled"],
        "target_sizes" : [5]
    }
    ev = Evaluator(tasks,1,1,"dummy.csv")
    ev.evaluate()