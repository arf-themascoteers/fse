from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["bsnet"],
        "datasets" : ["brazilian", "lucas_skipped", "lucas_full"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    tasks = {
        "algorithms" : ["bsnet"],
        "datasets" : ["lucas_skipped_min"],
        "target_sizes" : [5]
    }
    ev = Evaluator(tasks,1,1,"bsnet.csv")
    ev.evaluate()