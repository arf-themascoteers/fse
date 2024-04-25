from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["bsnet","fsdr"],
        "datasets" : ["lucas_skipped", "lucas_full", "lucas_min"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,5,"fsdr_vs_bsnet.csv")
    ev.evaluate()