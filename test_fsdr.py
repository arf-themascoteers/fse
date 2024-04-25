from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdr"],
        "datasets" : ["lucas_skipped", "lucas_full"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,1,"fsdr.csv")
    ev.evaluate()