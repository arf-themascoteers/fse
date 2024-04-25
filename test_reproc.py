from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdr", "fsdrd"],
        "datasets" : ["lucas_skipped_min"],
        "target_sizes" : [10]
    }
    ev = Evaluator(tasks,3,1,"fsdr.csv")
    ev.evaluate()