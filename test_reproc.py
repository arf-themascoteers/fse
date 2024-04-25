from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdr","fsdrd"],
        "datasets" : ["lucas_full"],
        "target_sizes" : [10]
    }
    ev = Evaluator(tasks,10, 1,"fsdr.csv")
    ev.evaluate()