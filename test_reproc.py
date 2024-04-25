from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdr","fsdrd"],
        "datasets" : ["lucas_full"],
        "target_sizes" : [10]
    }
    ev = Evaluator(tasks,1, 10,"fsdr.csv")
    ev.evaluate()