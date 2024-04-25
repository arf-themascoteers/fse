from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdr","fsdr","fsdr","fsdr","fsdr","fsdr","fsdr","fsdr","fsdr","fsdr"],
        "datasets" : ["lucas_full"],
        "target_sizes" : [10]
    }
    ev = Evaluator(tasks,1,"fsdr.csv")
    ev.evaluate()