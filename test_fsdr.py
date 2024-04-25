from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdr"],
        "datasets" : ["ghsi"],
        "target_sizes" : [10]
    }
    ev = Evaluator(tasks,1, 1,"fsdr.csv")
    ev.evaluate()