from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : [],
        "datasets" : ["indian_pines"],
        "target_sizes" : [10]
    }
    ev = Evaluator(tasks,1, 10,"fsdr.csv")
    ev.evaluate()