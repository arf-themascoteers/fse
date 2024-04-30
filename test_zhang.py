from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["zhang"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [30]
    }
    ev = Evaluator(tasks,1,1,"zhang.csv")
    ev.evaluate()