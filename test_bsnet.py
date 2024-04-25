from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["bsnet"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,3,"bsnet.csv")
    ev.evaluate()