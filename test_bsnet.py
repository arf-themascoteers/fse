from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["bsnet"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [30]
    }
    ev = Evaluator(tasks,4,1,"bsnet3.csv")
    ev.evaluate()