from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["bsnet","fsdr"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [30]
    }
    ev = Evaluator(tasks,4,1,"bsnet.csv")
    ev.evaluate()