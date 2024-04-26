from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["bsnet","fsdr"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,3,1,"bsnet.csv")
    ev.evaluate()