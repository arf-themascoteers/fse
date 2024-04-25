from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["bsnetig"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [5]
    }
    ev = Evaluator(tasks,1,1,"bsnetig.csv")
    ev.evaluate()