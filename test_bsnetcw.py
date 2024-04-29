from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["bsnetcw","bsnet"],
        "datasets" : ["indian_pines", "ghsi"],
        "target_sizes" : [30,25,20,15,10,5]
    }
    ev = Evaluator(tasks,1,1,"bsnetcw.csv")
    ev.evaluate()