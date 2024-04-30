from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["mcuve"],
        "datasets" : ["indian_pines", "ghsi", "lucas_skipped", "lucas_downsampled"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,10,"mcuve.csv")
    ev.evaluate()