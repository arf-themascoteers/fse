from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["mcuve"],
        "datasets" : ["lucas_min", "lucas_full"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,10,"mcuve_luc.csv")
    ev.evaluate()