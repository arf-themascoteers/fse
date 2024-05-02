from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["spa"],
        "datasets" : ["lucas_full"],
        "target_sizes" : [20]
    }
    ev = Evaluator(tasks,1,1,"7_1_20.csv")
    ev.evaluate()