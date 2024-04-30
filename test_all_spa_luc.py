from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["spa"],
        "datasets" : ["lucas_min"],
        "target_sizes" : [20, 25, 30]
    }
    ev = Evaluator(tasks,1,1,"luc2.csv")
    ev.evaluate()