from evaluator import Evaluator

if __name__ == '__main__':
    for i in range(10):
        j = i+1
        tasks = {
            "algorithms" : ["fsdr"],
            "datasets" : ["lucas_full"],
            "target_sizes" : [10]
        }
        ev = Evaluator(tasks,10, 1,f"fsdr_{j}.csv")
        ev.evaluate()