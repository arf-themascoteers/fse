from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["fsdrl","pcal","lasso","mcuve"],
        "datasets" : ["lucas_min"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,10,"min.csv")
    ev.evaluate()