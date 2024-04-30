from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["zhang","fsdrl"],
        "datasets" : ["indian_pines","ghsi"],
        "target_sizes" : [5,10,15,20,25,30]
    }
    ev = Evaluator(tasks,1,10,"zhang.csv")
    ev.evaluate()