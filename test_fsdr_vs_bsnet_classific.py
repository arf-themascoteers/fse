from evaluator import Evaluator

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["bsnet","fsdr"],
        "datasets" : ["indian_pines", "ghsi"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = Evaluator(tasks,1,5,"test_fsdr_vs_bsnet_classific.csv")
    ev.evaluate()