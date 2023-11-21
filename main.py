from evaluator import Evaluator

if __name__ == '__main__':
    sizes = [5,10,15,20,25,30]
    #sizes = [10, 50, 100, 150, 200, 250, 300, 350]
    tasks = []
    for size in sizes:
        for algorithm in ["fscrns","fscr"]:
            tasks.append(
                {
                    "reduced_features":False,
                    "reduced_rows":False,
                    "target_feature_size": size,
                    "algorithm": algorithm
                }
            )
    ev = Evaluator(tasks)
    ev.evaluate()