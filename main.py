from evaluator import Evaluator

if __name__ == '__main__':
    sizes = [2]
    #sizes = [10, 50, 100, 150, 200, 250, 300, 350]
    tasks = []
    for size in sizes:
        for algorithm in ["fscr"]:
            tasks.append(
                {
                    "reduced_features":True,
                    "reduced_rows":True,
                    "target_feature_size": size,
                    "algorithm": algorithm
                }
            )
    ev = Evaluator(tasks)
    ev.evaluate()