from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    for algorithm in ["fm","fscr"]:
        for size in [10,100,200,1000,2000]:
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