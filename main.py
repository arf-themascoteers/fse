from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    for algorithm in ["boruta","fm","kbest"]:
        for size in [10,50,100,150,200,1000,2000]:
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