from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    for algorithm in ["rfe"]:
        for size in range(4200):
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