from evaluator import Evaluator

if __name__ == '__main__':
    sizes = [2,5,10,15,20]
    tasks = []
    for size in sizes:
        for algorithm in ["mi","rfe","lasso","fscr"]:
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