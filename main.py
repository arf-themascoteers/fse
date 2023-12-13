from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    for algorithm in ["mi","sfs","lasso","fsdr"]:
        for size in [2, 5, 10, 15, 20]:
            tasks.append(
                {
                    "reduced_features":True,
                    "reduced_rows":False,
                    "target_feature_size": size,
                    "algorithm": algorithm
                }
            )
    ev = Evaluator(tasks)
    ev.evaluate()