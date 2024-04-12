from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    for size in [2, 5, 10, 20, 30]:
        for dataset in ["lucas_full","brazilian","lucas_skipped"]:
            for algorithm in ["fsdr","mi","lasso"]:
                tasks.append(
                    {
                        "dataset": dataset,
                        "target_feature_size": size,
                        "algorithm": algorithm
                    }
                )
    ev = Evaluator(tasks)
    ev.evaluate()