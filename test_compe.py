from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    for dataset in ["brazilian","lucas_skipped","lucas_full"]:
        for size in [2, 5, 10, 20, 30]:
            for algorithm in ["spa"]:
                tasks.append(
                    {
                        "dataset": dataset,
                        "target_feature_size": size,
                        "algorithm": algorithm
                    }
                )
    ev = Evaluator(tasks)
    ev.evaluate()