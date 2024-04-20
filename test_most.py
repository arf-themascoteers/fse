from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    for algorithm in ["mcuve", "cars", "lasso", "rf"]:
        for dataset in ["brazilian","lucas_skipped","lucas_full"]:
            for size in [5, 10, 15, 20, 25, 30]:
                tasks.append(
                    {
                        "dataset": dataset,
                        "target_feature_size": size,
                        "algorithm": algorithm
                    }
                )
    ev = Evaluator(tasks,"quick.csv")
    ev.evaluate()