from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    for algorithm in ["mcuve", "cars", "spa", "rf"]:
        for dataset in ["brazilian","lucas_skipped","lucas_full"]:
            for size in [2, 5, 10, 20, 30]:
                tasks.append(
                    {
                        "dataset": dataset,
                        "target_feature_size": size,
                        "algorithm": algorithm
                    }
                )
    ev = Evaluator(tasks,"mcsr.csv")
    ev.evaluate()