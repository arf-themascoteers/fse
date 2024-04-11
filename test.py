from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    #for dataset in ["lucas_full","brazilian","lucas_skipped"]:
    #for dataset in ["demmin","brazilian"]:
    for dataset in ["brazilian"]:
        for algorithm in ["fsdr"]:
            for size in [2,5,10,20]:
                tasks.append(
                    {
                        "dataset": dataset,
                        "target_feature_size": size,
                        "algorithm": algorithm
                    }
                )
    ev = Evaluator(tasks)
    ev.evaluate()