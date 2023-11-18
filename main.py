from ds_manager import DSManager
from evaluator import Evaluator

if __name__ == '__main__':
    algorithms = ["pca"]
    tasks = []
    # tasks.append(
    #     {
    #         "dataset":DSManager(reduced_features=True, reduced_rows=True),
    #         "target_feature_size":10,
    #         "algorithm": "pca"
    #     }
    # )
    # tasks.append(
    #     {
    #         "dataset":DSManager(reduced_features=True, reduced_rows=True),
    #         "target_feature_size":10,
    #         "algorithm": "pls"
    #     }
    # )
    tasks.append(
        {
            "dataset":DSManager(reduced_features=True, reduced_rows=True),
            "target_feature_size":10,
            "algorithm": "rfe"
        }
    )

    ev = Evaluator(tasks)
    ev.evaluate()