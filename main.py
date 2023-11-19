from ds_manager import DSManager
from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    tasks.append(
        {
            "dataset": DSManager(reduced_features=True, reduced_rows=True),
            "target_feature_size": 10,
            "algorithm": "pca"
        }
    )
    tasks.append(
        {
            "dataset": DSManager(reduced_features=True, reduced_rows=True),
            "target_feature_size": 10,
            "algorithm": "pcat95"
        }
    )
    tasks.append(
        {
            "dataset": DSManager(reduced_features=True, reduced_rows=True),
            "target_feature_size": 10,
            "algorithm": "rfe"
        }
    )
    tasks.append(
        {
            "dataset": DSManager(reduced_features=True, reduced_rows=True),
            "target_feature_size": 10,
            "algorithm": "fscr"
        }
    )
    ev = Evaluator(tasks)
    ev.evaluate()