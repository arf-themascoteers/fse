from ds_manager import DSManager
from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    tasks.append(
        {
            "dataset": DSManager(reduced_features=False, reduced_rows=False),
            "target_feature_size": 100,
            "algorithm": "pca"
        }
    )
    tasks.append(
        {
            "dataset": DSManager(reduced_features=False, reduced_rows=False),
            "target_feature_size": 100,
            "algorithm": "pcat95"
        }
    )
    tasks.append(
        {
            "dataset": DSManager(reduced_features=False, reduced_rows=False),
            "target_feature_size": 100,
            "algorithm": "fscr"
        }
    )
    tasks.append(
        {
            "dataset": DSManager(reduced_features=False, reduced_rows=False),
            "target_feature_size": 100,
            "algorithm": "rfe"
        }
    )
    ev = Evaluator(tasks)
    ev.evaluate()