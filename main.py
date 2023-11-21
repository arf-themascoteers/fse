from evaluator import Evaluator

if __name__ == '__main__':
    #short_size = [5,10,15,20,25,30]
    big_size = [10,50,100,150,200,250,300,350]
    tasks = []
    tasks.append(
        {
            "reduced_features": False,
            "reduced_rows": False,
            "target_feature_size": 10,
            "algorithm": "pcat95"
        }
    )
    for size in big_size:
        for algorithm in ["fscrns","fscr"]:
            tasks.append(
                {
                    "reduced_features":False,
                    "reduced_rows":False,
                    "target_feature_size": size,
                    "algorithm": algorithm
                }
            )
    ev = Evaluator(tasks)
    ev.evaluate()