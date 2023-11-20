from evaluator import Evaluator

if __name__ == '__main__':
    short_size = [5,10,15,20,25,30]
    big_size = [10,100,200,250,1000,2000]
    tasks = []
    for size in short_size:
        tasks.append(
            {
                "reduced_features":True,
                "reduced_rows":True,
                "target_feature_size": size,
                "algorithm": "fscr"
            }
        )
    ev = Evaluator(tasks)
    ev.evaluate()