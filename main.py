from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    for algorithm in ["pca","pcat95","rfe","sfs","sbs","fscr"]:
        for size in [5,10,15,20,25,30]:
            tasks.append(
                {
                    "reduced_features":True,
                    "reduced_rows":False,
                    "target_feature_size": size,
                    "algorithm": algorithm
                }
            )
    ev = Evaluator(tasks)
    ev.evaluate()