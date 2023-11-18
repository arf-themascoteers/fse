import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import time


class Evaluator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.filename = str(int(time.time())) + ".csv"

    def write_head_row(self):
        with open(self.filename, 'w') as file:
            file.write("algorithm,dataset,rows,columns,time,target_size,r2,rmse,features\n")

    def write_row(self,algorithm,dataset,time,target_size,r2,rmse,features):
        with open(self.filename, 'w') as file:
            file.write(f"{algorithm},{dataset.count_name()},{dataset.count_features()},"
                       f"{time},{target_size},{r2},{rmse},{';'.join(features)}\n")

    def evaluate(self):
        for task in self.tasks:
            dataset = task["dataset"]
            target_feature_size = task["target_feature_size"]
            algorithm = task["algorithm"]
            X = dataset.X
            y = data.target

            X_standardized = StandardScaler().fit_transform(X)

            pca = PCA()
            X_pca = pca.fit_transform(X_standardized)

            cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            num_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1

            print(f"Number of components to retain 95% variance: {num_components}")

            pca = PCA(n_components=num_components)
            X_selected = pca.fit_transform(X_standardized)

            print(f"Selected Features:\n{X_selected}")