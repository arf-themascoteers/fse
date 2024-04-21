import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold


class DSManager:
    def __init__(self, name, folds=1):
        self.name = name
        self.folds = folds
        np.random.seed(0)
        dataset_path = f"data/{name}.csv"
        df = pd.read_csv(dataset_path)
        self.X_columns = DSManager.get_spectral_columns(df)
        self.y_column = DSManager.get_y_column(self.name)
        df = df[self.X_columns+[self.y_column]]
        df = df.sample(frac=1)
        self.full_data = df.to_numpy()
        self.full_data = DSManager._normalize(self.full_data)

    def __repr__(self):
        return self.get_name()

    def get_name(self):
        return self.name

    def count_rows(self):
        return self.full_data.shape[0]

    def count_features(self):
        return len(self.X_columns)

    @staticmethod
    def wavelengths_itr():
        wvs = []
        spec = 400
        while spec <= 2499.5:
            n_spec = spec
            if int(n_spec) == spec:
                n_spec = int(n_spec)
            wavelength = str(n_spec)
            yield wavelength
            spec = spec + 0.5
        return wvs

    @staticmethod
    def get_spectral_columns(df):
        return list(df.columns)[1:]

    @staticmethod
    def get_y_column(dataset):
        col = "oc"
        if "brazilian" in dataset:
            col = "MO (gddm3)"
        return col

    @staticmethod
    def _normalize(data):
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        return data

    def get_k_folds(self):
        if self.folds == 1:
            return self.get_train_test_X_y()
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
            train_data = self.full_data[train_index]
            test_data = self.full_data[test_index]
            yield *DSManager.get_X_y_from_data(train_data), *DSManager.get_X_y_from_data(test_data)

    def get_all_X_y(self):
        return self.get_X_y_from_data(self.full_data)

    @staticmethod
    def get_X_y_from_data(data):
        x = data[:, :-1]
        y = data[:, -1]
        return x, y

    def get_train_test(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.3, random_state=2)
        return train_data, test_data

    def get_train_test_X_y(self):
        train_data, test_data = self.get_train_test()
        return *DSManager.get_X_y_from_data(train_data), *DSManager.get_X_y_from_data(test_data)


if __name__ == "__main__":
    d = DSManager("brazilian")
    print(d.full_data.shape)
    d = DSManager("lucas_skipped",10)
    for fold_number, (train_x, train_y, test_x, test_y) in enumerate(d.get_k_folds()):
        print(fold_number, test_y[0])
