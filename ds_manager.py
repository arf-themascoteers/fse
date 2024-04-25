import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from data_splits import DataSplits


class DSManager:
    def __init__(self, name, folds=1):
        self.name = name
        self.folds = folds
        dataset_path = f"data/{name}.csv"
        df = pd.read_csv(dataset_path)
        self.X_columns = DSManager.get_spectral_columns(df)
        self.y_column = DSManager.get_y_column(self.name)
        df = df[self.X_columns+[self.y_column]]
        df = df.sample(frac=1, random_state=0)
        if self.get_task() != "regression":
            df[self.y_column], class_labels = pd.factorize(df[self.y_column])
        self.full_data = df.to_numpy()
        self.full_data = DSManager._normalize(self.full_data)

    def __repr__(self):
        return self.get_name()

    def get_task(self):
        DSManager.get_task_by_name(self.name)

    @staticmethod
    def get_dataset_names():
        return [
            "demmin",
            "brazilian",
            "lucas_min",
            "lucas_skipped",
            "lucas_downsampled",
            "lucas_full",
            "ghsi",
            "indian_pines"
        ]

    @staticmethod
    def get_task_by_name(dataset):
        if dataset in ["ghsi", "indian_pines"]:
            return "classification"
        return "regression"

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
        if dataset == "brazilian":
            return "MO (gddm3)"
        if dataset == "ghsi":
            return "crop"
        if dataset == "indian_pines":
            return "class"
        return "oc"


    @staticmethod
    def _normalize(data):
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        return data

    def get_k_folds(self):
        if self.folds == 1:
            yield self.get_all_set_X_y()
        else:
            kf = KFold(n_splits=self.folds)
            for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
                train_data = self.full_data[train_index]
                test_data = self.full_data[test_index]
                yield self.split_train_validation_ev_parts(train_data, test_data)

    def split_train_validation_ev_parts(self, train_data, test_data):
        train_data, validation_data = train_test_split(train_data, test_size=0.1, random_state=40)
        test_for_train_data, test_for_test_data = train_test_split(test_data, test_size=0.5, random_state=40)
        return DataSplits(*DSManager.get_X_y_from_data(train_data),
               *DSManager.get_X_y_from_data(validation_data),
               *DSManager.get_X_y_from_data(test_for_train_data),
               *DSManager.get_X_y_from_data(test_for_test_data)
               )

    def get_all_X_y(self):
        return self.get_X_y_from_data(self.full_data)

    @staticmethod
    def get_X_y_from_data(data):
        x = data[:, :-1]
        y = data[:, -1]
        return x, y

    def get_train_test(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.3, random_state=40)
        return train_data, test_data

    def get_all_set_X_y(self):
        train_data, test_data = self.get_train_test()
        return self.split_train_validation_ev_parts(train_data, test_data)


if __name__ == "__main__":
    d = DSManager("lucas_skipped")
    print(d.full_data.shape)
    d = DSManager("lucas_skipped",1)
    for fold, (train_x, train_y, validation_x, validation_y, test_for_train_x, test_for_train_y, test_for_test_x, test_for_test_y) in enumerate(d.get_k_folds()):
        print(fold)
        print(train_x.shape)
        print(train_y.shape)
        print(validation_x.shape)
        print(validation_y.shape)
        print(test_for_train_x.shape)
        print(test_for_train_y.shape)
        print(test_for_test_x.shape)
        print(test_for_test_y.shape)


    d = DSManager("lucas_skipped",10)
    for fold, (train_x, train_y, validation_x, validation_y, test_for_train_x, test_for_train_y, test_for_test_x, test_for_test_y) in enumerate(d.get_k_folds()):
        print(fold)
        print(train_x.shape)
        print(train_y.shape)
        print(validation_x.shape)
        print(validation_y.shape)
        print(test_for_train_x.shape)
        print(test_for_train_y.shape)
        print(test_for_test_x.shape)
        print(test_for_test_y.shape)
        print("bye")
        break
