import os
import pandas as pd


class Reporter:
    def __init__(self, filename="results.csv"):
        self.summary_filename = filename
        self.details_filename = f"details_{self.summary_filename}"
        self.all_features_details_filename = f"all_features_details_{self.summary_filename}"
        self.all_features_summary_filename = f"all_features_summary_{self.summary_filename}"

        self.summary_file = os.path.join("results", self.summary_filename)
        self.details_file = os.path.join("results", self.details_filename)
        self.all_features_summary_file = os.path.join("results", self.all_features_summary_filename)
        self.all_features_details_file = os.path.join("results", self.all_features_details_filename)

        if not os.path.exists(self.summary_file):
            with open(self.summary_file, 'w') as file:
                file.write("dataset,target_size,algorithm,time,metric1,metric2,selected_features\n")

        if not os.path.exists(self.details_file):
            with open(self.details_file, 'w') as file:
                file.write("dataset,target_size,fold,algorithm,repeat,final_size,time,metric1,metric2,selected_features\n")

        if not os.path.exists(self.all_features_summary_file):
            with open(self.all_features_summary_file, 'w') as file:
                file.write("dataset,metric1,metric2\n")

        if not os.path.exists(self.all_features_details_file):
            with open(self.all_features_details_file, 'w') as file:
                file.write("fold,dataset,metric1,metric2\n")

    def write_details(self, algorithm, fold, repeat, final_size, time, metric1, metric2, selected_features):
        time = Reporter.sanitize_metric(time)
        metric1 = Reporter.sanitize_metric(metric1)
        metric2 = Reporter.sanitize_metric(metric2)
        with open(self.details_file, 'a') as file:
            file.write(f"{algorithm.splits.get_name()},{algorithm.target_size},{fold},{algorithm.get_name()},"
                       f"{repeat},"
                       f"{final_size},{time},{metric1},{metric2},{'-'.join([str(i) for i in selected_features])}\n")
        self.update_summary(algorithm)

    def update_summary(self, algorithm):
        df = pd.read_csv(self.details_file)
        df = df[(df["dataset"] == algorithm.splits.get_name()) & (df["algorithm"] == algorithm.get_name()) & (df["target_size"] == algorithm.target_size)]
        if len(df) == 0:
            return
        time = round(df["time"].mean(), 2)
        metric1 = Reporter.sanitize_metric(df["metric1"].mean())
        metric2 = Reporter.sanitize_metric(df["metric2"].mean())
        selected_features = '||'.join(df['selected_features'].astype(str))

        df2 = pd.read_csv(self.summary_file)
        mask = ((df2["dataset"] == algorithm.splits.get_name()) & (df2["target_size"] == algorithm.target_size) & (df2["algorithm"] == algorithm.get_name()))
        if len(df2[mask]) == 0:
            df2.loc[len(df2)] = {
                "dataset":algorithm.splits.get_name(), "target_size":algorithm.target_size, "algorithm": algorithm.get_name(),
                "time":time,"metric1":metric1,"metric2":metric2, "selected_features":selected_features
            }
        else:
            df2.loc[mask, 'time'] = time
            df2.loc[mask, 'metric1'] = metric1
            df2.loc[mask, 'metric2'] = metric2
            df2.loc[mask, 'selected_features'] = selected_features
        df2.to_csv(self.summary_file, index=False)

    def write_details_all_features(self, fold, dataset, metric1, metric2):
        metric1 = Reporter.sanitize_metric(metric1)
        metric2 = Reporter.sanitize_metric(metric2)
        with open(self.all_features_details_file, 'a') as file:
            file.write(f"{fold},{dataset},{metric1},{metric2}\n")
        self.update_summary_for_all_features(dataset)

    def update_summary_for_all_features(self, dataset):
        df = pd.read_csv(self.all_features_details_file)
        df = df[df["dataset"] == dataset]
        if len(df) == 0:
            return

        metric1 = max(round(df["metric1"].mean(),2),0)
        metric2 = max(round(df["metric2"].mean(),2),0)

        df2 = pd.read_csv(self.all_features_summary_file)
        mask = (df2['dataset'] == dataset)
        if len(df2[mask]) == 0:
            df2.loc[len(df2)] = {"dataset":dataset, "metric1":metric1, "metric2": metric2}
        else:
            df2.loc[mask, 'metric1'] = metric1
            df2.loc[mask, 'metric2'] = metric2
        df2.to_csv(self.all_features_summary_file, index=False)

    def get_saved_metrics(self, algorithm, fold, repeat):
        df = pd.read_csv(self.details_file)
        if len(df) == 0:
            return None, None, None, None, None
        rows = df.loc[(df["dataset"] == algorithm.splits.get_name()) & (df["target_size"] == algorithm.target_size) &
                      (df["fold"] == fold) & (df["algorithm"] == algorithm.get_name()) &
                      (df["repeat"] == repeat)
                      ]
        if len(rows) == 0:
            return None, None, None, None, None
        row = rows.iloc[0]
        return row["final_size"], row["time"], row["metric1"], row["metric2"], row["selected_features"]

    def get_saved_metrics_for_all_feature(self, fold, dataset):
        df = pd.read_csv(self.all_features_details_file)
        if len(df) == 0:
            return None, None
        rows = df.loc[(df['fold'] == fold) & (df['dataset'] == dataset)]
        if len(rows) == 0:
            return None, None
        row = rows.iloc[0]
        return row["metric1"], row["metric2"]

    @staticmethod
    def sanitize_metric(metric):
        return max(round(metric, 2), 0)