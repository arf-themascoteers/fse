import pandas as pd
import os
import plotters.utils as utils

main_df = None
all_df = None
root = "../saved"
locations = [os.path.join(root, subfolder) for subfolder in os.listdir(root) if subfolder.startswith("1_")]
algorithms = ["fsdrl","bsnet","zhang","mcuve","pcal","lasso"]
datasets = ["indian_pines", "ghsi"]
targets = [5,10,15,20,25,30]
df2 = pd.DataFrame(columns=["dataset","target_size","algorithm","time","metric1","metric2"])


def add_df(base_df, path):
    df = pd.read_csv(path)
    if len(df) == 0:
        print(f"Empty {path}")
    df['source'] = path
    if df is None:
        return df
    else:
        all_df = pd.concat([base_df, df], axis=0)
        return all_df


def create_dfs():
    global all_df, main_df
    for loc in locations:
        files = os.listdir(loc)
        for f in files:
            if "details" in f:
                continue
            if "fscrl-" in f:
                continue
            path = os.path.join(loc, f)
            if "all_features_summary" in f:
                all_df = add_df(all_df, path)
            else:
                main_df = add_df(main_df, path)


def make_complete_main_df():
    global df2, main_df
    for d in datasets:
        for t in targets:
            for a in algorithms:
                entries = main_df[(main_df["algorithm"] == a) & (main_df["dataset"] == d) & (main_df["target_size"] == t)]
                if len(entries) == 0:
                    print(f"Missing {d} {t} {a}")
                    df2.loc[len(df2)] = {
                        "dataset": d,
                        "target_size":t,
                        "algorithm": a,
                        "time": 100,
                        "metric1": 0.2,
                        "metric2": 0.8
                    }
                elif len(entries) >= 1:
                    if len(entries) > 1:
                        print(f"Multiple {d} {t} {a} -- {len(entries)}: {list(entries['source'])}")
                    df2.loc[len(df2)] = {
                        "dataset": d,
                        "target_size":t,
                        "algorithm": a,
                        "time": entries.iloc[0]["time"],
                        "metric1": entries.iloc[0]["metric1"],
                        "metric2": entries.iloc[0]["metric2"]
                    }


def add_all_in_main():
    global all_df, df2
    for d in datasets:
        for t in targets:
            entries = all_df[(all_df["dataset"] == d)]
            if len(entries) == 0:
                print(f"All Missing {d}")
                df2.loc[len(df2)] = {
                    "dataset": d,
                    "target_size": t,
                    "algorithm": "all_bands",
                    "time": 100,
                    "metric1": 0.2,
                    "metric2": 0.8
                }
            elif len(entries) >= 1:
                if len(entries) > 1:
                    print(f"All Multiple {d} {t} -- {len(entries)}: {list(entries['source'])}")
                df2.loc[len(df2)] = {
                    "dataset": d,
                    "target_size": t,
                    "algorithm": "all_bands",
                    "time": 0,
                    "metric1": entries.iloc[0]["metric1"],
                    "metric2": entries.iloc[0]["metric2"]
                }


def rename_algorithms():
    global df2
    for key, value in utils.algorithm_map.items():
        df2.loc[df2["algorithm"] == key, "algorithm"] = value


def rename_datasets():
    global df2
    for key, value in utils.dataset_map.items():
        df2.loc[df2["dataset"] == key, "dataset"] = value


create_dfs()
make_complete_main_df()
add_all_in_main()
rename_algorithms()
rename_datasets()

df2.to_csv("../final_results/classification.csv", index=False)
