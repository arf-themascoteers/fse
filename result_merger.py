import pandas as pd
import os

df = None
root = "saved"
locations = [os.path.join(root, subfolder) for subfolder in os.listdir(root) if subfolder.startswith("1_")]


for loc in locations:
    files = os.listdir(loc)
    for f in files:
        if "details" in f:
            continue
        if "all_features_" in f:
            continue
        if "fscrl-" in f:
            continue
        p = os.path.join(loc, f)
        d = pd.read_csv(p)
        if len(d) == 0:
            print(f"Empty {p}")
        d['source'] = p
        if df is None:
            df = d
        else:
            df = pd.concat([df, d], axis=0)


algorithms = ["fsdrl","bsnet","mcuve","pcal","lasso", "spa"]
datasets = ["lucas_full","lucas_skipped","lucas_downsampled","lucas_min","indian_pines", "ghsi"]
targets = [5,10,15,20,25,30]

df2 = pd.DataFrame(columns=["dataset","target_size","algorithm","time","metric1","metric2"])

for d in datasets:
    for a in algorithms:
        for t in targets:
            entries = df[ (df["algorithm"] == a) & (df["dataset"] == d) & (df["target_size"] == t)]
            if len(entries) == 0:
                print(f"Missing {d} {t} {a}")
            elif len(entries) > 1:
                print(f"Multiple {d} {t} {a} -- {len(entries)}: {list(entries['source'])}")
                pass
            elif len(entries) == 1:
                df2.loc[len(df2)] = {
                    "dataset": d,
                    "target_size":t,
                    "algorithm": a,
                    "time": entries.iloc[0]["time"],
                    "metric1": entries.iloc[0]["metric1"],
                    "metric2": entries.iloc[0]["metric2"]
                }
            else:
                df2.loc[len(df2)] = {
                    "dataset": d,
                    "target_size":t,
                    "algorithm": a,
                    "time": 100,
                    "metric1": 0,
                    "metric2": 1
                }

maps = {
    "fsdrl":"BSDR",
    "bsnet":"BS-Net-FC",
    "mcuve":"MCUVE",
    "pcal":"PCA-loading",
    "lasso":"LASSO",
    "spa":"SPA",
}

for key, value in maps.items():
    df2.loc[df2["algorithm"] == key, "algorithm"] = value

df2.to_csv("saved/final.csv", index=False)