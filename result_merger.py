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
        d['source'] = p
        if df is None:
            df = d
        else:
            df = pd.concat([df, d], axis=0)


algorithms = ["fsdrl","bsnet","mcuve","pcal","lasso", "spa"]
datasets = ["lucas_full","lucas_skipped","lucas_downsampled","lucas_min","indian_pines", "ghsi"]
targets = [5,10,15,20,25,30]

for d in datasets:
    for a in algorithms:
        for t in targets:
            entries = df[ (df["algorithm"] == a) & (df["dataset"] == d) & (df["target_size"] == t)]
            if len(entries) == 0:
                print(f"------Missing {d} {t} {a}")
                pass
            if len(entries) > 1:
                print(f"Multiple {d} {t} {a} -- {len(entries)}: {list(entries['source'])}")
                pass
