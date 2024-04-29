import pandas as pd
import os

locations = ["saved/1_all_ex_spa","saved/1_bsnet_lucas_min","saved/1_mcuve_spa"]
df = None

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
        if df is None:
            df = d
        else:
            df = pd.concat([df, d], axis=0)

#6
algorithms = ["fsdrl","bsnet","mcuve","pcal","lasso", "spa"]

#6
datasets = ["lucas_full","lucas_skipped","lucas_downsampled","lucas_min","indian_pines", "ghsi"]
# for al in algorithms:
#     lasso = df[df["algorithm"] == al]
#     print(al,len(lasso))
#
# counter = 0
# for al in datasets:
#     lasso = df[df["dataset"] == al]
#     print(al,len(lasso))
#     counter = counter + len(lasso)
#

#6
targets = [5,10,15,20,25,30]

for d in datasets:
    for a in algorithms:
        for t in targets:
            entries = df[ (df["algorithm"] == a) & (df["dataset"] == d) & (df["target_size"] == t)]
            if len(entries) == 0:
                print(f"Missing {d} {t} {a}")
            if len(entries) > 1:
                print(f"Multiple {d} {t} {a}")
