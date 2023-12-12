import pandas as pd

df = pd.read_csv("data/dataset_66.csv")
df = df.sample(frac=1)
spl = int(.9 * len(df))
train = df.iloc[:spl]
test = df.iloc[spl:]

train.to_csv("data/train/dataset_66.csv", index=False)
test.to_csv("data/test/dataset_66.csv", index=False)


# df = pd.read_csv("data/dataset.csv")
# df = df.sample(frac=1)
# spl = int(.9 * len(df))
# train = df.iloc[:spl]
# test = df.iloc[spl:]
#
# train.to_csv("data/train/dataset.csv", index=False)
# test.to_csv("data/test/dataset.csv", index=False)
#
# truncated_train = train.sample(frac=0.04)
#
# truncated_train.to_csv("data/train/dataset_min.csv", index=False)