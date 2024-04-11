import pandas as pd

d = "data/brazilian.csv"
df = pd.read_csv(d)
c = list(df.columns)
print(len(c))
print(c)
