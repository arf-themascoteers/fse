import pandas as pd

d = "data/demmin.csv"
df = pd.read_csv(d)
c = list(df.columns)
print(len(c))
print(c)
