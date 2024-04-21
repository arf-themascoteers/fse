import pandas as pd


df = pd.DataFrame(data=[[1,2,3],[4,5,6],[1,2,4]],columns=["fold","dataset","r2"])
df = df[df['dataset'] == 2]
m = df["r2"].mean()
print(m)