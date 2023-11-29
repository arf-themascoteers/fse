import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/bra.csv")
x = df.loc[0]
print(df.columns)