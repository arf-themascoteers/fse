import pandas as pd

df = pd.read_csv("../../results/fscr.csv")
first_row_band_columns = df.loc[0, df.columns[df.columns.str.startswith('band')]]
ar = [f"{int(a)}" for a in first_row_band_columns]
print(ar)

first_row_band_columns = df.loc[len(df)-1, df.columns[df.columns.str.startswith('band')]]
ar = [f"{int(a)}" for a in first_row_band_columns]
print(ar)