import pandas as pd
import matplotlib.pyplot as plt

lucas = pd.read_csv("data/lucas_down_min.csv")
data = lucas.iloc[120,13:].to_numpy()
x = list(range(data.shape[0]))
plt.plot(x,data)
plt.show()

# lucas = pd.read_csv("data/brazilian.csv")
# data = lucas.iloc[50,1:].to_numpy()
# x = list(range(data.shape[0]))
# plt.plot(x,data)
# plt.show()