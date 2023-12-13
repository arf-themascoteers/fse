import pandas as pd

def get_dataset(rows, columns):
    if columns == 66:
        return "downsampled"
    if rows == 21782:
        return "original"
    return "truncated"

df = pd.read_csv("results/good/results.csv")
df.drop(inplace=True, columns=["selected_features"])
time = {}
r2 = {}
rmse = {}

dataset = None
data_time = {}
data_r2 = {}
data_rmse = {}
algorithm = None

for index, row in df.iterrows():
    dataset = get_dataset(row["rows"], row["columns"])

    if dataset not in data_time:
        data_time[dataset] = {}
        data_r2[dataset] = {}
        data_rmse[dataset] = {}

    t = str(row["target_size"])

    if t not in data_time[dataset]:
        data_time[dataset][t] = {}
        data_r2[dataset][t] = {}
        data_rmse[dataset][t] = {}

    algorithm = row["algorithm"]

    data_time[dataset][t][algorithm] = row["time"]
    data_r2[dataset][t][algorithm] = row["r2_test"]
    data_rmse[dataset][t][algorithm] = row["rmse_test"]

data = {
    "$time$" : data_time,
    "$R^2$" : data_r2,
    "$RMSE$" : data_rmse,
}

print(r"\begin{tabular}")
print(r"{|l|l|r|r|r|r|r|}\hline Metric & Dataset & t  & MI & SFS & LASSO & FSDR \\\hline")

for metric, metric_data in data.items():
    for dataset, dataset_data in metric_data.items():
        print(r"\multirow{15}{*}{"+metric+r"} & \multirow{5}{*}{"+dataset+"}")
        first = True
        for t,t_data in dataset_data.items():
            if not first:
                print(" &\t ", end="")
            else:
                print(" \t ", end="")
            first = False
            print(f" & {t} ", end="")
            for algorithm, value in t_data.items():
                formatted_number = "{:.2f}".format(value)
                print(f" & {algorithm}{formatted_number} ", end="")
            print("")

print("")
print(r"\end{tabular}")