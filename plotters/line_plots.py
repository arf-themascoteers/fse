import os.path
import numpy as np
import plotly.express as px
import pandas as pd
import plotters.utils as utils

root = "../saved_figs"


df_original = pd.read_csv("../saved/final.csv")
df_original["metric1"] = df_original["metric1"] * 100
#df_original['time'] = df_original['time'].apply(lambda x: np.log(x))

for metric in ["time","metric1", "metric2"]:
    for dataset in utils.dataset_map.values():
        main_df = df_original[df_original["dataset"] == dataset]
        # df_all_bands = df[df["algorithm"] == "All Bands"]
        # df_ex_all_bands = df[df["algorithm"] != "All Bands"]
        fig = px.line(main_df, x='target_size', y=metric,
                      color="algorithm",
                      markers= ".",
                      labels={"target_size": "Number of selected bands", metric: utils.metric_map[metric][dataset], "algorithm":"Algorithms"})


        fig.update_layout({
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'title_x':0.5
        })

        fig.update_layout(
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            )
        )

        #fig.show()
        subfolder = os.path.join(root, metric)
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        path = os.path.join(subfolder, f"{dataset}.png")
        fig.write_image(path, scale=5)


