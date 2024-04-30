import os.path
import numpy as np
import plotly.express as px
import pandas as pd
import plotters.utils as utils
import plotly.graph_objects as go

root = "../saved_figs"


df_original = pd.read_csv("../final_results/regression.csv")


for metric in ["time","metric1", "metric2"]:
    for dataset in ["LUCAS","LUCAS (Skipped)", "LUCAS (Downsampled)", "LUCAS (Truncated)"]:
        main_df = df_original[df_original["dataset"] == dataset]
        df_all_bands = main_df[main_df["algorithm"] == "All Bands"]
        df_ex_all_bands = main_df[main_df["algorithm"] != "All Bands"]
        fig = px.line(df_ex_all_bands, x='target_size', y=metric,
                      color="algorithm",
                      markers= ".",
                      labels={"target_size": "Number of selected bands", metric: utils.metric_map[metric][dataset], "algorithm":"Algorithms"})

        if metric != "time":
            additional_trace = go.Scatter(x=df_all_bands["target_size"], y=df_all_bands[metric], mode='lines', line=dict(dash='dash'), name='All Bands')
            fig.add_trace(additional_trace)

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
        subfolder = os.path.join(root, "regression")
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        subfolder = os.path.join(subfolder, metric)
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        path = os.path.join(subfolder, f"{dataset}.png")
        fig.write_image(path, scale=5)


