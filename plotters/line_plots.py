import os.path

import plotly.express as px
import pandas as pd

root = "../saved_figs"

dataset_maps = {
    "lucas_full":"LUCAS",
    "lucas_skipped":"LUCAS (Skipped)",
    "lucas_downsampled":"LUCAS (Downsampled)",
    "lucas_min":"LUCAS (Truncated)",
    "indian_pines":"Indian Pines",
    "ghsi":"GHSI",
}

metric_map = {
    "time":{
        "LUCAS": "Time (seconds)",
        "LUCAS (Skipped)": "Time (seconds)",
        "LUCAS (Downsampled)": "Time (seconds)",
        "LUCAS (Truncated)": "Time (seconds)",
        "Indian Pines": "Time (seconds)",
        "GHSI": "Time (seconds)",
    },
    "metric1":{
        "LUCAS": "$R^2$",
        "LUCAS (Skipped)": "$R^2$",
        "LUCAS (Downsampled)": "$R^2$",
        "LUCAS (Truncated)": "$R^2$",
        "Indian Pines": "OA (%)",
        "GHSI": "OA (%)",
    },
    "metric2":{
        "LUCAS": "$RMSE$",
        "LUCAS (Skipped)": "$RMSE$",
        "LUCAS (Downsampled)": "$RMSE$",
        "LUCAS (Truncated)": "$RMSE$",
        "Indian Pines": "$\kappa$",
        "GHSI": "$\kappa$",
    }
}

df_original = pd.read_csv("../saved/final.csv")
df_original["metric1"] = df_original["metric1"] * 100

for metric in ["time","metric1", "metric2"]:
    for dataset in dataset_maps.values():
        df = df_original[df_original["dataset"]==dataset]
        fig = px.line(df, x='target_size', y=metric,
                      # text='metric1',
                        title=dataset,
                         color="algorithm",
                         markers= ".",
                         labels={"target_size": "Number of selected bands", "metric1": metric_map[metric][dataset], "algorithm":"Algorithms"})

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

