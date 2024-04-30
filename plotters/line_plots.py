import plotly.express as px
import pandas as pd

dataset_maps = {
    "lucas_full":"LUCAS",
    "lucas_skipped":"LUCAS (Skipped)",
    "lucas_downsampled":"LUCAS (Downsampled)",
    "lucas_min":"LUCAS (Truncated)",
    "indian_pines":"Indian Pines",
    "ghsi":"GHSI",
}

metric1_map = {
    "LUCAS": "R^2",
    "LUCAS (Skipped)": "R^2",
    "LUCAS (Downsampled)": "R^2",
    "LUCAS (Truncated)": "R^2",
    "Indian Pines": "OA (%)",
    "GHSI": "OA (%)",
}

df_original = pd.read_csv("../saved/final.csv")
df_original["metric1"] = df_original["metric1"] * 100

for dataset in dataset_maps.values():
    df = df_original[df_original["dataset"]==dataset]
    fig = px.line(df, x='target_size', y='metric1',# text='metric1',
                    title=dataset,
                     color="algorithm",
                     markers= ".",
                     labels={"target_size": "Number of selected bands", "metric1": metric1_map[dataset], "algorithm":"Algorithms"})  # Adding labels and title

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

    fig.write_image(f"../saved_figs/{dataset}.png", scale=5)

