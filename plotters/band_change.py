import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os


df = pd.read_csv("../results/fscrl-indian_pines-5-1714482682207401.csv")
band_labels = []
for i in range(1,6):
    band_labels.append(f"band_{i}")
band_labels = ["epoch"] + band_labels
df = df[band_labels]
limit = 4000
df = df[df["epoch"]<700]

fig = px.line(df, x='epoch', y="band_1",labels={"epoch": "Epoch", "band_1": "Band Index 1"})

for i in range(2,6):
    additional_trace = go.Scatter(x=df["epoch"], y=df[f"band_{i}"], mode='lines', name=f'Band Index {i}')
    fig.add_trace(additional_trace)

fig.update_layout({
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
    'title_x': 0.5
})
fig.show()
fig.write_image(f"../saved_figs/band_update_{limit}_epochs_IP.png", scale=5)
