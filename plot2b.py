import pandas as pd
import plotly.express as px

df = pd.read_csv("b22.csv")

fig = px.scatter(df, x='band1', y='band2',
                    color='score',
                    color_continuous_scale='Viridis',
                    title='3D Scatter Plot',
                    opacity=1
                    )

fig.write_image("fig1.png", scale=5)