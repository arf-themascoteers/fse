import plotly.express as px
import pandas as pd

df = pd.read_csv("../saved/final.csv")
df["metric1"] = df["metric1"] * 100
df = df[df["dataset"]=="indian_pines"]

#df2 = df[df["algorithm"]=="fsdrl"]
# Create a scatter plot
fig = px.line(df, x='target_size', y='metric1',# text='metric1',
                 color="algorithm",
                 markers= ".",
                 labels={"target_size": "Number of selected bands", "metric1": "OA (%)", "algorithm":"Algorithms"})  # Adding labels and title

fig.update_layout({
    'plot_bgcolor': 'white',  # Changes the plot background color to white
    'paper_bgcolor': 'white',  # Changes the paper background color to white
})
fig.show()

fig.write_image("../saved_figs/indian_pines.png", scale=5)