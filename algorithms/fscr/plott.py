import matplotlib.pyplot as plt
import pandas as pd


def plot_me_plz(filename):
    df = pd.read_csv(filename)
    all_columns = list(df.columns)
    band_columns = [col for col in all_columns if "band" in col]
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(20, 10))
    axes = axes.flatten()

    axes[0].plot(df["train_r2"].tolist(), label="train_r2")
    axes[0].set_title("R2")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("R2")
    axes[0].legend(loc='lower right')

    axes[1].plot(df["train_rmse"].tolist(), label="train_rmse")
    axes[1].set_title("RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].legend(loc='upper right')
    plt.tight_layout()

    plt.show()
    plt.figure(figsize=(30, 30))
    for band in band_columns:
        plt.plot(df[band].tolist(), label=band)
    plt.title("Bands")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel("Epoch")
    plt.ylabel("Band Index")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_me_plz("../../results/fscr.csv")
