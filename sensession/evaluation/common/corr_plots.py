from typing import List
from pathlib import Path
from itertools import product

import polars as pl
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.pyplot.set_loglevel(level="warning")


def corrplot(
    corr_data: pl.DataFrame,
    x_label: str,
    y_label: str,
    ticks: List[str],
    ax: plt.Axes,
    title: str,
):
    img = sns.heatmap(
        corr_data.to_pandas(),
        linewidths=0.5,
        ax=ax,
        xticklabels=ticks,
        yticklabels=ticks,
    )
    plt.draw()

    # Set ticks
    ax.tick_params(axis="x", labelrotation=60)
    for label in ax.get_xticklabels()[::2]:
        label.set_visible(False)

    for label in ax.get_yticklabels()[::2]:
        label.set_visible(False)

    # Add title and labels
    img.set_title(title)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)


def pearson_corr(receiver: str, df: pl.DataFrame, outdir: Path):
    num_used_antennas = df.select(pl.col("used_antenna_idxs")).n_unique()
    num_sessions = df.select(pl.col("timed_session_id")).n_unique()

    subplt_idxs = product(range(num_sessions), range(num_used_antennas))

    group_df = (
        df.groupby(
            "row_nr", "used_antenna_idxs", "session_timestamp", maintain_order=True
        )
        .agg("shape_normed_csi_abs", "subcarrier_idxs")
        .groupby("session_timestamp", "used_antenna_idxs")
    )

    # Create subplots for all antennas
    sns.set(font_scale=1.2)

    f, axes = plt.subplots(
        num_sessions,
        num_used_antennas,
        squeeze=False,
        figsize=(14 * num_used_antennas, 8 * num_sessions),
        sharey="all",
        sharex="all",
    )

    axes = axes.reshape(num_sessions, num_used_antennas)

    for (i, j), ((timed_session_id, antenna), data) in zip(subplt_idxs, group_df):
        csi = data.select(pl.col("shape_normed_csi_abs").list.to_struct()).unnest(
            "shape_normed_csi_abs"
        )
        corr = csi.corr()
        subcarrier_idxs = list(data.item(0, "subcarrier_idxs"))

        ax = axes[i, j]
        corrplot(
            corr_data=corr,
            x_label="Subcarrier",
            y_label="Subcarrier",
            ticks=subcarrier_idxs,
            ax=ax,
            title=(
                f"Receiver: {receiver}, Antenna: {antenna}, \nCaptured at:"
                f" {timed_session_id}"
            ),
        )

    rcv_out_dir = outdir / receiver
    rcv_out_dir.mkdir(parents=True, exist_ok=True)
    f.suptitle("Pearson Correlation Coefficient of CSI Abs")
    f.savefig(rcv_out_dir / "csi_corr.png", dpi=f.dpi)
    plt.close(f)


def create_corrplots(df: pl.DataFrame, outdir: Path):
    """
    Plot correlation matrix of normalized CSI

    Args:
        df     : A completely unwrapped dataframe (i.e. each CSI value on separate row)
        outdir : Directory to store images in
    """
    outdir.mkdir(exist_ok=True, parents=True)

    for receiver, data in df.groupby("receiver_name"):
        pearson_corr(receiver, data, outdir)
