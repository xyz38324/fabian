from pathlib import Path

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from sensession.evaluation.common.scatterplot import scatterplot

plt.set_loglevel(level="warning")

basecolor = (0.4, 0.6, 0.8, 0.5)


def boxplot(
    data: pl.DataFrame,
    x_label: str,
    x_col: str,
    y_label: str,
    y_col: str,
    ax: plt.Axes,
    title: str,
    hue_col: str = None,
    color=None,
    leg_bbox=(1.02, 0.00),
    leg_loc="lower left",
    lower_ylim=None,
):
    img = sns.boxplot(
        x=x_col,
        y=y_col,
        data=data.to_pandas(),
        ax=ax,
        hue=hue_col,
        color=color,
        linewidth=0.85,
        flierprops={"marker": "x", "markersize": 4},
    )

    plt.draw()
    for i, box in enumerate(img.artists):
        col = box.get_facecolor()
        # last Line2D object for each box is the box's fliers
        plt.setp(img.lines[i * 6 + 5], mfc=col, mec=col)

    # Set ticks
    ax.tick_params(axis="x", labelrotation=60)
    for label in ax.get_xticklabels()[::2]:
        label.set_visible(False)

    # Add title and labels
    if hue_col:
        img.legend(loc=leg_loc, bbox_to_anchor=leg_bbox, ncol=1)

    if lower_ylim:
        ax.set_ylim(bottom=lower_ylim)

    img.set_title(title)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)


# -------------------------------------------------------------------------------------
# Phase plots
# -------------------------------------------------------------------------------------
def raw_phaseplots(receiver: str, df: pl.DataFrame, outdir: Path):
    # Split by antenna to create separate plots there
    num_used_antennas = df.select(pl.col("used_antenna_idxs")).n_unique()
    antenna_group_df = df.groupby("used_antenna_idxs")

    # Scale font for readability
    sns.set(font_scale=1.2)

    # Create subplots. Leave space for legend to the right.
    # We put box and scatterplot side by side.
    f, axes = plt.subplots(
        num_used_antennas,
        figsize=(14, 8 * num_used_antennas),
        sharey="row",
        squeeze=False,
    )
    plt.subplots_adjust(right=0.75)

    for i, (antenna, data) in enumerate(antenna_group_df):
        csi_phase = data.select("subcarrier_idxs", "timed_session_id", "csi_phase")

        # --------------------------------------------------------------------------
        # Boxplot
        ax = axes[i, 0]
        scatterplot(
            data=csi_phase,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Phase",
            y_col="csi_phase",
            hue_col="timed_session_id",
            ax=ax,
            title=f"Receiver: {receiver}, Antenna: {antenna}",
        )

    rcv_out_dir = outdir / receiver
    rcv_out_dir.mkdir(parents=True, exist_ok=True)
    f.suptitle("Raw CSI phase values")
    f.savefig(rcv_out_dir / "raw_csi_phase.png", dpi=f.dpi)
    plt.close(f)


def normed_phaseplots(receiver: str, df: pl.DataFrame, outdir: Path):
    # Split by antenna to create separate plots there
    num_used_antennas = df.select(pl.col("used_antenna_idxs")).n_unique()
    antenna_group_df = df.groupby("used_antenna_idxs")

    # Scale font for readability
    sns.set(font_scale=1.2)

    # Create subplots. Leave space for legend to the right.
    # We put box and scatterplot side by side.
    f, axes = plt.subplots(
        num_used_antennas,
        2,
        figsize=(35, 8 * num_used_antennas),
        sharey="row",
        squeeze=False,
    )
    plt.subplots_adjust(right=0.75)
    axes = axes.reshape(num_used_antennas, 2)

    for i, (antenna, data) in enumerate(antenna_group_df):
        csi_phase = data.select(
            "subcarrier_idxs", "timed_session_id", "normed_csi_phase"
        )

        # --------------------------------------------------------------------------
        # Boxplot
        ax = axes[i, 0]
        boxplot(
            data=csi_phase,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Phases",
            y_col="normed_csi_phase",
            color=basecolor,
            ax=ax,
            title=f"Receiver: {receiver}, Antenna: {antenna}",
        )

        # --------------------------------------------------------------------------
        # Scatter plot
        ax = axes[i, 1]
        scatterplot(
            data=csi_phase,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Phases",
            y_col="normed_csi_phase",
            hue_col="timed_session_id",
            ax=ax,
            title=f"Receiver: {receiver}, Antenna: {antenna}",
        )

    rcv_out_dir = outdir / receiver
    rcv_out_dir.mkdir(parents=True, exist_ok=True)
    f.suptitle("CSI phase values (boundary-lock corrected)")
    f.savefig(rcv_out_dir / "normed_csi_phase.png", dpi=f.dpi)
    plt.close(f)


def session_normed_phaseplots(receiver: str, df: pl.DataFrame, outdir: Path):
    # Split by antenna to create separate plots there
    num_used_antennas = df.select(pl.col("used_antenna_idxs")).n_unique()
    antenna_group_df = df.groupby("used_antenna_idxs")

    # Scale font for readability
    sns.set(font_scale=1.2)

    # Create subplots for side-by-side voltage and shape normalized plots
    f, axes = plt.subplots(
        num_used_antennas,
        2,
        figsize=(34, 8 * num_used_antennas + 5),
        sharey="row",
        squeeze=False,
    )
    axes = axes.reshape(num_used_antennas, 2)
    plt.subplots_adjust(bottom=0.3)

    for i, (antenna, data) in enumerate(antenna_group_df):
        csi_phase = data.select(
            "subcarrier_idxs",
            "timed_session_id",
            "normed_csi_phase",
            "shape_normed_csi_phase",
        )

        # --------------------------------------------------------------------------
        # Boxplot of voltage normed CSI
        ax = axes[i, 0]
        boxplot(
            data=csi_phase,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Phase",
            y_col="normed_csi_phase",
            ax=ax,
            title=(
                f"Boundary-normed CSI phases \nReceiver: {receiver}, Antenna: {antenna}"
            ),
            hue_col="timed_session_id",
            leg_bbox=(0.0, -0.1),
            leg_loc="upper left",
        )

        # --------------------------------------------------------------------------
        # Boxplot of shape + voltage normed CSI
        ax = axes[i, 1]
        boxplot(
            data=csi_phase,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Phase",
            y_col="shape_normed_csi_phase",
            ax=ax,
            title=(
                f"Shape-normalized CSI phases \nReceiver: {receiver}, Antenna:"
                f" {antenna}"
            ),
            hue_col="timed_session_id",
            leg_bbox=(0.0, -0.1),
            leg_loc="upper left",
        )

    rcv_out_dir = outdir / receiver
    rcv_out_dir.mkdir(parents=True, exist_ok=True)
    f.savefig(rcv_out_dir / "sess_normed_csi_phase.png", dpi=f.dpi)
    plt.close(f)


# -------------------------------------------------------------------------------------
# Magnitude plots
# -------------------------------------------------------------------------------------
def raw_absplots(receiver: str, df: pl.DataFrame, outdir: Path):
    # Split by antenna to create separate plots there
    num_used_antennas = df.select(pl.col("used_antenna_idxs")).n_unique()
    antenna_group_df = df.groupby("used_antenna_idxs")

    # Scale font for readability
    sns.set(font_scale=1.2)

    # Create subplots. Leave space for legend to the right.
    # We put box and scatterplot side by side.
    f, axes = plt.subplots(
        num_used_antennas,
        2,
        figsize=(35, 8 * num_used_antennas),
        sharey="row",
        squeeze=False,
    )
    plt.subplots_adjust(right=0.75)
    axes = axes.reshape(num_used_antennas, 2)

    for i, (antenna, data) in enumerate(antenna_group_df):
        csi_abs = data.select("subcarrier_idxs", "timed_session_id", "csi_abs")

        # --------------------------------------------------------------------------
        # Boxplot
        ax = axes[i, 0]
        boxplot(
            data=csi_abs,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Amplitudes",
            y_col="csi_abs",
            color=basecolor,
            ax=ax,
            title=f"Receiver: {receiver}, Antenna: {antenna}",
            lower_ylim=0,
        )

        # --------------------------------------------------------------------------
        # Scatter plot
        ax = axes[i, 1]
        scatterplot(
            data=csi_abs,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Amplitudes",
            y_col="csi_abs",
            hue_col="timed_session_id",
            ax=ax,
            title=f"Receiver: {receiver}, Antenna: {antenna}",
            lower_ylim=0,
        )

    rcv_out_dir = outdir / receiver
    rcv_out_dir.mkdir(parents=True, exist_ok=True)
    f.suptitle("Raw reported CSI absolute values")
    f.savefig(rcv_out_dir / "raw_csi_abs.png", dpi=f.dpi)
    plt.close(f)


def normed_absplots(receiver: str, df: pl.DataFrame, outdir: Path):
    # Split by antenna to create separate plots there
    num_used_antennas = df.select(pl.col("used_antenna_idxs")).n_unique()
    antenna_group_df = df.groupby("used_antenna_idxs")

    # Scale font for readability
    sns.set(font_scale=1.2)

    # Create subplots for side-by-side voltage and shape normalized plots
    f, axes = plt.subplots(
        num_used_antennas,
        2,
        figsize=(34, 8 * num_used_antennas),
        sharey="row",
        squeeze=False,
    )
    axes = axes.reshape(num_used_antennas, 2)

    for i, (antenna, data) in enumerate(antenna_group_df):
        csi_abs = data.select(
            "subcarrier_idxs",
            "timed_session_id",
            "volt_normed_csi",
            "shape_normed_csi_abs",
        )

        # --------------------------------------------------------------------------
        # Boxplot of voltage normed CSI
        ax = axes[i, 0]
        boxplot(
            data=csi_abs,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Amplitudes",
            y_col="volt_normed_csi",
            color=basecolor,
            ax=ax,
            title=(
                f"Voltage-normalized CSI absolute values \nReceiver: {receiver},"
                f" Antenna: {antenna}"
            ),
            lower_ylim=0,
        )

        # --------------------------------------------------------------------------
        # Boxplot of shape + voltage normed CSI
        ax = axes[i, 1]
        boxplot(
            data=csi_abs,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Amplitudes",
            y_col="shape_normed_csi_abs",
            color=basecolor,
            ax=ax,
            title=(
                f"Shape-normalized CSI absolute values \nReceiver: {receiver}, Antenna:"
                f" {antenna}"
            ),
            lower_ylim=0,
        )

    rcv_out_dir = outdir / receiver
    rcv_out_dir.mkdir(parents=True, exist_ok=True)
    f.savefig(rcv_out_dir / "normed_csi_abs.png", dpi=f.dpi)
    plt.close(f)


def session_normed_absplots(receiver: str, df: pl.DataFrame, outdir: Path):
    # Split by antenna to create separate plots there
    num_used_antennas = df.select(pl.col("used_antenna_idxs")).n_unique()
    antenna_group_df = df.groupby("used_antenna_idxs")

    # Scale font for readability
    sns.set(font_scale=1.2)

    # Create subplots for side-by-side voltage and shape normalized plots
    f, axes = plt.subplots(
        num_used_antennas,
        2,
        figsize=(34, 8 * num_used_antennas + 5),
        sharey="row",
        squeeze=False,
    )
    axes = axes.reshape(num_used_antennas, 2)
    plt.subplots_adjust(bottom=0.3)

    for i, (antenna, data) in enumerate(antenna_group_df):
        csi_abs = data.select(
            "subcarrier_idxs",
            "timed_session_id",
            "volt_normed_csi",
            "shape_normed_csi_abs",
        )

        # --------------------------------------------------------------------------
        # Boxplot of voltage normed CSI
        ax = axes[i, 0]
        boxplot(
            data=csi_abs,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Amplitudes",
            y_col="volt_normed_csi",
            ax=ax,
            title=(
                f"Voltage-normalized CSI absolute values \nReceiver: {receiver},"
                f" Antenna: {antenna}"
            ),
            hue_col="timed_session_id",
            leg_bbox=(0.0, -0.1),
            leg_loc="upper left",
            lower_ylim=0,
        )

        # --------------------------------------------------------------------------
        # Boxplot of shape + voltage normed CSI
        ax = axes[i, 1]
        boxplot(
            data=csi_abs,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Amplitudes",
            y_col="shape_normed_csi_abs",
            ax=ax,
            title=(
                f"Shape-normalized CSI absolute values \nReceiver: {receiver}, Antenna:"
                f" {antenna}"
            ),
            hue_col="timed_session_id",
            leg_bbox=(0.0, -0.1),
            leg_loc="upper left",
            lower_ylim=0,
        )

    rcv_out_dir = outdir / receiver
    rcv_out_dir.mkdir(parents=True, exist_ok=True)
    f.savefig(rcv_out_dir / "sess_normed_csi_abs.png", dpi=f.dpi)
    plt.close(f)


def create_boxplots(df: pl.DataFrame, outdir: Path):
    """
    Plot mean values

    Args:
        df     : A completely unwrapped dataframe (i.e. each CSI value on separate row)
        outdir : Directory to store images in
    """
    outdir.mkdir(exist_ok=True, parents=True)

    for receiver, data in df.groupby("receiver_name"):
        raw_absplots(receiver, data, outdir)
        normed_absplots(receiver, data, outdir)
        session_normed_absplots(receiver, data, outdir)

        raw_phaseplots(receiver, data, outdir)
        normed_phaseplots(receiver, data, outdir)
        session_normed_phaseplots(receiver, data, outdir)
