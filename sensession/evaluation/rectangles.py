import gc
from pathlib import Path
from itertools import product

import polars as pl
import matplotlib.pyplot as plt
from loguru import logger

from sensession.lib.database import Database
from sensession.lib.frame_generation import TRIVIAL_MASK_NAME
from sensession.evaluation.common.boxplots import boxplot
from sensession.evaluation.common.lineplot import lineplot
from sensession.evaluation.common.preprocess import (
    unwrap,
    detrend_phase,
    equalize_phase,
    equalize_magnitude,
)


def scale_magnitude(df: pl.DataFrame) -> pl.DataFrame:
    # ----------------------------------------------------------------------------------
    # -- Voltage correction to normalize CSI magnitudes
    # ----------------------------------------------------------------------------------
    # Normalize by voltages.
    # Calculate voltage by CSI magnitude mean per capture
    # Then divide by voltage to equalize AGC influence
    logger.trace(
        "Scaling CSI magnitude to remove AGC influence -- dividing by voltage "
        + "in each frame."
    )
    df = df.with_columns(
        # NOTE: factor is stored in the session name. Because dots arent nice there, the number extracted is
        # factor * 100. Hence we need to divide by 100 to get the actual factor.
        # NOTE: Our sessions are named like:
        #  > receiver_idx_{idx}_absfactor_{int(factor*100)}e-2
        # This regex groups into the modified indices
        factor=pl.col("session_name")
        .str.extract("^.*absfactor_(.+)e-2")
        .cast(pl.Float32)
        .truediv(100),
        modified_idx=pl.col("session_name")
        .str.extract("^.*_(.+)_absfactor_.*")
        .cast(pl.Int16)
        - 32,
    )

    return df.join(
        # We filter out the modified and neighboring subcarriers to not have them affect our
        # voltage normalization.
        df.filter(
            (pl.col("subcarrier_idxs") > pl.col("modified_idx") + 4)
            | (pl.col("subcarrier_idxs") < pl.col("modified_idx") - 4)
        )
        .group_by("row_nr")
        .agg(pl.col("csi_abs").mean().alias("voltage")),
        on="row_nr",
    ).with_columns(
        volt_normed_csi=pl.col("csi_abs") / pl.col("voltage"),
    )


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocess CSI to attach correct values etc.
    """

    if df.is_empty():
        raise ValueError(
            f"Cant preprocess empty dataframe. Maybe you set the wrong checkpoint?"
        )
    logger.trace(f"Dataframe shape before preprocessing: {df.shape}")

    # Unwrap and explode everything
    df = unwrap(df)
    gc.collect()
    logger.trace(f"Dataframe shape post unwrapping: {df.shape}")

    df = scale_magnitude(df)
    gc.collect()
    logger.trace(f"Dataframe shape post scaling: {df.shape}")

    df = equalize_magnitude(df)
    gc.collect()
    logger.trace(f"Dataframe shape post magnitude equalization: {df.shape}")

    df = detrend_phase(df)
    gc.collect()
    logger.trace(f"Dataframe shape post phase detrending: {df.shape}")

    df = equalize_phase(df)
    gc.collect()
    logger.trace(f"Dataframe shape post phase equalization: {df.shape}")

    # df = remove_phase_outliers(df)
    gc.collect()
    logger.trace(f"Dataframe shape post phase outlier removal: {df.shape}")

    # Attach timed session id, nice for visualization.
    return df.with_columns(
        pl.concat_str(
            [
                pl.col("session_name"),
                pl.lit("\n("),
                pl.col("session_timestamp"),
                pl.lit(", "),
                pl.col("session_id"),
                pl.lit(")"),
            ],
            separator="",
        ).alias("timed_session_id")
    )


def get_masked_data(exp_name: str):
    cache_path = Path.cwd() / ".cache" / "preprocessed" / exp_name
    cache_path.mkdir(exist_ok=True, parents=True)
    cache_file = cache_path / "db.parquet"

    if cache_file.is_file():
        logger.trace("Data already preprocessed. Loading cached file ...")
        data = pl.read_parquet(cache_file)
        logger.trace(f"Loaded data of shape: {data.shape}")
        return data

    db_path = Path.cwd() / "data" / exp_name / "db.parquet"
    logger.trace(f"Loading data from database {db_path}")
    if not db_path.is_file():
        logger.trace(f"Path {db_path} does not point to a file")

    db = Database(db_path)
    df = db.get()

    # TODO: We actually would like not to drop stuff.
    # Turns out, for an intricate experiment, we are processing structures too large for memory though (> 100 GB)
    # Therefore, we drop unused data here.
    # Ultimately this might be better suited with using a second dataframe that contains experiment ids mapped
    # to some duplicated values and employing the polars lazy API more (doesnt work well right now)
    unused_columns = [
        "experiment_start_time",
        "mask_id",
        "bandwidth",
        "tx_gain",
        "channel",
        "rssi",
        "timestamp",
        "frame_id",
        "experiment_id",
    ]
    df = df.drop(unused_columns)

    # Run preprocessing to calculate shapes etc
    data = preprocess(df)

    # Drop not further needed data (training data and some fields)
    data = data.filter(pl.col("mask_name") != TRIVIAL_MASK_NAME)
    data = data.drop("session_timestamp")

    logger.trace(
        f"Storing preprocessed dataframe in cache ({cache_file}) for future speedup."
    )
    data.write_parquet(cache_file)

    return data


def plot_single_mask_amplitudes(idx: int, df: pl.DataFrame, outdir: Path):
    """
    Plot shapes for modifications with a single (rectangle) mask.

    Shapes graphed are corrected by a session profile extracted as mean from the initial
    warmup training frames.

    All receivers are amassed as subplots in one figure.
    """
    num_receiver = df.select(pl.col("receiver_name")).n_unique()
    num_used_antennas = df.select(pl.col("used_antenna_idxs")).n_unique()

    df = df.group_by(
        "factor", "receiver_name", "used_antenna_idxs", "subcarrier_idxs"
    ).agg(pl.col("shape_normed_csi_abs").mean())

    group_df = df.sort("receiver_name").group_by(
        "receiver_name", "used_antenna_idxs", maintain_order=True
    )

    # Create subplots. Leave space for legend to the right.
    # We put box and scatterplot side by side.
    f, axes = plt.subplots(
        num_receiver,
        num_used_antennas,
        figsize=(14 * num_used_antennas, 6 * num_receiver),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    plt.subplots_adjust(right=0.75)

    subplt_idxs = product(range(num_receiver), range(num_used_antennas))

    for (i, j), ((receiver_name, antenna), data) in zip(subplt_idxs, group_df):
        ax = axes[i, j]

        lineplot(
            data=data,
            x_label="subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Amplitude",
            y_col="shape_normed_csi_abs",
            ax=ax,
            title=f"Receiver: {receiver_name}, Antenna: {antenna}",
            hue_col="factor",
            lower_ylim=0,
        )
        plt.tight_layout(h_pad=0.15, rect=[0, 0.03, 1, 0.95])

    f.suptitle("Shape-normalized CSI from different amplitude-precoded masks")
    f.savefig(outdir / f"idx_{idx}_normed_csi.png", dpi=f.dpi)
    plt.close(f)


def plot_single_mask_phases(idx: int, df: pl.DataFrame, outdir: Path):
    """
    Plot phase shapes for one mask
    """
    num_factors = df.select(pl.col("factor")).n_unique()
    num_receiver = df.select(pl.col("receiver_name")).n_unique()
    num_used_antennas = df.select(pl.col("used_antenna_idxs")).n_unique()

    assert num_used_antennas == 1, (
        "This visualization assumes single antenna captures. "
        "Rewrite it (in a loop) to create multiple pictures."
    )

    group_df = df.sort("factor", "receiver_name").group_by(
        "receiver_name", "factor", maintain_order=True
    )

    # Create subplots. Leave space for legend to the right.
    # We put box and scatterplot side by side.
    f, axes = plt.subplots(
        num_factors,
        num_receiver,
        figsize=(10 * num_receiver, 6 * num_factors),
        sharex=True,
        squeeze=False,
    )
    plt.subplots_adjust(right=0.75)

    subplt_idxs = product(range(num_factors), range(num_receiver))

    for (i, j), ((receiver_name, factor), data) in zip(subplt_idxs, group_df):
        ax = axes[i, j]
        csi_phase = data.select(
            "subcarrier_idxs",
            "timed_session_id",
            "normed_csi_phase",
            "shape_normed_csi_phase",
        )

        # --------------------------------------------------------------------------
        # Boxplot of boundary-corrected CSI phase
        boxplot(
            data=csi_phase,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Phase",
            y_col="normed_csi_phase",
            ax=ax,
            title=(
                f"Boundary-normed CSI phases \nReceiver: {receiver_name}, factor:"
                f" {factor}"
            ),
            leg_bbox=(0.0, -0.1),
            leg_loc="upper left",
        )

        plt.tight_layout(h_pad=0.15, rect=[0, 0.03, 1, 0.95])

    f.suptitle("Detected CSI phase values (boundary-corrected)")
    f.savefig(outdir / f"idx_{idx}_raw_phases.png", dpi=f.dpi)
    plt.close(f)


def plot_single_mask_phase_corrected(idx: int, df: pl.DataFrame, outdir: Path):
    """
    Plot phases (corrected by session profiles) for one mask
    """
    num_factors = df.select(pl.col("factor")).n_unique()
    num_receiver = df.select(pl.col("receiver_name")).n_unique()
    num_used_antennas = df.select(pl.col("used_antenna_idxs")).n_unique()

    assert num_used_antennas == 1, (
        "This visualization assumes single antenna captures. "
        "Rewrite it (in a loop) to create multiple pictures."
    )

    group_df = df.sort("factor", "receiver_name").group_by(
        "receiver_name", "factor", maintain_order=True
    )

    # Create subplots. Leave space for legend to the right.
    # We put box and scatterplot side by side.
    f, axes = plt.subplots(
        num_factors,
        num_receiver,
        figsize=(10 * num_receiver, 6 * num_factors),
        sharex=True,
        squeeze=False,
    )
    plt.subplots_adjust(right=0.75)

    subplt_idxs = product(range(num_factors), range(num_receiver))

    for (i, j), ((receiver_name, factor), data) in zip(subplt_idxs, group_df):
        ax = axes[i, j]
        csi_phase = data.select(
            "subcarrier_idxs",
            "timed_session_id",
            "normed_csi_phase",
            "shape_normed_csi_phase",
        )

        # --------------------------------------------------------------------------
        # Boxplot of shape normed CSI phase
        boxplot(
            data=csi_phase,
            x_label="Subcarrier idx",
            x_col="subcarrier_idxs",
            y_label="CSI Phase",
            y_col="shape_normed_csi_phase",
            ax=ax,
            title=(
                f"Shape-normalized CSI phases \nReceiver: {receiver_name}, Factor:"
                f" {factor}"
            ),
            leg_bbox=(0.0, -0.1),
            leg_loc="upper left",
        )

        plt.tight_layout(h_pad=0.15, rect=[0, 0.03, 1, 0.95])

    f.suptitle("Detected CSI phase values (boundary-corrected)")
    f.savefig(outdir / f"idx_{idx}_shape_corrected_phases.png", dpi=f.dpi)
    plt.close(f)


def plot_all_shapes(df: pl.DataFrame, outdir: Path):
    """
    Plot separate images, each of which shows the masked and corrected CSI shape.
    """

    for modified_idx, group in df.group_by(pl.col("modified_idx")):
        logger.trace(
            "Plotting CSI magnitude for session with amplitude scaling on subcarrier"
            + f" index {modified_idx}"
        )

        plot_single_mask_amplitudes(modified_idx, group, outdir)
        plot_single_mask_phases(modified_idx, group, outdir)
        plot_single_mask_phase_corrected(modified_idx, group, outdir)


if __name__ == "__main__":
    exp_name = "rectangle_precoding"
    outdir = Path.cwd() / "data" / exp_name / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    data = get_masked_data(exp_name)
    plot_all_shapes(data, outdir)
