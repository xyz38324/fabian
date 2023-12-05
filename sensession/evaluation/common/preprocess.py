import gc

import numpy as np
import polars as pl
from loguru import logger

from sensession.lib.frame_generation import TRIVIAL_MASK_NAME

### Need th
def unwrap(df: pl.DataFrame) -> pl.DataFrame:
    # ----------------------------------------------------------------------------------
    # -- Unwrapping all fields as far as possible
    # ----------------------------------------------------------------------------------
    # Prepare dataframe by exploding antenna idxs
    logger.trace("Unwrapping CSI DataFrame ...")
    return (
        df.explode("used_antenna_idxs")
        .with_columns(
            csi_abs=pl.col("csi_abs").list.get(pl.col("used_antenna_idxs")),
            csi_phase=pl.col("csi_phase").list.get(pl.col("used_antenna_idxs")),
            antenna_rssi=pl.col("antenna_rssi").list.get(pl.col("used_antenna_idxs")),
        )
        .with_columns(
            unwrapped_phase=pl.col("csi_phase").apply(
                lambda x: np.unwrap(np.array(x)).tolist()
            ),
        )
        .with_row_count()
        .explode("subcarrier_idxs", "csi_abs", "csi_phase", "unwrapped_phase")
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
    return df.join(
        df.group_by("row_nr").agg(pl.col("csi_abs").mean().alias("voltage")),
        on="row_nr",
    ).with_columns(
        volt_normed_csi=pl.col("csi_abs") / pl.col("voltage"),
    )


def equalize_magnitude(df: pl.DataFrame) -> pl.DataFrame:
    # ----------------------------------------------------------------------------------
    # -- Profile correction to flatten CSI magnitudes
    # ----------------------------------------------------------------------------------
    # First calculate profile per session by averaging CSI amplitudes across sessions in
    # subcarriers separately. Then join back to unwrapped dataframe to allow calculation
    # of shape normed CSI by per-value division.
    # Afterwards, filter out divisions by zero (invalid subcarriers) and regroup.
    logger.trace(
        "Equalizing CSI magnitude by calculating per-session magnitude profile "
        + "and dividing by that."
    )
    return (
        df.filter(pl.col("mask_name") == TRIVIAL_MASK_NAME)
        .group_by("session_id", "receiver_name", "subcarrier_idxs", maintain_order=True)
        .agg(pl.col("volt_normed_csi").mean().alias("session_abs_profile"))
        .join(df, on=["session_id", "receiver_name", "subcarrier_idxs"])
        .with_columns(
            shape_normed_csi_abs=pl.col("volt_normed_csi")
            / pl.col("session_abs_profile")
        )
        .filter(pl.col("shape_normed_csi_abs").is_not_nan())
    )


### Need this
def detrend_phase(df: pl.DataFrame) -> pl.DataFrame:
    # ----------------------------------------------------------------------------------
    # -- Linear phase correction to get rid of STO effect
    # ----------------------------------------------------------------------------------
    # NOTE: This part would be cleaner if done above. However, we are filtering out
    # invalid subcarriers only at the end of magnitude processing, and this influences
    # phase normalization.
    logger.trace(
        "Detrending phase by fitting line through first and last phase values "
        + "and subtracting that."
    )

    # Columns to use for linear phase normalization
    phase_helper_cols = {
        "phase_slope": (
            pl.col("unwrapped_phase").list.last()
            - pl.col("unwrapped_phase").list.first()
        ) / pl.col("subcarrier_idxs").list.lengths(),
        "lowest_sc": pl.col("subcarrier_idxs").list.first(),
        "sc_offset": pl.col("unwrapped_phase").list.first(),
    }

    # Aggregate phases and calculate helper columns
    phase_helpers = (
        df.group_by("row_nr", maintain_order=True)
        .agg("unwrapped_phase", "subcarrier_idxs")
        .select("row_nr", **phase_helper_cols)
    )

    # To correct linear trend: Shift down by first value, then remove linear slope
    # in dependence of subcarrier number
    lin_corr_expr = (
        pl.col("unwrapped_phase")
        - pl.col("sc_offset")
        - ((pl.col("subcarrier_idxs") - pl.col("lowest_sc")) * pl.col("phase_slope"))
    )

    # Attach helper columns and calculate linear-corrected CSI phases
    return (
        df.join(phase_helpers, on="row_nr")
        .with_columns(normed_csi_phase=lin_corr_expr)
        .drop(phase_helper_cols.keys())
    )


def equalize_phase(df: pl.DataFrame) -> pl.DataFrame:
    # ----------------------------------------------------------------------------------
    # -- Profile correction of phase
    # ----------------------------------------------------------------------------------
    logger.trace(
        "Equalizing phases by calculating session profile and "
        + "subtracting it from all CSI captures."
    )
    return (
        df.filter(pl.col("mask_name") == TRIVIAL_MASK_NAME)
        .group_by("session_id", "receiver_name", "subcarrier_idxs", maintain_order=True)
        .agg(
            pl.col("normed_csi_phase").mean().alias("session_phase_profile"),
            pl.col("normed_csi_phase").std().alias("session_phase_std"),
        )
        .join(df, on=["session_id", "receiver_name", "subcarrier_idxs"])
        .with_columns(
            shape_normed_csi_phase=pl.col("normed_csi_phase")
            - pl.col("session_phase_profile")
        )
    )


def remove_phase_outliers(df: pl.DataFrame, n: int = 3) -> pl.DataFrame:
    # Means are already calculated as the session phase profile
    # Calculate per-subcarrier deviations, drop Nans (expected at edges),
    # then filter out those that have high average deviation. These are
    # outliers.
    outliers = (
        df.with_columns(
            devs=(
                (pl.col("normed_csi_phase") - pl.col("session_phase_profile"))
                / pl.col("session_phase_std")
            ).abs()
        )
        .filter(pl.col("devs").is_not_nan())
        .group_by("row_nr", maintain_order=True)
        .agg(pl.col("devs").mean())
        .filter(pl.col("devs") > n)
    )

    logger.trace(
        f"Detected {len(outliers)} outlier captures. Dropping them from the data ..."
    )
    return df.join(outliers, on="row_nr", how="anti").drop("session_phase_std")


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

    df = remove_phase_outliers(df)
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
