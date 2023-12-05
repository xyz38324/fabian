"""
Database implementation

We use polars DataFrames as "database", persisting them into parquet files
"""

import json
from typing import List
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
from loguru import logger
from polars.type_aliases import EpochTimeUnit

DEFAULT_TIMESTAMP_UNIT: EpochTimeUnit = "ns"


# -------------------------------------------------------------------------------------
# -- Database Schema Definition
# -----------------------------
# Our database is, currently, just a simple parquet file managed with polars.
# The following section describes the schema to be used and adhered to within
# that database.
# -------------------------------------------------------------------------------------
def get_csi_schema(timestamp_unit: EpochTimeUnit = DEFAULT_TIMESTAMP_UNIT):
    """
    Captured Data Schema.

    NOTE: We expect CSI to be reported for every subcarrier.
    CSI Parsers should set values to zero where no CSI is reported.

    CSI from multiple antennas is laid out flat in the CSI lists.
    The `used_antenna_idxs` may be used to infer to which antenna CSI belongs

    Args:
        timestamp_unit : Timestamp (integer) unit
    """
    # fmt: off
    return {
        "timestamp" :        pl.Datetime(timestamp_unit),              # Timestamp (may be local to capture device)
        "sequence_number":   pl.UInt16,                                # Sequence number of frame from which CSI was extracted
        "subcarrier_idxs":   pl.List(inner=pl.Int16),                  # Subcarrier idx to which list index i corresponds
        "csi_abs":           pl.List(inner=pl.List(inner=pl.Float64)), # Absolute values of CSI symbols (one List per Antenna)
        "csi_phase":         pl.List(inner=pl.List(inner=pl.Float64)), # Phase values of CSI symbols    (one List per Antenna)
        "rssi":              pl.Int8,                                  # Reported Received Signal Strength
        "antenna_rssi":      pl.List(inner=pl.Int8),                   # Per-antenna Signal strength (for antennas specified in idxs)
        "used_antenna_idxs": pl.List(inner=pl.UInt8),                  # idxs of antennas used to capture
    }
    # fmt: on


def get_labels_schema():
    """
    Get DataFrame label schema. Labels map collected data onto a specific sensing
    session performed within an experiment.
    """
    # fmt: off
    return {
        "experiment_id":          pl.Utf8,
        "experiment_start_time" : pl.Datetime,
        "session_id":             pl.Utf8,
        "session_name":           pl.Utf8,
        "frame_id":               pl.Utf8,
        "mask_id":                pl.Utf8,
        "mask_name" :             pl.Utf8,
        "receiver_name":          pl.Utf8,
        "channel":                pl.UInt16,
        "bandwidth":              pl.UInt8,
        "tx_gain":                pl.UInt8,
        "session_timestamp":      pl.Datetime,
    }
    # fmt: on


def get_full_schema():
    """
    Get full schema of DataFrame-based database

    NOTE: There is a lot of duplication of the labels for the CSI values. This
    is not necessary and we could instead use a single index pointing into a
    second table with deduplicated labels.

    If space becomes an issue, we can migrate the database accordingly as optimization.
    """
    return get_csi_schema().update(get_labels_schema())


# -------------------------------------------------------------------------------------
# -- Database Conversion Helpers
# ------------------------------
# Some helper functions to create suitable DataFrames for the database.
# -------------------------------------------------------------------------------------


def csi_to_dataframe(
    timestamps: List[np.uint64],
    sequence_nums: List[np.uint8],
    subcarrier_idxs: List[List[np.int16]],
    csi_vals: List[List[np.ndarray]],
    rssi: List[np.int8],
    antenna_rssi: List[List[np.int8]],
    antenna_idxs: List[List[np.uint8]],
    timestamp_unit: EpochTimeUnit = DEFAULT_TIMESTAMP_UNIT,
) -> pl.DataFrame:
    """
    Create a DataFrame from list of CSI and accompanying values.

    Args:
        csi_vals       : List of complex-valued CSI arrays
        sequence_nums  : List of sequence nums of corresponding frames
        rssi           : List of rssi values for each captured CSI frame
        antenna_rssi   : List of per-antenna RSSI values for each captured frame
        timestamps     : List of capture timestamps
        antenna_idxs   : List of antenna indices used to capture respective frames
        timestamp_unit : Unit of timestamps (defaults to microseconds)
    """
    # fmt: off
    df = pl.DataFrame(
        {
            "timestamp" :         timestamps,
            "sequence_number":    sequence_nums,
            "subcarrier_idxs":    subcarrier_idxs,
            "csi_abs":            [np.abs(csi).tolist() for csi in csi_vals],
            "csi_phase":          [np.angle(csi).tolist() for csi in csi_vals],
            "rssi":               rssi,
            "antenna_rssi":       antenna_rssi,
            "used_antenna_idxs" : antenna_idxs,
        },
        schema=get_csi_schema(timestamp_unit),
    )
    # fmt: on

    # Cast to common time unit of nanoseconds
    df = df.with_columns(pl.col("timestamp").dt.cast_time_unit(DEFAULT_TIMESTAMP_UNIT))

    return df


def attach_label_data(
    df: pl.DataFrame,
    frame_id: str,
    mask_id: str,
    session_id: str,
    session_name: str,
    experiment_id: str,
    experiment_time: datetime,
    mask_name: str,
    receiver_name: str,
    channel: int,
    bandwidth: int,
    tx_gain: int,
    session_timestamp: datetime,
) -> pl.DataFrame:
    """
    Attach label data to existing dataframe.
    Assumes existing dataframe contains csi schema (`get_csi_schema`).
    """
    columns = {
        "frame_id": frame_id,
        "mask_id": mask_id,
        "experiment_id": experiment_id,
        "experiment_start_time": experiment_time,
        "session_id": session_id,
        "session_name": session_name,
        "mask_name": mask_name,
        "receiver_name": receiver_name,
        "channel": channel,
        "bandwidth": bandwidth,
        "tx_gain": tx_gain,
        "session_timestamp": session_timestamp,
    }
    schema = get_labels_schema()
    pl_columns = {col: pl.lit(val, dtype=schema[col]) for col, val in columns.items()}
    return df.with_columns(**pl_columns)


# -------------------------------------------------------------------------------------
# -- Database Wrapper Class
# -------------------------
# The following section contains a database wrapper that handles persistence and
# manages data over time by appending to present data.
# NOTE: If databases become too large, this part could handle a database backend
# or at least work with LazyFrames.
# -------------------------------------------------------------------------------------
class Database:
    """
    Database wrapper around DataFrame to automatically append and store data
    """

    def __init__(self, filepath: Path = Path.cwd() / "data" / "db.parquet"):
        self.filepath = filepath
        self.filepath.parent.mkdir(exist_ok=True, parents=True)

        if filepath.is_file():
            self.df: pl.DataFrame = pl.read_parquet(filepath)
            logger.debug(
                "Database/Dataframe loaded -- Recovered previous data from"
                + f" {filepath}. "
            )
            logger.trace(f"Dataframe schema: {self.df.schema}")
        else:
            self.df = pl.DataFrame(schema=get_full_schema())

        self.changed = False

    def __del__(self):
        """
        Destroy database object - persists values to database file
        """
        if self.changed:
            if not self.df.is_empty():
                self.df.write_parquet(self.filepath)
            else:
                logger.debug("DataFrame is empty; skipping storage")

    def add_data(self, df: pl.DataFrame):
        """
        Add data to the database

        Args:
            df : DataFrame of data to append
        """
        if df.is_empty():
            logger.debug("No data to add. Ignoring ...")
            return

        logger.trace(f"Adding dataframe of shape {df.shape} to database ...")
        self.df = self.df.vstack(df)
        self.changed = True

    def remove_after(self, time: datetime):
        """
        Remove all entries in the database after given time.

        Args:
            time : Timestamp after which to remove all data
        """
        # Only keep what was before
        self.df = self.df.filter(pl.col("session_timestamp") < time)
        self.changed = True

    def remove_before(self, time: datetime):
        """
        Remove all entries in the database before given time.

        Args:
            time : Timestamp before which to remove all data
        """
        # Only keep what was after
        self.df = self.df.filter(pl.col("session_timestamp") > time)
        self.changed = True

    def get(self) -> pl.DataFrame:
        """
        Get the internal polars DataFrame
        """
        return self.df

    def describe(self):
        """
        Print a detailed description of what is currently inside the database.
        """
        print("Peek into database:")
        print(self.df)

        session_times = (
            self.df.groupby("session_timestamp", maintain_order=True)
            .agg()
            .unique()
            .sort("session_timestamp")
        )
        print(
            "==================================================================\n"
            + "====================== Database description ======================\n"
            + "Database schema    :\n"
            + f"{json.dumps(self.df.schema, indent=4, default=str)}\n"
            + "Database shape     :\n"
            + f"{self.df.shape}\n "
            + "Number of sessions:\n"
            + f"{self.df.groupby('session_id', maintain_order=True).agg().n_unique()}\n "
            + "Database heap size: "
            + f"{self.df.estimated_size()}\n"
            + "Number of CSI data :\n"
            + f"{self.df.groupby('receiver_name').count()}\n "
            + "Session times:\n"
            + f"{session_times}\n"
            + "Number of masks:\n"
            + f"{self.df.groupby(['mask_id', 'mask_name']).count()}\n"
            + "Bandwidths:\n"
            + f"{self.df.select(pl.col('bandwidth').unique())}\n "
            + "Experiment IDs:\n"
            + f" {self.df.select(pl.col('experiment_id').unique())}\n"
            + "Channels:\n"
            + f"{self.df.select(pl.col('channel').unique())}\n"
        )
