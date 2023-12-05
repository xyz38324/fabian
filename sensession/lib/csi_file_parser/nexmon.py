"""
Nexmon file parser

Parses tcpdump files received from Nexmon. For now, relies on the provided
Matlab script to do the actual extraction, float conversion, etc.
"""

from typing import List
from pathlib import Path
from operator import itemgetter

import numpy as np
import polars as pl
import scipy.io as sio
from loguru import logger

from sensession.lib.database import get_csi_schema, csi_to_dataframe
from sensession.lib.frame_generation import matlab_to_numpy_array
from sensession.lib.matlab.matlab_parallelizer import EngineWrapper


def read_nexmon_csi(
    engine, csi_file: str, bandwidth_mhz: int, antenna_idxs: List[int]
) -> dict:
    """
    Read nexmon data from file and return as dict

    Args:
        engine        : matlab engine to use for call to file parser
        csi_file      : Path to capture file
        bandwidth_mhz : Channel bandwidth used during capture
        antenna_idxs  : Indices of antennas used in capture
    """
    num_antennas = len(antenna_idxs)

    # Run matlab CSI extractor and store in file. Do not use the returns from
    # the matlab function here because of memory leaks. Open matlab instances
    # of engines would cause a memory overflow after a while.
    postproc_file = f"{csi_file}.postprocessed.mat"
    engine.read_csi(csi_file, "4366c0", bandwidth_mhz, postproc_file)

    # Read the temp file and extract the variables
    postproc = sio.loadmat(postproc_file)
    sequence_nums, timestamps, subcarrier_idxs, csi = itemgetter(
        "sequence_nums", "timestamps", "subcarriers", "csi"
    )(postproc)

    # Properly format the retrieved values
    timestamps = timestamps.flatten().astype(int).tolist()
    sequence_nums = sequence_nums.flatten().tolist()
    subcarrier_idxs = subcarrier_idxs.flatten().tolist()
    csi = np.expand_dims(csi, 1)

    # Sanity check CSI shape before formatting to list
    num_captures = len(sequence_nums)
    num_subcarrs = len(subcarrier_idxs)
    assert csi.shape == (
        num_captures,
        num_antennas,
        num_subcarrs,
    ), f"Wrong CSI shape, got: {csi.shape}"
    csi = csi.tolist()

    # Assemble into dict of named argument values for database/dataframe creation
    num_captured = len(sequence_nums)
    logger.trace(f"Nexmon file parser: Loaded {num_captured} data points!")
    return {
        "subcarrier_idxs": [subcarrier_idxs] * num_captured,
        "csi_vals": csi,
        "sequence_nums": sequence_nums,
        "rssi": [0] * num_captured,
        "antenna_rssi": [[0] * num_antennas] * num_captured,
        "timestamps": timestamps,
        "antenna_idxs": [antenna_idxs] * num_captured,
    }


# -------------------------------------------------------------------------------------
# Module-wide matlab instance solely used for nexmon postprocessing.
# Theoretically, we could probably write the pool used in the frame generation so
# that it would allow for a reuse here.
# For now, doing at least some caching of such an instance is already good enough.
# TODO: Get rid of Matlab here.
# Instead, use Nexmon CSI capture solution that directly unpacks floats properly.
logger.trace("Starting matlab subprocess to extract nexmon data from tmp file ...")

matlab_nexmon_path = Path.cwd() / "matlab" / "nexmon_csi"
engine = EngineWrapper(
    processing_callback=read_nexmon_csi, start_path=matlab_nexmon_path
)


def load_nexmon_data(
    file: Path, antenna_idxs: List[int], bandwidth_mhz: int
) -> pl.DataFrame:
    """
    Load nexmon data from file and parse into dataframe

    Args:
        file          : Path fo pcap capture file
        antenna_idxs  : List of used antenna indices
        bandwidth_mhz : Bandwidth used in capture, in MHz
    """
    # TODO: Support more antennas.
    # This required the csiread part to also allow for multiple antennas
    assert (
        len(antenna_idxs) == 1
    ), "Cant parse nexmon CSI with multiple antenna captures"

    # If file is empty, failsafe return empty DatFrame
    if not file.exists() or file.stat().st_size == 0:
        return pl.DataFrame({}, schema=get_csi_schema())

    data = engine.process(
        kwargs={
            "csi_file": str(file.resolve()),
            "csi_file": str(file.resolve()),
            "bandwidth_mhz": bandwidth_mhz,
            "antenna_idxs": antenna_idxs,
        }
    )

    df = csi_to_dataframe(**data, timestamp_unit="us")
    return df
