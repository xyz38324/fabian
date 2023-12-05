"""
Picoscenes file parser

File parser for log files from PicoScenes. Relies on the picoscenes python
toolbox for the initial parsing, but offers some further functionality for
further extraction and conversion to our internal data format (dataframes)
"""

from typing import List, Tuple
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from csi_tools.PicoscenesToolbox.picoscenes import Picoscenes

from sensession.lib.database import get_csi_schema, csi_to_dataframe


def get_rssi_values(frames: Picoscenes, antenna_idx, rescale=True, linear=True):
    """
    Extract RSSI values for the captures from each frame.

    Args:
        frames      : Picoscenes frame collection
        antenna_idx : Index of the antenna used to capture
        rescale     : Whether to upshift RSSI dB values to a minimum of zero
        linear      : Whether to convert RSSI dB values to linear scale (voltage)
    """
    logger.trace("Reading RSSI values from picoscenes frame ...")
    rssi_strs = ["rssi1", "rssi2", "rssi3"]
    rssi_antenna = rssi_strs[antenna_idx]
    rssi = np.array([frame.get("RxSBasic").get(rssi_antenna) for frame in frames.raw])

    # Check distribution of RSSI values
    uniq_rssi = np.unique(rssi)
    logger.trace(
        "----------------- RSSI distribution -----------------\n"
        + f"Unique RSSI values : {uniq_rssi}\n"
        + f"Histogram          : {np.histogram(rssi, bins = len(uniq_rssi))}"
    )

    # Shifts rssi values into [0, x]
    if rescale:
        rssi = rssi - np.min(rssi)

    if linear:
        rssi = 10 ** (rssi / 10)

    return rssi


def extract_timestamps(frames: Picoscenes) -> List[int]:
    """
    Extract timestamps for all captured frames

    Args:
        frames : Picoscenes frame collection
    """
    return [frame.get("RxSBasic").get("systemns") for frame in frames.raw]


def extract_rssi(frames: Picoscenes) -> List[int]:
    """
    Extract RSSI for all captured frames

    Args:
        frames : Picoscenes frame collection
    """
    return [frame.get("RxSBasic").get("rssi") for frame in frames.raw]


def extract_antenna_rssi(
    frames: Picoscenes, antenna_idxs: List[int]
) -> List[List[int]]:
    """
    Extract per-antenna RSSI for all captured frames

    Args:
        frames       : Picoscenes frame collection
        antenna_idxs : List of antennas used for capture
    """
    # Picoscenes stores antenna rssi in rssi1, rssi2, rssi3
    rssi_strs = [f"rssi{idx+1}" for idx in antenna_idxs]
    return [
        [frame.get("RxSBasic").get(key) for key in rssi_strs] for frame in frames.raw
    ]


def extract_antenna(
    data: np.ndarray, antenna_idxs: List[int], num_antennas: int, num_tones: int
) -> np.ndarray:
    """
    Extract CSI captured with relevant antenna of receiver from raw CSI data.

    Args:
        data         : Original data in (n_captures, n_subcarriers) shape
        antenna_idx  : Index of antenna used
        num_antennas : Number of available antennas
        num_tones    : Number of tones/subcarriers captured per antenna
    """
    logger.trace(
        "Slicing CSI from relevant captures antennas ...\n"
        + f"  -- num antennas     : {num_antennas}\n"
        + f"  -- capture antennas : {antenna_idxs}\n"
        + f"  -- num tones        : {num_tones}\n"
    )

    # Extract the captured CSI from the relevant antenna attached to SDR.

    # NOTE: Picoscenes reports data from multiple antennas.
    # Data is of shape: [num_csi_captures, num_antennas * num_subcarriers]
    # We reshape into : [num_csi_captures, num_antennas, num_subcarriers]
    data = data.reshape((data.shape[0], num_antennas, num_tones))

    # Now extract the antennas used for capture
    return data[:, antenna_idxs, :]


def extract_csi(
    frames: Picoscenes, antenna_idxs: List[int], num_tones: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract CSI from all captures frames. No masks were applied, so all frames are
    equal.

    Args:
        frames       : Picoscenes frame object
        antenna_idxs : List of antennas used in the capture
        num_tones    : Number of tones (usable subcarriers)
    """
    logger.trace("Extracting CSI from Picoscenes frame collection ...")

    # Extract CSI and sequence number
    csi = np.array([frame.get("CSI").get("CSI") for frame in frames.raw])
    num_antennas = frames.raw[0].get("RxSBasic").get("numRx")
    csi = extract_antenna(csi, antenna_idxs, num_antennas, num_tones)
    logger.trace(f"Extracted Picoscenes CSI values of shape: {csi.shape}")

    # Retrieve subcarrier indices
    subc_indices = get_subcarrier_idxs(frames)

    # Remove zero subcarrier from CSI values and subcarrier indices
    csi = csi[:, :, np.flatnonzero(subc_indices)]
    subc_indices = subc_indices[np.flatnonzero(subc_indices)]

    logger.trace(f"Removed suppressed subcarrier, final CSI shape: {csi.shape}")

    return subc_indices, csi


def extract_sequence_nums(frames: Picoscenes) -> List[int]:
    """
    Extract list of sequence numbers for each of the frames

    Args:
        frames : Picoscenes frames object
    """
    return [frame.get("StandardHeader").get("Sequence") for frame in frames.raw]


def get_num_tones(frames: Picoscenes) -> int:
    """
    Extract number of tones from capture

    Args:
        frames : Picoscenes frames object
    """
    return frames.raw[0].get("CSI").get("numTones")


def get_subcarrier_idxs(frames: Picoscenes) -> np.ndarray:
    """
    Extract array of subcarrier idxs tones correspond to

    Args:
        frames : Picoscenes frames object
    """
    logger.trace("Extracting subcarrier indices ...")
    return np.array(frames.raw[0].get("CSI").get("SubcarrierIndex"), dtype=np.int16)


def extract_data_dict(file: Path, antenna_idxs: List[int]) -> dict:
    """
    Extract data into dictionary form

    Args:
        file         : File written by PicoScenes with the data
        antenna_idxs : Indices of antennas used for capturing
    """
    frames = Picoscenes(str(file))

    logger.trace(f"Rx Info: {frames.raw[0].get('RxSBasic')}")

    # Extract sequence nums
    sequence_nums = extract_sequence_nums(frames=frames)

    # Extract CSI values
    num_tones = get_num_tones(frames)
    subcs, csi = extract_csi(
        frames=frames, antenna_idxs=antenna_idxs, num_tones=num_tones
    )
    num_captures = csi.shape[0]

    # Extract overall RSSI as well as per antenna ones
    rssi = extract_rssi(frames=frames)
    antenna_rssi = extract_antenna_rssi(frames=frames, antenna_idxs=antenna_idxs)

    # Extract timestamps
    timestamps = extract_timestamps(frames=frames)

    return {
        "subcarrier_idxs": [subcs.tolist()] * num_captures,
        "csi_vals": list(csi),
        "sequence_nums": sequence_nums,
        "rssi": rssi,
        "antenna_rssi": antenna_rssi,
        "timestamps": timestamps,
        "antenna_idxs": [antenna_idxs] * num_captures,
    }


def load_picoscenes_data(file: Path, antenna_idxs: List[int]) -> pl.DataFrame:
    """
    Parse picoscene log files to extract CSI data

    Args:
        file         : Path to picoscenes output file
        antenna_idxs : Indices of used antennas in the capture
    """
    # If file is empty, failsafe return empty DatFrame
    if not file.exists() or file.stat().st_size == 0:
        return pl.DataFrame({}, schema=get_csi_schema())

    data = extract_data_dict(file, antenna_idxs)
    df = csi_to_dataframe(**data, timestamp_unit="ns")
    return df
