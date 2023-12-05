"""
Parser for files created by ath9k driver. In the driver, we pack some information into
C structs, together with the complex-valued matrix of CSI data. Here, we use ctypes to
adhere to the C-ABI and parse those structures
"""

# pylint: disable=R0903
from ctypes import Structure, c_int, c_int8, sizeof, c_uint8, c_uint16, c_uint64
from typing import List
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl

from sensession.lib.database import get_csi_schema, csi_to_dataframe


class CsiPktStatus(Structure):
    """
    CSI Packet metadata extracted from ath9k driver
    """

    _fields_ = [
        ("timestamp", c_uint64, 64),
        ("csi_len", c_uint16, 16),
        ("channel", c_uint16, 16),
        ("phyerr", c_uint8, 8),
        ("noise", c_uint8, 8),
        ("rate", c_uint8, 8),
        ("chan_bw", c_uint8, 8),
        ("num_tones", c_uint8, 8),
        ("num_rx_antennas", c_uint8, 8),
        ("num_tx_antennas", c_uint8, 8),
        ("rssi", c_int8, 8),
        ("rssi1_ctl0", c_int8, 8),
        ("rssi1_ctl1", c_int8, 8),
        ("rssi1_ctl2", c_int8, 8),
    ]

    def as_dict(self) -> dict:
        """
        Convert to dictionary
        """
        return {
            field[0]: (
                getattr(self, field[0]).as_dict()
                if isinstance(getattr(self, field[0]), Structure)
                else getattr(self, field[0])
            )
            for field in self._fields_
        }


class CsiUserInfo(Structure):
    """
    Custom-defined header data struct in the modified ath9k driver
    """

    _fields_ = [
        ("pkt_status", CsiPktStatus),
        ("sys_tstamp", c_uint64, 64),
        ("sequence_num", c_uint16),
        ("payload_len", c_uint16),
        ("csi_frame_len", c_uint16),
    ]

    def as_dict(self) -> dict:
        """
        Convert to dictionary
        """
        return {
            field[0]: (
                getattr(self, field[0]).as_dict()
                if isinstance(getattr(self, field[0]), Structure)
                else getattr(self, field[0])
            )
            for field in self._fields_
        }


class Complex(Structure):
    """
    Complex value class, as is used in the ath9k extractor from which CSI data
    is received here.
    """

    _fields_ = [
        ("real", c_int),
        ("imag", c_int),
    ]

    def as_dict(self) -> dict:
        """
        Convert to dictionary
        """
        return {
            field[0]: (
                getattr(self, field[0]).as_dict()
                if isinstance(getattr(self, field[0]), Structure)
                else getattr(self, field[0])
            )
            for field in self._fields_
        }


class CsiArray(Structure):
    """
    The full array of CSI values. Tool always uses arrays of maximum size,
    i.e. 114 subcarriers across a 3x3 MIMO stream
    """

    _fields_ = [("csi", ((Complex * 114) * 3) * 3)]

    def as_dict(self) -> dict:
        """
        Convert to dictionary
        """
        return {"csi": np.array(getattr(self, "csi"))}


class FileElement(Structure):
    """
    Wrapper class for CSI header and the actual CSI data. As the name suggests,
    this is what is stored in the file that we want to parse
    """

    _fields_ = [
        ("user_info", CsiUserInfo),
        ("csi", ((Complex * 114) * 3) * 3),
    ]

    def as_dict(self) -> dict:
        """
        Convert to dictionary
        """
        return {
            field[0]: (
                getattr(self, field[0]).as_dict()
                if isinstance(getattr(self, field[0]), Structure)
                else getattr(self, field[0])
            )
            for field in self._fields_
        }


def to_complex(x):
    """
    Convert tuple of real-/imaginary values to complex value
    """
    return x[0] + 1.0j * x[1]


# define vectorized complex value conversion function
to_complex_v = np.vectorize(to_complex)


def load_ath9k_data(file: Path, antenna_idxs: List[int]) -> pl.DataFrame:
    """
    Parsing function to read out data from ath9k driver tool

    Args:
        file : Path to ath9k CSI driver logfile
        antenna_idxs : List of antennas used
    """
    data = defaultdict(list)

    # If file is empty, failsafe return empty DatFrame
    if not file.exists() or file.stat().st_size == 0:
        return pl.DataFrame({}, schema=get_csi_schema())

    rssi_antenna_keys = [f"rssi1_ctl{idx}" for idx in antenna_idxs]
    num_tones = 0
    with open(file, "rb") as f:
        # Parse from binary file into struct contents
        x = FileElement()

        while f.readinto(x) == sizeof(x):
            # Extract CSI
            num_tones = x.user_info.pkt_status.num_tones
            csi = np.array(x.csi)[antenna_idxs, 0, 0:num_tones]
            csi = to_complex_v(csi)

            # Aggregate data into lists according to csi_to_dataframe spec
            data["timestamps"].append(x.user_info.sys_tstamp)
            data["sequence_nums"].append(x.user_info.sequence_num)
            data["csi_vals"].append(csi)
            data["rssi"].append(x.user_info.pkt_status.rssi)
            data["antenna_rssi"].append([
                getattr(x.user_info.pkt_status, rssi_antenna_key)
                for rssi_antenna_key in rssi_antenna_keys
            ])
            data["antenna_idxs"].append(antenna_idxs)

    num_captures = len(data["timestamps"])

    # In case file parsing somehow failed, no data.
    if num_captures == 0:
        return pl.DataFrame({}, schema=get_csi_schema())

    if num_tones == 56:
        subc_idxs = np.concatenate((np.arange(-28, 0), np.arange(1, 29))).tolist()
    elif num_tones == 114:
        subc_idxs = np.arange(-64, 64)
        null_subcs = [-64, -63, -62, -61, -60, -59, -1, 0, 1, 59, 60, 61, 62, 63]
        subc_idxs = list(set(subc_idxs) - set(null_subcs))
    else:
        raise NotImplementedError(
            "Only 56 and 114 tones (20/40 MHz bandwidth) are supported for Ath9k CSI"
            + " Tool."
        )

    data["subcarrier_idxs"] = [subc_idxs] * num_captures

    # Convert aggregated data to dataframe, done!
    df = csi_to_dataframe(**data, timestamp_unit="ns")
    return df
