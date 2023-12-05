"""
Most of the experiments for receiver comparison have a lot of common ground.
This file contains a few common helper functions that minimize duplication
by providing some sensible defaults and thus simplify scripting experiments.

At its core, to cover all the receivers, we always need to work with two sets
of frames. The reason being that the CSI extraction using the qca and iwl5300
tools are fundamentally incompatible, hence we need different frames for both.
"""

import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="TRACE")
logger.add("experiment.log", rotation="500 MB", retention=3)


from typing import List
from pathlib import Path

import numpy as np

from sensession.lib.config import (
    Channel,
    SessionConfig,
    BaseFrameConfig,
    ExperimentConfig,
    FrameGroupConfig,
    load_hardware_setup,
)
from sensession.lib.experiment import run_experiment
from sensession.lib.frame_generation import TRIVIAL_MASK_NAME, create_fg_config


def default_mask_group() -> np.ndarray:
    return np.ones((64, 1), dtype=np.complex64)


def get_iwl_frame_group(
    mask: np.ndarray = default_mask_group(),
    mask_name: str = TRIVIAL_MASK_NAME,
    if_delay: int = 200000,
    group_reps: int = 1,
    rescale_factor: int = 30000,
) -> FrameGroupConfig:
    # fmt: off
    base_frame = BaseFrameConfig(
        index               = 2,                    # Old intel iwl5300 has the same issue, needing very specific parameters
        receiver_address    = "00:16:ea:12:34:56",  # Possibly hardcoded in card firmware; dont change!
        transmitter_address = "00:16:ea:12:34:56",  # Weird shenanigans of 5300 monitor mode; dont change!
        bssid_address       = "ff:ff:ff:ff:ff:ff",  # You guessed it: dont change for monitor mode.
        bandwidth           = int(20e6),            # Channel bandwidth
        enable_sounding     = False,
        rescale_factor      = rescale_factor,
    )
    # fmt: on

    return create_fg_config(
        base_frame=base_frame,
        mask_group=mask,
        group_repetitions=group_reps,
        mask_name=mask_name,
        interframe_delay=if_delay,
    )


def get_qca_frame_group(
    mask: np.ndarray = default_mask_group(),
    mask_name: str = TRIVIAL_MASK_NAME,
    if_delay: int = 200000,
    group_reps: int = 1,
    rescale_factor: int = 30000,
) -> FrameGroupConfig:
    # fmt: off
    base_frame = BaseFrameConfig(
        index               = 1,                    # QCA requires special frames, this is only for the QCA card
        receiver_address    = "90:48:9a:b6:1f:39",  # Who this frame is addressed to
        transmitter_address = "24:4b:fe:bc:a6:fc",  # Who this frame came from
        bssid_address       = "24:4b:fe:bc:a6:fc",  # bssid address, i.e. access point (equal transmitter in monitor mode)
        bandwidth           = int(20e6),            # Channel bandwidth
        enable_sounding     = True,                 # Whether to force sounding bit in PHY preamble
        rescale_factor      = rescale_factor,       # Scale factor (int16) of file output sample values
    )
    # fmt: on

    return create_fg_config(
        base_frame=base_frame,
        mask_group=mask,
        group_repetitions=group_reps,
        mask_name=mask_name,
        interframe_delay=if_delay,  # Number of inter-frame zero-padding samples
    )


def default_channel():
    return Channel(
        frequency=2_437_000_000,
        channel_number=6,
        channel_spec="HT20",
        bandwidth_mhz=20,
    )


def get_session(
    session_name: str,
    frame_group,
    receivers: List[str],
    channel: Channel = default_channel(),
    tx_gain: int = 5,
    training_reps: int = 500,
    tx_reps: int = 1,
) -> SessionConfig:
    # Build a basic session
    return SessionConfig(
        name=session_name,
        frame_group=frame_group,
        channel=channel,
        receivers=receivers,
        transmitter=["usrp x310"],
        tx_gain=tx_gain,
        n_repeat=tx_reps,
        training_reps=training_reps,
    )


def get_iwl_session(**kwargs) -> SessionConfig:
    return get_session(**kwargs, receivers=["asus", "asus2", "ax210", "iwl5300"])


def get_qca_session(**kwargs) -> SessionConfig:
    return get_session(**kwargs, receivers=["asus", "asus2", "ax210", "qca"])


def exp_config(
    exp_id: str,
    sessions: List[SessionConfig],
    database_path: Path = Path.cwd() / "data",
    cache_dir: Path = Path.cwd() / ".cache",
    matlab_batch_size: int = 8,
):
    return ExperimentConfig(
        exp_id=exp_id,
        database_path=database_path,
        sessions=sessions,
        cache_dir=cache_dir,
        matlab_batch_size=matlab_batch_size,
    )


def run_exp(
    experiment_cfg: ExperimentConfig, hardware_cfg_file=Path.cwd() / "setup.toml"
):
    hardware_cfg = load_hardware_setup(hardware_cfg_file)
    run_experiment(experiment_cfg, hardware_cfg)
