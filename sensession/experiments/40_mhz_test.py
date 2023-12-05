import numpy as np

from sensession.lib.config import (
    Channel,
    Sideband,
    SessionConfig,
    BaseFrameConfig,
    ExperimentConfig,
    FrameGroupConfig,
)
from sensession.lib.frame_generation import create_fg_config
from sensession.experiments.exp_fixture import run_exp, exp_config


def get_mask_group() -> np.ndarray:
    return np.ones((128, 1), dtype=np.complex64)


def get_qca_frame_group() -> FrameGroupConfig:
    # fmt: off
    base_frame = BaseFrameConfig(
        index               = 1,                    # QCA requires special frames, this is only for the QCA card
        receiver_address    = "90:48:9a:b6:1f:39",  # Who this frame is addressed to
        transmitter_address = "24:4b:fe:bc:a6:fc",  # Who this frame came from
        bssid_address       = "24:4b:fe:bc:a6:fc",  # bssid address, i.e. access point (equal transmitter in monitor mode)
        bandwidth           = int(40e6),            # Channel bandwidth
        send_rate           = int(50e6),            # Upsample to 50 MS/s
        enable_sounding     = True,                 # Whether to force sounding bit in PHY preamble
        rescale_factor      = 25000,                # Scale factor (int16) of file output sample values
    )
    # fmt: on

    mask_group = get_mask_group()

    return create_fg_config(
        base_frame=base_frame,
        mask_group=mask_group,
        group_repetitions=0,
        mask_name="unmodified",
        interframe_delay=200000,  # Number of inter-frame zero-padding samples
    )


def chan40mhz_qca_session():
    channel = Channel(
        frequency=5_795_000_000,  # 2_447_000_000,
        channel_number=157,  # 6,
        channel_spec="HT40+",  # NOTE: the "+" means the always 20MHz beacons are transmitted in the right half; irrelevant for us in monitor mode.
        bandwidth_mhz=40,
        sideband=Sideband.NONE,  # Sideband.UPPER,
    )

    # Build frame group for remote qca
    qca_frame_group = get_qca_frame_group()
    return SessionConfig(
        name="qca_40mhz_session",
        frame_group=qca_frame_group,
        channel=channel,
        receivers=["asus"],
        transmitter=["usrp x310"],
        tx_gain=5,
        training_reps=1000,
    )


def chan40mhz_exp_cfg() -> ExperimentConfig:
    qca_session = chan40mhz_qca_session()
    return exp_config(exp_id="40mhz_test", sessions=[qca_session])


if __name__ == "__main__":
    run_exp(experiment_cfg=chan40mhz_exp_cfg())
