from typing import List
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
from loguru import logger

from sensession.lib.config import SessionConfig, ExperimentConfig
from sensession.experiments.exp_fixture import (
    run_exp,
    exp_config,
    get_iwl_session,
    get_qca_session,
    get_iwl_frame_group,
    get_qca_frame_group,
)

# Just to in-/exclude some manual stiff like an example plot of the aril data
debugmode = False


##############################################################################
# ARIL stuff
# Data loading, preprocessing, etc.
##############################################################################

# --------------------------------------------------------------
# Map of labels to their corresponding activities
label_map = {
    0: "hand up",
    1: "hand down",
    2: "hand left",
    3: "hand right",
    4: "hand circle",
    5: "hand cross",
}

# Location labels range from 0 to 15:
#        +----+----+----+----+
#        | 12 | 13 | 14 | 15 |
#        ---------------------
#        | 8  | 9  | 10 | 11 |
# [TX]   ---------------------   [Rx]
#        | 4  | 5  | 6  | 7  |
#        ---------------------
#        | 0  | 1  | 2  | 3  |
#        +----+----+----+----+
#
# Distance between all positions in each direction is 80cm.
# --------------------------------------------------------------


def example_plot(df):
    num_capture = 0
    test = df.filter((pl.col("num_capture") == num_capture))

    activity_idx = test.select(pl.first("activity")).item(0, 0)
    label = label_map[activity_idx]

    logger.trace(f"Test dataframe shape: {test.shape}")

    plt.figure(figsize=(18, 10))
    sns.set_theme(font_scale=2)
    sns.lineplot(
        data=test,
        x="time_idx",
        y="csi_amplitude",
        hue="subcarrier_idx",
    ).set(title=f"Capture {num_capture}, activity: {label}")
    logger.trace(f"Test figure drawn!")

    plt.show()


def get_aril_data() -> pl.DataFrame:
    data_dir = Path.cwd() / "src" / "sensession" / "experiments" / "aril-data"

    assert data_dir.exists(), (
        "Aril data not present, "
        + f"please download it from their github and unpack into {data_dir}"
    )

    # Read and extract training data
    data_amp = sio.loadmat(
        "src/sensession/experiments/aril-data/train_data_split_amp.mat"
    )
    data_phs = sio.loadmat(
        "src/sensession/experiments/aril-data/train_data_split_pha.mat"
    )
    train_data_amp = data_amp["train_data"]
    label_acti_amp = data_amp["train_activity_label"]
    label_posi_amp = data_amp["train_location_label"]

    train_data_phs = data_phs["train_data"]
    label_acti_phs = data_phs["train_activity_label"]
    label_posi_phs = data_phs["train_location_label"]

    assert np.all(
        label_acti_amp == label_acti_phs
    ), "Labels of amplitudes and phases should just be duplicated"
    assert np.all(
        label_posi_amp == label_posi_phs
    ), "Labels of amplitudes and phases should just be duplicated"

    logger.trace(
        "Loaded training data. \n"
        + f" -- Amplitude data shape : {train_data_amp.shape}\n"
        + f" -- Phase data shape     : {train_data_amp.shape}\n"
    )

    # Normalize the train data amplitudes. We need this to encode the
    # variations in an amplitude mask where the maximum is 1.
    train_data_amp = train_data_amp / np.max(np.abs(train_data_amp))

    # --------------------------------------------------------------
    # Build dataframe.
    # Start by collecting all relevant data
    num_captures = train_data_amp.shape[0]
    subcarriers = list(range(-26, 26))
    times = list(range(0, 192))
    data = {
        "csi_amplitude": train_data_amp.tolist(),
        "csi_phase": train_data_phs.tolist(),
        "activity": label_acti_amp.reshape(-1).tolist(),
        "position": label_posi_amp.reshape(-1).tolist(),
        "subcarrier_idx": [subcarriers] * num_captures,
        "time_idx": [times] * num_captures,
    }

    # Build the dataframe with all its nested data. Each row corresponds
    # to one capture, i.e. an array of shape (n_subcarrier, n_timesteps)
    df = pl.DataFrame(
        data,
        schema={
            "csi_amplitude": pl.List(inner=pl.List(inner=pl.Float64)),
            "csi_phase": pl.List(inner=pl.List(inner=pl.Float64)),
            "activity": pl.UInt8,
            "position": pl.UInt8,
            "subcarrier_idx": pl.List(inner=pl.Int8),
            "time_idx": pl.List(pl.UInt8),
        },
    ).with_row_count("num_capture")

    if debugmode:
        # Now explode to have everything on separate rows and allow processing
        df = df.explode("csi_amplitude", "csi_phase", "subcarrier_idx").explode(
            "csi_amplitude", "csi_phase", "time_idx"
        )

        # Just to see some visuals that should look similar to the paper
        example_plot(df)

    return df


##############################################################################
# Experiment stuff
# Generating masks from aril data etc.
##############################################################################
def extract_mask(aril_df: pl.DataFrame, run_idx: int) -> np.ndarray:
    test_amps = np.stack(aril_df[run_idx, "csi_amplitude"].to_numpy())
    test_phs = np.stack(aril_df[run_idx, "csi_phase"].to_numpy())

    # The captures in aril are used as precoding here.
    mask = test_amps * np.exp(1j * test_phs)

    # Aril captured 192 samples. Sending back-to-back, thats a fraction
    # of a second. Hence, we send an upsampled version to account for
    # some losses of packets
    mask = signal.resample_poly(mask, up=4, down=1, padtype="line", axis=1)

    # Aril performed some weird amplitude magic across subcarriers to achieve
    # this splitting. We dont want that, so we just create the mask from the
    # maximum value.
    envelope = np.max(mask, axis=0)
    mask = np.repeat(envelope[np.newaxis, :], repeats=64, axis=0)
    return mask


def aril_sessions(aril_df: pl.DataFrame) -> List[SessionConfig]:
    sessions = []

    for run_idx in range(20):
        act = aril_df[run_idx, "activity"]
        pos = aril_df[run_idx, "position"]
        mask = extract_mask(aril_df, run_idx)
        mask_name = f"activity_{act}_position_{pos}"

        sessions.append(
            get_iwl_session(
                session_name=f"iwl_run_{run_idx}_{mask_name}",
                frame_group=get_iwl_frame_group(mask=mask, mask_name=mask_name),
                training_reps=500,
                tx_gain=25,
            )
        )

        sessions.append(
            get_qca_session(
                session_name=f"qca_run_{run_idx}_{mask_name}",
                frame_group=get_qca_frame_group(mask=mask, mask_name=mask_name),
                training_reps=500,
                tx_gain=25,
            )
        )

    return sessions


def aril_exp_config() -> ExperimentConfig:
    # Define parameters to construct precoding masks and construct sessions

    aril_df = get_aril_data()
    sessions = aril_sessions(aril_df)

    return exp_config(exp_id="aril_emulation", sessions=sessions, matlab_batch_size=4)


if __name__ == "__main__":
    run_exp(experiment_cfg=aril_exp_config())
