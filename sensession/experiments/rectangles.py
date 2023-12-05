import itertools

import numpy as np

from sensession.lib.config import ExperimentConfig
from sensession.experiments.exp_fixture import (
    run_exp,
    exp_config,
    get_iwl_session,
    get_qca_session,
    get_iwl_frame_group,
    get_qca_frame_group,
)


def rectangle_mask(sc_idx: int, scale: float) -> np.ndarray:
    base_mask = np.ones((64, 1), dtype=np.complex64)

    # Rectangular precoding for a total of 7 subcarriers
    for k in range(-3, 4):
        base_mask[sc_idx + k] *= scale

    return base_mask


def rectangles_iwl_session(sc_idx, scale):
    mask = rectangle_mask(sc_idx, scale)

    frame_group = get_iwl_frame_group(
        mask=mask,
        mask_name=f"idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        group_reps=1000,
        rescale_factor=12000,
    )

    return get_iwl_session(
        session_name=f"iwl_idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        frame_group=frame_group,
    )


def rectangles_qca_session(sc_idx, scale):
    mask = rectangle_mask(sc_idx, scale)

    frame_group = get_qca_frame_group(
        mask=mask,
        mask_name=f"idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        group_reps=1000,
        rescale_factor=12000,
    )

    return get_qca_session(
        session_name=f"qca_idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        frame_group=frame_group,
    )


def rectangles_exp_config() -> ExperimentConfig:
    # Define parameters to construct precoding masks and construct sessions
    idxs = [
        4,  # 1: First  data subcarrier
        5,  # Mirror of 8
        6,  # 2: ax210 strange edge effect
        13,  # 3: ax210 skewed adjacency influence
        16,  # 4: just a "normal" subcarrier
        24,  # 5: qca weird improved detection performance
        25,  # 6: Pilot
        31,  # 7: Right next do DC zero subcarrier
        33,  # Mirror of 7
        39,  # Mirror of 6
        40,  # Mirror of 5
        48,  # Mirror of 4
        51,  # Mirror of 3
        58,  # Mirror of 2
        59,  # 8: Edge subcarrier for asus (because last edge sc is broken, nexmon issue?)
        60,  # Mirror of 1
    ]
    factors = [0, 0.5, 0.75, 1.25, 2, 3, 4.5]

    iwl_sessions = [
        rectangles_iwl_session(sc_idx, scale)
        for sc_idx, scale in itertools.product(idxs, factors)
    ]

    qca_sessions = [
        rectangles_qca_session(sc_idx, scale)
        for sc_idx, scale in itertools.product(idxs, factors)
    ]

    sessions = iwl_sessions + qca_sessions

    return exp_config(exp_id="rectangle_precoding", sessions=sessions)


if __name__ == "__main__":
    run_exp(experiment_cfg=rectangles_exp_config())
