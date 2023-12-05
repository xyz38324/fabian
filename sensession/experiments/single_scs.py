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


def singlesc_mask(sc_idx: int, scale: float) -> np.ndarray:
    base_mask = np.ones((64, 1), dtype=np.complex64)
    base_mask[sc_idx] *= scale
    return base_mask


def singlesc_iwl_session(sc_idx, scale):
    mask = singlesc_mask(sc_idx, scale)

    frame_group = get_iwl_frame_group(
        mask=mask,
        mask_name=f"idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        group_reps=1000,
        rescale_factor=25000,
    )

    return get_iwl_session(
        session_name=f"iwl_idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        frame_group=frame_group,
    )


def singlesc_qca_session(sc_idx, scale):
    mask = singlesc_mask(sc_idx, scale)

    frame_group = get_qca_frame_group(
        mask=mask,
        mask_name=f"idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        group_reps=1000,
        rescale_factor=25000,
    )

    return get_qca_session(
        session_name=f"qca_idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        frame_group=frame_group,
    )


def singlesc_exp_config() -> ExperimentConfig:
    # Define parameters to construct precoding masks and construct sessions
    idxs = list(range(64))
    factors = [0, 0.5, 0.75, 1.25, 2, 3, 4.5]

    iwl_sessions = [
        singlesc_iwl_session(sc_idx, scale)
        for sc_idx, scale in itertools.product(idxs, factors)
    ]

    qca_sessions = [
        singlesc_qca_session(sc_idx, scale)
        for sc_idx, scale in itertools.product(idxs, factors)
    ]

    sessions = iwl_sessions + qca_sessions

    return exp_config(exp_id="single_sc_precoding", sessions=sessions)


if __name__ == "__main__":
    run_exp(experiment_cfg=singlesc_exp_config())
