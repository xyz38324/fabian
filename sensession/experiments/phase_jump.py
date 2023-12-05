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


def phasejump_mask(sc_idx: int, scale: float) -> np.ndarray:
    base_mask = np.ones((64, 1), dtype=np.complex64)
    base_mask[sc_idx] *= scale
    return base_mask


def phasejump_iwl_session(sc_idx, scale):
    mask = phasejump_mask(sc_idx, scale)

    frame_group = get_iwl_frame_group(
        mask=mask,
        mask_name=f"idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        group_reps=1000,
    )

    return get_iwl_session(
        session_name=f"iwl_idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        frame_group=frame_group,
    )


def phasejump_qca_session(sc_idx, scale):
    mask = phasejump_mask(sc_idx, scale)

    frame_group = get_qca_frame_group(
        mask=mask,
        mask_name=f"idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        group_reps=1000,
    )

    return get_qca_session(
        session_name=f"qca_idx_{sc_idx}_absfactor_{int(scale*100)}e-2",
        frame_group=frame_group,
    )


def phasejump_exp_config() -> ExperimentConfig:
    # Define parameters to construct precoding masks and construct sessions
    idxs = list(range(64))
    factors = [0.1, 0.2, 0.3, 0.4, 0.5]

    iwl_sessions = [
        phasejump_iwl_session(sc_idx, scale)
        for sc_idx, scale in itertools.product(idxs, factors)
    ]

    qca_sessions = [
        phasejump_qca_session(sc_idx, scale)
        for sc_idx, scale in itertools.product(idxs, factors)
    ]

    sessions = iwl_sessions + qca_sessions

    return exp_config(exp_id="phase_jump", sessions=sessions)


if __name__ == "__main__":
    run_exp(experiment_cfg=phasejump_exp_config())
