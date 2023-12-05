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


def scaleblock_mask(blocksize: int, scale: float) -> np.ndarray:
    base_mask = np.ones((64, 1), dtype=np.complex64)
    # NOTE: Assumes 20 MHz channels, fft size 64
    base_mask[32 - blocksize // 2 : 32 + blocksize // 2 + 1] *= scale
    return base_mask


def expscaleblock_iwl_session(blocksize, scale):
    mask = scaleblock_mask(blocksize, scale)

    frame_group = get_iwl_frame_group(
        mask=mask,
        mask_name=f"blocksize_{blocksize}_absfactor_{int(scale*100)}e-2",
        group_reps=1000,
        rescale_factor=5000,  # Scaling happens before precoding; large blocks with 4.5 amplitude require small scale
    )

    return get_iwl_session(
        session_name=f"iwl_size_{blocksize}_absfactor_{int(scale*100)}e-2",
        frame_group=frame_group,
    )


def expscaleblock_qca_session(blocksize, scale):
    mask = scaleblock_mask(blocksize, scale)

    frame_group = get_qca_frame_group(
        mask=mask,
        mask_name=f"blocksize_{blocksize}_absfactor_{int(scale*100)}e-2",
        group_reps=1000,
        rescale_factor=5000,  # Scaling happens before precoding; large blocks with 4.5 amplitude require small scale
    )

    return get_qca_session(
        session_name=f"qca_size_{blocksize}_absfactor_{int(scale*100)}e-2",
        frame_group=frame_group,
    )


def expscaleblock_exp_config() -> ExperimentConfig:
    #     # Define parameters to construct precoding masks and construct sessions
    block_sizes = list(range(1, 50, 2))
    factors = [0, 0.5, 0.75, 1.25, 2, 3, 4.5]

    iwl_sessions = [
        expscaleblock_iwl_session(blocksize, scale)
        for blocksize, scale in itertools.product(block_sizes, factors)
    ]

    qca_sessions = [
        expscaleblock_qca_session(blocksize, scale)
        for blocksize, scale in itertools.product(block_sizes, factors)
    ]

    sessions = iwl_sessions + qca_sessions

    return exp_config(exp_id="expanding_scaleblock", sessions=sessions)


if __name__ == "__main__":
    run_exp(experiment_cfg=expscaleblock_exp_config())
