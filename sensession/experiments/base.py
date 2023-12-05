from sensession.lib.config import ExperimentConfig
from sensession.experiments.exp_fixture import (
    run_exp,
    exp_config,
    get_iwl_session,
    get_qca_session,
    get_iwl_frame_group,
    get_qca_frame_group,
)


def base_exp_cfg() -> ExperimentConfig:
    iwl_framegr = get_iwl_frame_group(group_reps=0)
    iwl_session = get_iwl_session(
        session_name="iwl_base", frame_group=iwl_framegr, training_reps=1000
    )

    qca_framegr = get_qca_frame_group(group_reps=0)
    qca_session = get_qca_session(
        session_name="qca_base", frame_group=qca_framegr, training_reps=1000
    )

    sessions = [iwl_session, qca_session]

    return exp_config(exp_id="base", sessions=sessions)


if __name__ == "__main__":
    run_exp(experiment_cfg=base_exp_cfg())
