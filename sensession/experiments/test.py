from sensession.lib.config import ExperimentConfig
from sensession.experiments.exp_fixture import (
    run_exp,
    exp_config,
    get_iwl_session,
    get_qca_session,
    get_iwl_frame_group,
    get_qca_frame_group,
)


def test_exp_config() -> ExperimentConfig:
    iwl_framegr = get_iwl_frame_group(group_reps=10)
    iwl_session = get_iwl_session(session_name="iwl_test", frame_group=iwl_framegr)

    qca_framegr = get_qca_frame_group(group_reps=10)
    qca_session = get_qca_session(session_name="qca_test", frame_group=qca_framegr)

    sessions = [iwl_session, qca_session]

    return exp_config(exp_id="test", sessions=sessions)


if __name__ == "__main__":
    run_exp(experiment_cfg=test_exp_config())
