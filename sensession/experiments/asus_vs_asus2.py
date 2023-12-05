from sensession.lib.config import ExperimentConfig
from sensession.experiments.exp_fixture import (
    run_exp,
    exp_config,
    get_session,
    get_iwl_frame_group,
)


def asuscomp_exp() -> ExperimentConfig:
    framegr = get_iwl_frame_group(group_reps=0)

    session = get_session(
        session_name="ac86u_comparison",
        frame_group=framegr,
        receivers=["asus", "asus2"],
        training_reps=10000,
    )

    return exp_config(exp_id="ac86u_compare", sessions=[session])


if __name__ == "__main__":
    run_exp(experiment_cfg=asuscomp_exp())
