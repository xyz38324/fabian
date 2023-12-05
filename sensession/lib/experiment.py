"""
Experiment management implementation

An experiment consists of possible multiple sessions. Here we also maintain things
like persisting experiments from failed sessions for rerunning later, etc.
"""

import gc
import time
import pickle
import shelve
import hashlib
import traceback
from typing import List, Tuple
from pathlib import Path
from datetime import datetime

from loguru import logger

from sensession.lib.config import SessionConfig, ExperimentConfig, EnvironmentConfig
from sensession.lib.session import (
    Session,
    SessionResult,
    SessionGenerator,
    ReadySessionConfig,
)
from sensession.lib.database import Database


# ---------------------------------------------------------------------------------
# Handling of experiment caching
# We store:
# - Failed experiments to allow for quick rerun with the exact same session settings
#   that have failed
# ---------------------------------------------------------------------------------
def load_experiment_list(file: Path) -> List[ExperimentConfig]:
    """
    Load list of experiments from pickle

    Args:
        file: Pickle containing list of experiments
    """
    if not file.is_file():
        return []

    with open(file, "rb") as f:
        explist = pickle.load(f)
        assert isinstance(
            explist, list
        ), f"Pickle file {file} does not contain a list!!"
        return explist


def persist_experiment(config: ExperimentConfig, file: Path):
    """
    Store (append) experiment to a list on disk.

    Args:
        config : Experiment config to store
        file   : Pickle file to store/append experiment to
    """
    previous_exps: List[ExperimentConfig] = load_experiment_list(file)

    logger.trace(
        "Persisting experiment configuration: \n"
        + f" -- Previously cached experiments: {len(previous_exps)} \n"
        + f" -- Num sessions in new experiment: {len(config.sessions)} \n"
        + f" -- Cache file : {file}\n"
    )
    previous_exps.append(config)

    with open(file, "wb") as f:
        pickle.dump(previous_exps, f)


def consume_failed_exps(file: Path) -> List[ExperimentConfig]:
    """
    Consume experiments from a pickle file.
    Consuming means this will remove all pickled elements from the file.
    To further persist them, you need to call `persist_experiment` again.

    Args:
        file : Path to pickle containing list of experiments
    """
    if not file.is_file():
        raise RuntimeError(
            f"Failed sessions can only be rerun if they are cached, but {file} is"
            + " empty."
        )

    sessions = load_experiment_list(file)

    # Aptly named, we consume the file. What was cached before should now
    # be lost and has to be cached again, if desired!
    file.unlink()
    logger.trace(
        f"Consumed previously failed {len(sessions)} experiments from {file} to rerun"
    )
    return sessions


def cache_experiment(
    config: ExperimentConfig, sessions: List[SessionConfig], file: Path
):
    """
    Create a new experiment and persist it. The new experiment will adopt generic
    settings from config, but use only the specified sessions.

    Args:
        config   : Experiment config
        sessions : List of sessions to overwrite on config
        file     : File to save experiment to
    """
    experiment = ExperimentConfig(
        exp_id=config.exp_id,
        database_path=config.database_path,
        sessions=sessions,
        cache_dir=config.cache_dir,
        matlab_batch_size=config.matlab_batch_size,
        cache_trim=config.cache_trim,
    )
    persist_experiment(config=experiment, file=file)


# ---------------------------------------------------------------------------------
# Handling of session caching
# We store completed sessions in a dictionary for quick access in processing scripts
# ---------------------------------------------------------------------------------
def persist_sessions(sessions: List[SessionConfig], file: Path):
    """
    Persist a list of sessions as a shelve. Keys are session ids.

    Args:
        sessions : List of sessions to put into shelve
        file     : Shelve file to persist sessions in
    """
    logger.trace(f"Persisting {len(sessions)} sessions in {file}")
    with shelve.open(str(file)) as db:
        for session in sessions:
            if session.session_id in db:
                raise RuntimeError(
                    f"Fatal: Session id {session.session_id} already present in db."
                    + "Hash collision?"
                )
            db[session.session_id] = session


# ---------------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------------
def sanity_check(config: ExperimentConfig, env: EnvironmentConfig):
    """
    Perform a few sanity checks on the config
    """
    logger.trace("Sanity checking experiment config ...")
    session_keys = set(s.name for s in config.sessions)
    if len(config.sessions) != len(session_keys):
        raise KeyError(
            f"Session keys must be unique, but found duplicates. Keys: {session_keys}"
        )

    session_ids = set(s.session_id for s in config.sessions)
    if len(config.sessions) != len(session_ids):
        raise KeyError(
            f"Session IDs must be unique, but found duplicates! IDs: {session_ids}"
        )

    receiver_names = set(r.short_name for r in env.receivers)
    if len(env.receivers) != len(receiver_names):
        raise KeyError(
            "Receiver short names must be unique, but found duplicates! "
            + f"Names: {receiver_names}"
        )

    transmitter_names = set(r.name for r in env.transmitter)
    if len(env.transmitter) != len(transmitter_names):
        raise KeyError(
            "Transmitter names must be unique, but found duplicates! "
            + f"Names: {transmitter_names}"
        )

    for sess_cfg in config.sessions:
        mask = sess_cfg.frame_group.mask_group
        num_scs, _ = mask.shape
        bandwidth = sess_cfg.frame_group.base_frame.bandwidth / 1e6

        if num_scs != bandwidth // 20 * 64:
            raise ValueError(
                "Mask first dimension must match number of subcarriers "
                + "as specified implicitly by bandwidth.\n"
                + f" -- bandwidth : {bandwidth}\n"
                + f" -- mask dimension : {mask.shape}"
            )


def get_exp_time() -> str:
    """
    Create a current timestamp string
    """
    now = datetime.today()
    curr_time = now.strftime("%Y_%m_%d-T%H-%M-%S")
    return curr_time


def get_timed_hash(name: str) -> str:
    """
    Create a time-salted hash

    Args:
        name : Experiment name to create salted hash with
    """
    curr_time = get_exp_time()
    hash_id = hashlib.md5()
    hash_id.update((name + curr_time).encode("utf-8"))
    return hash_id.hexdigest()


def prepare_config(
    config: ExperimentConfig, env: EnvironmentConfig
) -> ExperimentConfig:
    """
    Prepare config, i.e. populate missing fields and perform validation

    Args:
        config : Experiment configuration to check and prepare
        env    : Hardware environment configuration
    """
    logger.trace("Creating time-salted session ids to ensure uniqueness")
    for session in config.sessions:
        session.session_id = get_timed_hash(session.name)

    sanity_check(config, env)

    return config


# ---------------------------------------------------------------------------------
# Experiment running
# ---------------------------------------------------------------------------------
class ExperimentState:
    """
    Class to manage experiment state, specifically about what happened in the
    individual sessions. Also used to summarize these details for future reference.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_dir = config.database_path / f"{config.exp_id}"
        self.db = Database(self.data_dir / "db.parquet")
        self.failed_sessions: List[SessionConfig] = []
        self.successful_sessions: List[SessionConfig] = []
        self.crashed_sessions: List[Tuple[SessionConfig, str]] = []

    def ingest_session(self, session: ReadySessionConfig, res: SessionResult):
        """
        Ingest a single completed session

        Args:
            session : The session config
            res     : The result the session run returned
        """

        orig_session = SessionGenerator.convert_from_ready(session)
        if res.failed_receivers:
            logger.warning(
                "Detected an issue with this session, some receivers failed!\n"
                + f" -- affected receivers : {res.failed_receivers}\n"
                + f" -- session id         : {session.session_id} \n"
                + f" -- session name       : {session.name} \n"
            )
            self.failed_sessions.append(orig_session)
        else:
            self.successful_sessions.append(orig_session)
            self.db.add_data(res.data)

    def ingest_crashed(self, session: ReadySessionConfig, tb: str):
        """
        Ingest a crashed session, i.e. one where an exception was thrown

        Args:
            session   : The session config
            tb        : A stacktrace string to report to the user later
        """
        orig_session = SessionGenerator.convert_from_ready(session)
        self.crashed_sessions.append((orig_session, tb))

    def _summarize_failed(self) -> str:
        """
        Create a string summary of the registered failed sessions
        """
        # If any sessions failed, i.e. a receiver didnt capture any data, print
        # some details for that
        sess_details = ""
        if len(self.failed_sessions) > 0:
            failed_details = "".join([
                f" -- idx: {s.session_id}, receivers: {[s.receivers]}\n"
                for s in self.failed_sessions
            ])

            sess_details = (
                "> Detail on failed sessions: \n"
                f"{failed_details}\n"
                "Caching these sessions for rerunning. To do so, execute:\n"
                "  ./app.py rerun_failed\n\n"
            )
        return sess_details

    def _summarize_crashed(self) -> str:
        """
        Create a string summary of the registered crashed sessions
        """
        sess_details = ""
        # We allow exceptions to avoid small errors hindering whole experiments.
        # Here we append all the info about those crashed sessions
        if len(self.crashed_sessions) > 0:
            crashed_details = "".join([
                f" -- idx: {s.session_id}, receivers: {[s.receivers]}\n"
                + f"Traceback: {tb}\n"
                for s, tb in self.crashed_sessions
            ])

            sess_details = (
                "> SOME SESSIONS CRASHED. Details on the crashes:"
                f" \n{crashed_details}\n\n"
            )

        return sess_details

    def summarize(self):
        """
        Log a full experiment state summary
        """
        # Get summaries for failed and crashed sessions
        sess_details = self._summarize_failed() + self._summarize_crashed()

        # In case non failed or crashed, let the user know :)
        if len(self.failed_sessions) == 0 and len(self.crashed_sessions) == 0:
            sess_details = "All experiments finished successfully, all good!"

        logger.warning(
            "Experiment summary: \n"
            + f" -- Successful Sessions : {len(self.successful_sessions)} \n"
            + f" -- Failed Sessions     : {len(self.failed_sessions)} \n"
            + f" -- Crashed Sessions    : {len(self.crashed_sessions)} \n \n"
            + "Details:\n "
            + f"{sess_details}"
        )

    def wrapup(self):
        """
        Wrap up! Store sessions in according shelves for later reuse and
        reference.
        """
        # If any sessions were unsuccessful, remember them for later reruns
        # NOTE: Here we dont distinguish between crashed and failed for now.
        if len(self.failed_sessions) > 0 or len(self.crashed_sessions) > 0:
            cache_experiment(
                self.config,
                self.failed_sessions + [s[0] for s in self.crashed_sessions],
                self.config.cache_dir / "failed_experiments",
            )

        # Also save successful experiments for later reference.
        if len(self.successful_sessions) > 0:
            persist_sessions(
                self.successful_sessions,
                self.data_dir / "completed_experiments",
            )


def run_experiment(config: ExperimentConfig, env: EnvironmentConfig):
    """
    Run a full experiment, i.e. all sessions specified in a config

    Args:
        config : The experiment configuration, containing all sessions
            and information required to run
        env : The config of the environment, i.e. all hardware installed
            in our experiment system
    """
    config = prepare_config(config, env)
    state = ExperimentState(config)
    total_sess_num = len(config.sessions)

    logger.trace("Firing up Session Generator to run experiments ...")

    for sess_num, session_cfg in enumerate(SessionGenerator(config, env)):
        logger.info(
            "\n"
            + "---------------------------------------------------------------\n"
            + "------------------- Starting to run session -------------------\n"
            + f"-- Session id   : {session_cfg.session_id} \n"
            + f"-- Session name : {session_cfg.name} \n"
            + f"-- Number       : {sess_num} / {total_sess_num} \n"
            + "---------------------------------------------------------------\n"
        )

        try:
            with Session(session_cfg, config.exp_id, config.performed_at) as s:
                session_res = s.run()
                # Ingest finished session to state handler
                state.ingest_session(session_cfg, session_res)
        except Exception as inst:
            tb = traceback.format_exc()
            logger.critical(
                " !! >> Session run caused an exception << !! \n"
                + f" -- Storing failed session to allow later reruns."
            )
            state.ingest_crashed(session_cfg, tb)
            continue

        # Ensure cleanup!
        # This is required because otherwise destructors of e.g. tools are not called.
        # Order of construction and destruction must be upheld, since e.g. PicoScenes
        # tool restores the interface to Managed afterwards.
        del s, session_cfg, session_res
        gc.collect()

        # Wait before next session
        if config.inter_session_delay_seconds > 0:
            time.sleep(config.inter_session_delay_seconds)

    # Wrapup to persist state for future reference and print summary of experiment
    state.wrapup()
    state.summarize()
