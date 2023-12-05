"""
Session Generation and Management

This file contains the following functionalities:

- A session generator that allows preparation of sessions, including management of
  frames to be sent in sessions, directories, etc.
- A session context wrapper to allow a clean session execution

Sessions are executed in the following way:
- First, a set of training frames (raw base frames) is sent, which may later be used
  for equalization and also provide a "warmup"
- The frame group specified by the user is selected.
  A frame group consists of a repetition set of masked frames, all derived from one base frame.
  Between every frame that is sent, the sequence number is increased in the group. For example,
  with two masks and 2 repetitions:
    > seq nr=0 : frame masked with mask 0  |- Repetition 0
    > seq nr=1 : frame masked with mask 1  /
    > seq nr=2 : frame masked with mask 0  |- Repetition 1
    > seq nr=3 : frame masked with mask 1  /
- All receivers participating in the session are activated to listen to the coming frame
- One after the other, each transmitter executes the transmission of the specified frame
  group
- The capture is stopped, data is converted into a common format and put in our database
"""

import time
import traceback
from typing import Set, Dict, List, Deque, Optional
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict
from dataclasses import dataclass

import polars as pl
from loguru import logger

from sensession.lib.config import (
    SessionConfig,
    CSICaptureTool,
    ExperimentConfig,
    FrameGroupConfig,
    EnvironmentConfig,
    GeneratedFrameInfo,
    ReadySessionConfig,
)
from sensession.lib.database import attach_label_data
from sensession.lib.transmitter import Transmitter
from sensession.lib.frame_generation import (
    FrameCache,
    to_training,
    generate_frame_group,
)
from sensession.lib.csi_tools.csi_tool import CaptureResult
from sensession.lib.csi_tools.tool_factory import instantiate_tool
from sensession.lib.matlab.matlab_parallelizer import Task, MatlabParallelizer


# fmt: off
@dataclass
class SessionResult:
    """
    Aggregated result data from one session
    """
    data             : pl.DataFrame              # Aggregated data from this session
    failed_receivers : List[str]                 # List of all receivers that failed (produced no data)


@dataclass
class FrameSessionGroup:
    """
    Dataclass to simply group sessions that use the same frame config
    """

    training_config : Optional[FrameGroupConfig] # Config of framegroup sent for training/equalization prior to actual session
    frame_config    : Optional[FrameGroupConfig] # Config of framegroup sent in actual session
    sessions        : List[SessionConfig]        # All sessions that use the above configuration of framegroups


@dataclass
class ReadySessionGroup:
    """
    Same as frame session group, but with the frames generated
    """

    sessions        : List[SessionConfig]
    training_config : "GeneratedFrameInfo | None" = None
    frame_config    : "GeneratedFrameInfo | None" = None
# fmt: on


# ---------------------------------------------------------------------------------
# Session Generator
# Ensures sessions are ready to be run
# ---------------------------------------------------------------------------------
def gen_frame_filename(base_dir: Path, framegroup_config: FrameGroupConfig) -> Path:
    """
    Generate a suitable filename for a framegroup config
    """
    digest = framegroup_config.group_id
    return base_dir / f"frame_digest_{digest}.dat"


class SessionGenerator:
    """
    Generator to perform full session setups. Mainly, this includes:
        - Efficiently generating frames (no duplicates etc)
        - Managing state of sessions (i.e. which ones are ready to run)
        - Properly distinguishing between runs that want training,
          modified frames, or both
    """

    def __init__(self, experiment: ExperimentConfig, env: EnvironmentConfig):
        self.experiment = experiment
        self.env = env
        frame_cache_dir = experiment.cache_dir / "frames"
        frame_cache_dir.mkdir(parents=True, exist_ok=True)

        self.frame_cache = FrameCache(frame_cache_dir)
        self.frame_generator = MatlabParallelizer(generate_frame_group)

        # Ready sessions contain sessions with associated cached frames.
        # Once cached sessions are "worked off", we generate new frames
        # from uncached sessions.
        self.ready_sessions: Deque[ReadySessionConfig] = deque()

        # Multiple sessions may use the same frames, so for the uncached
        # sessions we group them by frames they listen to. This distorts
        # the original order, but results in better caching and usage of
        # the frame generator
        self.unprepared_groups: Deque[FrameSessionGroup] = deque()

        # Split all sessions into the above queues
        self._split_session_groups(self.experiment.sessions)

        logger.debug(
            "Session Generator initialized. \n"
            + f" -- Number of unprepared groups : {len(self.unprepared_groups)}\n"
            + f" -- Number of ready sessions    : {len(self.ready_sessions)}\n"
        )

    def _split_session_groups(self, sessions: List[SessionConfig]):
        """
        Split all sessions into either ready or uncached queues

        Args:
            sessions : List of sessions to split
        """
        # Start by grouping according to the frames desired in the sessions
        sess_group_dct: Dict[str, List[SessionConfig]] = defaultdict(list)
        for session in sessions:
            frame_id = session.frame_group.group_id
            sess_group_dct[frame_id].append(session)

        # Extract the groups from there
        sess_groups = [
            FrameSessionGroup(training_config=None, frame_config=None, sessions=sess)
            for sess in sess_group_dct.values()
        ]
        logger.trace(
            f"Found {len(sess_groups)} different session groups "
            + "(grouped by wanting the same frame in their sessions)"
        )

        # First, populate the frame config fields of the default-initialized groups
        for group in sess_groups:
            training_desired = any(s.training_reps > 0 for s in group.sessions)
            mask_run_desired = any(s.n_repeat > 0 for s in group.sessions)

            if not training_desired and not mask_run_desired:
                raise ValueError(
                    "Session with neither training nor mask runs found. "
                    + "That makes no sense!"
                )

            # Extract one of the groups. Because of the grouping above, these are
            # the same between all sessions in this list.
            frame_info = group.sessions[0].frame_group

            # Resolve configs
            ready = True
            if training_desired:
                group.training_config = to_training(frame_info)
                ready &= self.frame_cache.is_cached(group.training_config)
            if mask_run_desired:
                group.frame_config = frame_info
                ready &= self.frame_cache.is_cached(group.frame_config)

            # Now we can sort them into the queues.
            # NOTE: Theoretically, there is another distinction on groups, namely
            # whether they desire only training phase, no training phase, or mask
            # and training phase. We ignore this here.
            if ready:
                self._enqueue_ready_session_group(group)
            else:
                self.unprepared_groups.append(group)

    def _enqueue_ready_session_group(self, group: FrameSessionGroup):
        """
        Enqueue a session group into the ready queue.
        Called for a group of sessions when their respective frames are creted
        and in cache.

        Args:
            group : A group of sessions whose frames are ready to roar
        """
        # Training frame is equal for all the group sessions
        training_info = self.frame_cache.get_cached_frame(group.training_config)

        for session in group.sessions:
            training_desired = session.training_reps > 0
            mask_run_desired = session.n_repeat > 0

            if not training_desired and not mask_run_desired:
                raise RuntimeError(
                    f"Session {session.session_id} specified neither training nor mask "
                    + "precoding phase. Nothing to rum; Assuming this is an error."
                )

            if training_desired and not training_info:
                raise RuntimeError(
                    f"Session {session.session_id} specified training phase, but no"
                    + " training frame was found in the cache."
                )

            # Cache appends metaparameters from frame group (e.g. repetitions), which
            # can differ even for sessions in this group. Hene we repeatedly query here
            modframe_info = self.frame_cache.get_cached_frame(session.frame_group)

            if mask_run_desired and not modframe_info:
                raise RuntimeError(
                    f"Session {session.session_id} specified mask precoding phase,"
                    + " but no corresponding frame was found in the cache."
                )

            # Finally after all the sanity checks, enqueue
            self.ready_sessions.append(
                self._session_to_ready(
                    session,
                    training_frame_info=training_info if training_desired else None,
                    frame_info=modframe_info if mask_run_desired else None,
                )
            )

    def _session_to_ready(
        self,
        session: SessionConfig,
        training_frame_info: "GeneratedFrameInfo | None",
        frame_info: "GeneratedFrameInfo | None",
    ) -> ReadySessionConfig:
        """
        Create a ready session config
        """
        cache_dir = self.experiment.cache_dir / "data" / f"capture_{session.session_id}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Filter desired receiver and transmitter
        receivers = [
            receiver
            for receiver in self.env.receivers
            if receiver.short_name in session.receivers
        ]
        transmitter_cfgs = [
            transmitter_cfg
            for transmitter_cfg in self.env.transmitter
            if transmitter_cfg.name in session.transmitter
        ]

        # Create corresponding objects
        transmitter = [Transmitter(cfg) for cfg in transmitter_cfgs]
        tool_types = set(cfg.tool_type for cfg in receivers)
        tools = {tool_type: instantiate_tool(tool_type) for tool_type in tool_types}

        return ReadySessionConfig(
            session_id=session.session_id,
            channel=session.channel,
            receivers=receivers,
            tools=tools,
            frame_info=frame_info,
            train_frame_info=training_frame_info,
            cache_dir=cache_dir,
            transmitter=transmitter,
            tx_gain=session.tx_gain,
            n_repeat=session.n_repeat,
            training_reps=session.training_reps,
            name=session.name,
        )

    def _get_session_group_batch(self) -> List[FrameSessionGroup]:
        """
        Get a batch of session configs to generate frames for
        """
        batch = []

        for _ in range(self.experiment.matlab_batch_size):
            try:
                e = self.unprepared_groups.popleft()
            except IndexError:
                break

            batch.append(e)

        return batch

    def _add_to_generator(self, task_id, frame_group: FrameGroupConfig):
        """
        Add a frame group config to the frame generator to generate the described
        frame later with it.

        Args:
            task_id     : An id of the task to identify this generation request later on
            frame_group : Config of the frame to generate
        """
        frame_file = gen_frame_filename(self.frame_cache.cache_dir, frame_group)
        self.frame_generator.add_task(
            Task(
                task_id=task_id,
                kwargs={"config": frame_group, "frame_file": frame_file},
            )
        )

    def __iter__(self):
        return self

    def __next__(self) -> ReadySessionConfig:
        """
        Iterator function to spit out a new session config ready to run session with
        """
        if len(self.ready_sessions) == 0 and len(self.unprepared_groups) == 0:
            raise StopIteration()

        if len(self.ready_sessions) != 0:
            return self.ready_sessions.popleft()

        # At this point, no sessions are ready, but we still have unprepared
        # ones, so we need to prepare them. Specifically, we need to create
        # frames for them. First, we trim the cache to ensure its size isnt
        # exploding.
        self.frame_cache.trim_cache(self.experiment.cache_trim)

        # We already have groups of sessions depending all on the same frame.
        # To exploit parallelism, we create multiple frames, i.e. work off
        # multiple groups at once.
        batch: List[FrameSessionGroup] = self._get_session_group_batch()
        batch_size = len(batch)
        logger.info(f"Retrieved batch of {batch_size} session configs to work on ...")

        if self.frame_generator.get_num_tasks() != 0:
            raise RuntimeError(
                "Found unfinished tasks in frame generator during "
                + "Session preparation. Frame generator should also be worked off "
                + "before creating new sessions due to task indices not making sense "
                + "otherwise. Contention issue?"
            )

        # To distinguish training from mask frames, we assign indices:
        # - Mask precoded frames get its from 0 to batch_size - 1
        # - Training (unmodified) frames from batch_size to 2 * batch_size - 1
        for idx, session_grp in enumerate(batch):
            training_info = session_grp.training_config
            modframe_info = session_grp.frame_config

            if training_info and not self.frame_cache.is_cached(training_info):
                self._add_to_generator(idx, training_info)
            if modframe_info and not self.frame_cache.is_cached(modframe_info):
                self._add_to_generator(idx + batch_size, modframe_info)

        # Process frame generation and cache the new frames
        new_frames = self.frame_generator.process()
        for task in new_frames:
            self.frame_cache.register_cached_file(frame_info=task.retval)

        # Move the groups to the ready queue.
        # NOTE: This retrieves frames from the cache. Since we haven't cleared
        # the cache after checking for cached frames and creating missing ones,
        # this must work.
        for group in batch:
            self._enqueue_ready_session_group(group)

        # Finally, return first element of the newly prepared ones
        return self.ready_sessions.popleft()

    @staticmethod
    def convert_from_ready(session: ReadySessionConfig) -> SessionConfig:
        """
        Convert back to original session config from a ReadySessionConfig.
        Ready sessions are those that have generated frames lying in a file
        stored in that struct.

        Args:
            session : Ready session to use to extract original session config
        """
        frame_info = session.frame_info
        train_info = session.train_frame_info

        base_info = train_info
        if frame_info:
            base_info = frame_info

        if not base_info:
            raise RuntimeError(
                "Tried to extract session config from faulty ready "
                + "session. It contained no frame information!"
            )

        # TODO: Improve? Why is python all about object-oriented stuff and then
        # I cant even cast a derived class object to its base type smh -.-
        framegroup_cfg = FrameGroupConfig(
            group_id=base_info.group_id,
            base_frame=base_info.base_frame,
            mask_group=base_info.mask_group,
            mask_id=base_info.mask_id,
            mask_name=base_info.mask_name,
            group_repetitions=base_info.group_repetitions,
            interframe_delay=base_info.interframe_delay,
        )

        return SessionConfig(
            session_id=session.session_id,
            name=session.name,
            channel=session.channel,
            receivers=[rcv.short_name for rcv in session.receivers],
            frame_group=framegroup_cfg,
            transmitter=[tx.config.name for tx in session.transmitter],
            tx_gain=session.tx_gain,
            training_reps=session.training_reps,
            n_repeat=session.n_repeat,
        )


# ---------------------------------------------------------------------------------
# Session
# Orchestrates a ready for execution session.
# ---------------------------------------------------------------------------------
def get_session_mode(config: ReadySessionConfig) -> str:
    """
    Get session mode, i.e. a string that described the combination of training and
    masked frame session that was specified

    Args:
        config : The prepared session with frame infos
    """
    training_info = config.train_frame_info
    modframe_info = config.frame_info
    mode = ""
    if training_info and modframe_info:
        mode = "Training + Modified Frame"
    elif training_info:
        mode = "Training Only"
    elif modframe_info:
        mode = "Modified Frame Only"
    else:
        raise RuntimeError(
            "Got config for ready session, but it contained no "
            + "frame information whatsoever!"
        )

    return mode


class Session:
    """
    CSI Sensing Session class
    """

    def __init__(
        self, config: ReadySessionConfig, experiment_id: str, experiment_time: datetime
    ):
        """
        Session Constructor.
        """
        self.config = config
        self.experiment_id = experiment_id
        self.experiment_time = experiment_time

        # Group receiver into tool-keyed lists
        self.receiver_groups = defaultdict(list)
        for receiver in config.receivers:
            self.receiver_groups[receiver.tool_type].append(receiver)

        # Get one of the frame infos for logging some info.
        # mod frame info > train frame info
        base_info = config.train_frame_info
        if config.frame_info:
            base_info = config.frame_info

        if not base_info:
            raise RuntimeError(
                "Tried to extract session config from faulty ready "
                + "session. It contained no frame information!"
            )
        self.base_info: GeneratedFrameInfo = base_info

        mode = get_session_mode(config)

        logger.trace(
            "Initializing Session... \n"
            + f" -- Mask name     : {base_info.mask_name}\n"
            + f" -- Mask id       : {base_info.mask_id}\n"
            + f" -- Session name  : {config.name}\n"
            + f" -- Session id    : {config.session_id}\n"
            + f" -- Training reps : {config.training_reps}\n"
            + f" -- Frame reps    : {base_info.group_repetitions}\n"
            + f" -- mode          : {mode}"
        )
        self.wait_time = 2 if CSICaptureTool.PICOSCENES in self.config.tools else 0

    def __enter__(self):
        """
        Session context enter -- prepare everything for session
        """
        base_frame = self.base_info.base_frame
        channel = self.config.channel

        # Prepare tools to capture for current session
        for tool_type, receiver_list in self.receiver_groups.items():
            self.config.tools[tool_type].prepare_for(receiver_list, base_frame, channel)

        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Context exit. Ensure that subprocesses are properly terminated.
        """
        logger.debug("Finishing capture session ...")

        if exc_type is not None:
            logger.error("Failed to stop session gracefully!")
            traceback.print_exception(exc_type, exc_value, tb)
            return False

        return True

    def _run(self, frame_info: GeneratedFrameInfo, n_reps: int) -> List[CaptureResult]:
        """
        Perform one sensing run with a given frame
        """
        if n_reps == 0:
            return []

        # Start the tools
        for tool in self.config.tools.values():
            tool.run_capture(session=self.config)

        # Wait for a given time for tool setup processes to ramp up.
        time.sleep(self.wait_time)

        # Perform transmission
        for transmitter in self.config.transmitter:
            try:
                transmitter.transmit(
                    frame_info=frame_info,
                    channel=self.config.channel,
                    gain=self.config.tx_gain,
                    n_reps=n_reps,
                )
            except Exception as inst:
                tb = traceback.format_exc()
                logger.critical(
                    " !! >> Transmitter transmission crashed !! \n"
                    + f" -- {type(inst)} \n"
                    + f" -- {inst.args} \n"
                    + f" -- {inst} \n"
                    + f" -- trace: {tb}"
                )

        # Aggregate CaptureResults
        train_results: List[CaptureResult] = []
        for tool in self.config.tools.values():
            train_results.extend(tool.stop_and_reap())

        # Some debug printing
        recv_num_data_strs = [
            f"\n -- {res.receiver_name} : {res.data.height}" for res in train_results
        ]
        logger.debug(
            "Session run finished. Receivers captured:"
            + "".join(recv_num_data_strs)
            + "\n"
        )
        return train_results

    def _attach_label_data(
        self,
        results: List[CaptureResult],
        frame_info: GeneratedFrameInfo,
        session_timestamp: datetime,
    ) -> List[pl.DataFrame]:
        """
        Attach (meta-) label data to the captured CSI results
        """
        # Aggregate all data frames with corresponding session labels in list
        logger.trace("Attaching label (meta-)data to captured CSI dataframes")

        return [
            attach_label_data(
                df=res.data,
                experiment_id=self.experiment_id,
                experiment_time=self.experiment_time,
                session_name=self.config.name,
                session_id=self.config.session_id,
                frame_id=frame_info.group_id,
                mask_id=frame_info.mask_id,
                mask_name=frame_info.mask_name,
                receiver_name=res.receiver_name,
                channel=self.config.channel.channel_number,
                bandwidth=self.config.channel.bandwidth_mhz,
                tx_gain=self.config.tx_gain,
                session_timestamp=session_timestamp,
            )
            for res in results
        ]

    def run(self) -> SessionResult:
        """
        Execute session; Start capturing with all receivers and transmit with Transmitter
        """
        participants = [rcv.short_name for rcv in self.config.receivers]
        logger.debug(f"Starting sensing session! Participants: {participants}")

        if len(participants) == 0:
            logger.warning("Session contains no receivers to listen! Exiting ...")
            return SessionResult(data=pl.DataFrame(), failed_receivers=[])

        failed_receivers: Set[str] = set()
        labeled_data: List[pl.DataFrame] = []

        # --------------------------------------------------------------------------
        # Before all, we ensure that the transmitters file usage is not too bad
        # --------------------------------------------------------------------------
        for transmitter in self.config.transmitter:
            transmitter.trim_usage()

        # --------------------------------------------------------------------------
        # First, we perform a training run. There, we simply repeatedly transmit the
        # unmodified base frame. The capture results may later be used for shape eq.
        # --------------------------------------------------------------------------
        if self.config.training_reps > 0:
            logger.trace("Performing training run with unmodified frame ...")
            assert self.config.train_frame_info, "Cant train without training frame"
            session_timestamp = datetime.now()
            results = self._run(self.config.train_frame_info, self.config.training_reps)

            # Check which receivers have failed
            failed_receivers = failed_receivers.union([
                res.receiver_name for res in results if res.data.is_empty()
            ])

            # Attach label data
            labeled_data.extend(
                self._attach_label_data(
                    results, self.config.train_frame_info, session_timestamp
                )
            )
        else:
            logger.warning("No training run desired - skipping.")

        # --------------------------------------------------------------------------
        # Then we run the actual frame groups.
        # --------------------------------------------------------------------------
        if (
            self.config.frame_info
            and self.config.frame_info.group_repetitions > 0
            and self.config.n_repeat > 0
            and failed_receivers == set()
        ):
            logger.trace("Performing run with non-trivial frame groups ...")
            session_timestamp = datetime.now()
            results = self._run(self.config.frame_info, n_reps=self.config.n_repeat)

            # Check which receivers have failed
            failed_receivers = failed_receivers.union([
                res.receiver_name for res in results if res.data.is_empty()
            ])

            # Attach label data
            labeled_data.extend(
                self._attach_label_data(
                    results, self.config.frame_info, session_timestamp
                )
            )
        else:
            logger.warning(
                "No frame groups to run; If expected, ignore. "
                + "Otherwise, check repetition parameters."
            )

        # Concatenate all captured data together!
        data = pl.concat(labeled_data)

        return SessionResult(data=data, failed_receivers=list(failed_receivers))
