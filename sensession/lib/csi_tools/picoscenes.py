"""
PicoScenes CSI Tool
"""

from typing import Dict, List, Deque, Tuple
from pathlib import Path
from collections import deque
from dataclasses import dataclass

from loguru import logger

from sensession.lib.shell import shell_run
from sensession.lib.config import (
    Channel,
    Receiver,
    BaseFrameConfig,
    PicoScenesConfig,
    ReadySessionConfig,
    NetworkInterfaceMode,
)
from sensession.lib.capture_process import CaptureProcess
from sensession.lib.csi_tools.csi_tool import CsiTool, CaptureResult
from sensession.lib.csi_file_parser.picoscenes import load_picoscenes_data


@dataclass
class PicoscenesMonitorConfig:
    """
    Network interface config parameters
    """

    interface: str
    channel_number: int
    channel_spec: str


@dataclass
class PicoscenesFilterConfig:
    """
    Picoscenes specific parameters to specify a device and what it captures
    """

    phy_path: int
    transmitter_mac: str
    receiver_mac: str
    antenna_idxs: List[int]


# type alias
ReceiverMap = Dict[str, Tuple[PicoscenesMonitorConfig, PicoscenesFilterConfig]]


@dataclass
class Capture:
    """
    Struct to connect receiver to temp file in which data captures are stored
    """

    receiver_name: str  # Name of receiver that captured
    file: Path  # File in which data was captured
    antenna_idxs: List[int]  # Antenna idxs used for the capture


class PicoScenes(CsiTool):
    """
    PicoScenes CSI collection adapter
    """

    def __init__(self):
        super().__init__()
        self.configured_receivers: ReceiverMap = {}

        # We use PicoScenes in a background process to stream the CSI data to files
        # When CSI capture is desired, we shoot up an instance of PicoScenes in the
        # background with its lifetime managed through the `bg_process` and enqueue
        # the corresponding file.
        # When `stop_and_reap` is called, the process is stopped and the respective
        # file is "worked off", i.e. converted to our internal data format.
        self.capture_files: Deque[Capture] = deque()
        self.bg_process = CaptureProcess()

    def __del__(self):
        """
        Destructor, clean up the tool instance.
        """
        logger.trace(
            "Destroying PicoScenes object -- Cleaning up"
            + f" -- Configured receivers:  {list(self.configured_receivers.keys())}"
        )

        self._restore(list(self.configured_receivers.keys()))

    def prepare_for(
        self,
        receiver_list: List[Receiver],
        frame: BaseFrameConfig,
        channel: Channel,
    ):
        """
        Prepare to capture
        """
        # Restore and remove unused receivers
        current_receivers = set(self.configured_receivers.keys())
        requested_receivers = set(receiver.short_name for receiver in receiver_list)
        unused_receivers = current_receivers - requested_receivers

        logger.info(
            "Preparing PicoScenes ! \n"
            + f" -- Currently configured receivers : {current_receivers}\n"
            + f" -- Requested receivers            : {requested_receivers}\n"
            + f" -- Unused (to be restored)        : {unused_receivers}\n"
        )
        self._restore(list(unused_receivers))

        # Sanity-check all receivers and put them into monitor mode.
        for receiver in receiver_list:
            config = receiver.tool_config
            receiver_name = receiver.short_name

            if receiver.access:
                raise NotImplementedError(
                    "Remote PicoScenes access not implemented. "
                    + "If you want to change this, wrap all local commands in "
                    + "Command().on_remote() where appropriate."
                )

            if not isinstance(config, PicoScenesConfig):
                raise ValueError(
                    "PicoScenes Config was not properly converted. "
                    + "Check config parsing implementation."
                )

            if receiver.mode != NetworkInterfaceMode.MONITOR:
                raise NotImplementedError(
                    "We currently support only monitor mode for PicoScenes"
                )

            monitor_config = PicoscenesMonitorConfig(
                interface=receiver.interface,
                channel_number=channel.channel_number,
                channel_spec=channel.channel_spec,
            )

            filter_config = PicoscenesFilterConfig(
                phy_path=config.phy_path,
                transmitter_mac=frame.transmitter_address.lower(),
                receiver_mac=frame.receiver_address.lower(),
                antenna_idxs=receiver.antenna_idxs,
            )

            monitor_changed = (
                receiver_name not in self.configured_receivers
                or self.configured_receivers[receiver_name][0] != monitor_config
            )

            self.configured_receivers[receiver_name] = (monitor_config, filter_config)

            # Monitor mode go brr!
            if monitor_changed:
                shell_run(
                    "./scripts/monitor_mode.sh "
                    + f"{monitor_config.interface} "
                    + f"{monitor_config.channel_number} "
                    + f"{monitor_config.channel_spec}"
                )

    def run_capture(self, session: ReadySessionConfig):
        """
        Start a capture on given capture process.

        Args:
            session : Session information to run with
        """
        picoscenes_cmd = "PicoScenes -d debug;"

        for receiver_name, (_, filter_config) in self.configured_receivers.items():
            # PicoScenes is weird and wants a relative path ...
            # NOTE: file ending attached per convention by PicoScenes
            outfile = f"./{session.cache_dir.relative_to(Path.cwd())}/{receiver_name}"

            self.capture_files.append(
                Capture(
                    receiver_name=receiver_name,
                    file=session.cache_dir / f"{receiver_name}.csi",
                    antenna_idxs=filter_config.antenna_idxs,
                )
            )

            picoscenes_cmd += (
                f" --interface {filter_config.phy_path}"
                f" --output {outfile}"
                f" --source-address-filter {filter_config.transmitter_mac}"
                f" --destination-address-filter {filter_config.receiver_mac}"
                " --mode logger;"
            )

        # Start one unified process with all the receivers.
        self.bg_process.start_process(picoscenes_cmd)

    def stop_and_reap(self) -> List[CaptureResult]:
        """
        Stop the capture and reap the captured data
        """
        logger.trace(
            "Stopping and reaping data from picoscenes. "
            + "Starting with capture process teardown ..."
        )
        self.bg_process.teardown()

        # Aggregate data from all receivers and check which of them produced none
        captures = []

        logger.trace(
            "Capture processes stopped, moving on to reap data from temp files ..."
        )
        while self.capture_files:
            capture = self.capture_files.popleft()

            if capture.file.is_file() and capture.file.stat().st_size < 10:
                capture.file.unlink()
                continue

            df = load_picoscenes_data(capture.file, capture.antenna_idxs)
            captures.append(CaptureResult(data=df, receiver_name=capture.receiver_name))

            # Clean up to avoid thrashing the cache directory
            # TODO: This should be a meta option, as well as for the Nexmon case, to possibly
            # not delete temp files automatically.
            capture.file.unlink(missing_ok=True)

        return captures

    def _restore(self, receiver_list: list):
        """
        Restore receivers into a basic managed NIC state

        Args:
            receiver_list : List of receivers (by short name) to restore
        """
        if not receiver_list:
            return

        # Sometimes the qca capture is fragile and yields no results, especially when other
        # cards are still active and in monitor mode. This seems to improve when those are
        # removed, so we do that after every capture run ...
        logger.trace(f"Restoring NIC interfaces for: {receiver_list}")

        for receiver_name in receiver_list:
            monitor_config, _ = self.configured_receivers[receiver_name]
            interface = monitor_config.interface

            logger.debug(
                "Cleaning up PicoScenes object: \n"
                + f" -- receiver : {receiver_name}\n"
                + f" -- interface : {interface}"
            )

            cleanup_cmd = (
                f"sudo ifconfig {interface} down && "
                f"sudo iwconfig {interface} mode managed"
            )
            shell_run(cleanup_cmd)

            self.configured_receivers.pop(receiver_name, None)
