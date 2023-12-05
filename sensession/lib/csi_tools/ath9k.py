"""
Atheros QCA CSI Tool
"""

from typing import List
from pathlib import Path
from dataclasses import dataclass

from loguru import logger

from sensession.lib.shell import shell_run
from sensession.lib.config import (
    Channel,
    Receiver,
    Ath9kConfig,
    BaseFrameConfig,
    ReadySessionConfig,
    NetworkInterfaceMode,
)
from sensession.lib.remote_access import Command, create_marshall_copy_command
from sensession.lib.capture_process import CaptureProcess
from sensession.lib.csi_tools.csi_tool import CsiTool, CaptureResult
from sensession.lib.csi_file_parser.ath9k import load_ath9k_data


@dataclass
class Ath9kMonitorConfig:
    """
    Struct of monitor mode interface configuration parameters
    """

    channel_number: int
    channel_spec: str
    interface: str


class Ath9k(CsiTool):
    """
    Ath9k modified atheros driver CSI collection adapter
    """

    def __init__(self):
        super().__init__()
        self.configured_once: bool = False
        self.monitor_config: "Ath9kMonitorConfig | None" = None
        self.receiver: "Receiver | None" = None
        self.config: "Ath9kConfig | None" = None
        self.tmp_file: "Path | None" = None
        self.bg_process = CaptureProcess()

    def prepare_for(
        self,
        receiver_list: List[Receiver],
        frame: BaseFrameConfig,
        channel: Channel,
    ):
        """
        Prepare to capture
        """
        # Extract tool specific config and receiver
        self.receiver = receiver_list[0]
        config = self.receiver.tool_config
        monitor_config = Ath9kMonitorConfig(
            channel.channel_number, channel.channel_spec, self.receiver.interface
        )

        if not isinstance(config, Ath9kConfig):
            raise ValueError(
                "Ath9k Config was not properly converted. Check config parsing "
                + "implementation."
            )

        if self.configured_once and self.config != config:
            raise RuntimeError(
                "Change of ath9k tool config is not allowed; "
                + "Ath9k must refer to the same host and repo throughout execution!"
            )

        if len(receiver_list) != 1:
            raise NotImplementedError(
                "Ath9k tool implementation currently supports only a single receiver "
                + "device"
            )

        if self.receiver.mode != NetworkInterfaceMode.MONITOR:
            raise NotImplementedError(
                "We currently support only monitor mode for PicoScenes"
            )

        self.config = config

        # If this is the first configuration, ensure driver is loaded. Since we dont allow
        # to change the tool config, this need not be done again during one experiment.
        if not self.configured_once:
            self._reload_driver()

        # If monitoring mode configuration changed, apply the changes.
        if self.monitor_config != monitor_config:
            self.monitor_config = monitor_config
            self._monitor_mode()

        self.configured_once = True

    def run_capture(self, session: ReadySessionConfig):
        """
        Start a capture on given capture process.
        """
        # Output file
        if not self.configured_once:
            raise RuntimeError("Tried to run before ever configuring!")
        assert isinstance(self.receiver, Receiver), "Receiver must be set"
        assert isinstance(self.config, Ath9kConfig), "Ath9k Config must be set"

        receiver_name = self.receiver.short_name

        # Basic command
        capture_cmd = f"sudo {self.config.repo_path}/extractor/build/csi-extractor"
        cleanup_cmd = None

        # File handling: Remember which file data is in for stop_and_reap
        file_name = f"{receiver_name}.log"
        file_dir = session.cache_dir
        self.tmp_file = file_dir / f"{file_name}"

        # If ath9k runs on remote, prepend the proper marshalling
        access_cfg = self.receiver.access
        if access_cfg:
            # Stream into file on remote and then sync back..
            tmp_file = f"/tmp/{file_name}"
            capture_cmd = f"{capture_cmd} {tmp_file}"
            capture_cmd = Command(capture_cmd).on_remote(access_cfg, pseudo_tty=False)
            cleanup_cmd = create_marshall_copy_command(access_cfg, tmp_file, file_dir)

            # NOTE: Remote process is actually kept alive after local background capture process
            # is killed when not using pseudo_tty. That is because SIGINT is not forwarded through
            # ssh. Here, we stash a cleanup command to hackily kill the process on the remote.
            # This command shall only be executed at cleanup time
            shutdown_cmd = Command("sudo pkill csi-extractor").on_remote(
                access_cfg, pseudo_tty=False
            )
            self.bg_process.start_process(
                shell_command=None, cleanup_command=shutdown_cmd
            )
        else:
            # If we run locally, no need to perform any marshalling. Because of sudo
            # cleanup permission issues with subprocess, we wrap capturing in a shell
            # script and clean up with a cleanup command.
            capture_cmd = (
                f"./scripts/capture_qca.sh {self.config.repo_path} {self.tmp_file}"
            )
            cleanup_cmd = "sudo pkill csi-extractor"

        self.bg_process.start_process(
            capture_cmd,
            cleanup_command=cleanup_cmd,
        )

    def stop_and_reap(self) -> List[CaptureResult]:
        """
        Stop the capture and reap the captured data
        """
        if not self.configured_once:
            raise RuntimeError("Tried to reap data before ever configuring!")
        assert isinstance(self.receiver, Receiver), "Receiver not configured"
        assert isinstance(self.tmp_file, Path), "tmp file not configured"
        # Stop capture processes
        self.bg_process.teardown()

        # Load data from the temp file
        tmp_file = self.tmp_file
        logger.trace(f"Reaping data for qca (from file : {tmp_file})")
        if tmp_file.is_file() and tmp_file.stat().st_size < 10:
            tmp_file.unlink()
            return []

        data = load_ath9k_data(tmp_file, self.receiver.antenna_idxs)

        return [CaptureResult(data=data, receiver_name=self.receiver.short_name)]

    def _reload_driver(self):
        """
        Reload driver to activate it
        """
        assert isinstance(self.receiver, Receiver), "Receiver not configured"
        assert isinstance(self.config, Ath9kConfig), "No proper ath9k config set"

        access_cfg = self.receiver.access
        logger.trace(
            "Reloading ath9k driver. \n"
            + f"Repo path     : {self.config.repo_path} \n"
            + f"Access config : {access_cfg} \n"
        )

        reload_driver = Command(
            f"{self.config.repo_path}/driver/helper-scripts/load_custom_driver.sh --y"
        ).on_remote(access_cfg)

        shell_run(f"{reload_driver}")

    def _monitor_mode(self):
        """
        Enable monitor mode
        """
        assert isinstance(self.receiver, Receiver), "Receiver not configured"
        assert isinstance(
            self.monitor_config, Ath9kMonitorConfig
        ), "Must have monitor config"

        logger.trace("Enabling ath9k monitor mode ...")

        enable_monitor_mode = Command(
            "./scripts/monitor_mode.sh "
            + f"{self.monitor_config.interface} "
            + f"{self.monitor_config.channel_number} "
            + f"{self.monitor_config.channel_spec}"
        ).script_through_remote(self.receiver.access)

        shell_run(f"{enable_monitor_mode}")
