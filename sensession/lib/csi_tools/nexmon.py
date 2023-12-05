"""
Nexmon CSI Tool

Implementation to allow usage of multiple Nexmon devices within sensession.
"""

from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass

from loguru import logger

from sensession.lib.shell import shell_run
from sensession.lib.config import (
    Channel,
    Receiver,
    NexmonConfig,
    BaseFrameConfig,
    ReadySessionConfig,
    NetworkInterfaceMode,
)
from sensession.lib.capture_process import CaptureProcess
from sensession.lib.csi_tools.csi_tool import CsiTool, CaptureResult
from sensession.lib.csi_file_parser.nexmon import load_nexmon_data


# fmt: off
@dataclass
class NexmonInterfaceSettings:
    """
    Collection of information fully specifying configuration of the network
    interface on an asus router with Nexmon on it.
    """
    interface: str
    channel_number: int
    bandwidth_mhz: int
    sideband: str

@dataclass
class NexmonDeviceConfig:
    """
    Collection of config info for a nexmon device
    """
    interface_settings : NexmonInterfaceSettings     # Settings for nexmon device interface
    csiparams : str                                  # csiparams to set (depend on channel and frame source mac addr)
    netcat_port : int                                # Netcat port to stream data to
    device_ip : str                                  # Ip address to stream data to
    antenna_idxs : List[int]                         # Antennas with which to capture
    tmp_file : "Path | None" = None                  # Tmp file to capture data to
# fmt: on


def create_csi_params(
    receiver: Receiver, frame: BaseFrameConfig, channel: Channel
) -> str:
    """
    Create CSI params used for configuring nexmon device for e.g. filter rules.
    """
    logger.trace("Creating csiparams for nexmon device configuration ...")
    channel_number = channel.channel_number
    channel_bw = channel.bandwidth_mhz
    source_addr = frame.transmitter_address
    sideband = channel.sideband.value

    # Create bitmask from used antennas
    recv_antenna_bitmask = 0

    for antenna_idx in receiver.antenna_idxs:
        recv_antenna_bitmask |= 1 << antenna_idx

    if recv_antenna_bitmask == 0:
        raise RuntimeError(
            "Nexmon receive antenna bitmask must not be zero! \n"
            + f" -- Receive antennas specified in config : {receiver.antenna_idxs}"
        )

    # Call csi parameter creation shell script for nexmon
    csi_param_shell_cmd = (
        "./scripts/nexmon/make_csi_params.sh "
        + f"{channel_number} {channel_bw} {source_addr} {recv_antenna_bitmask} {sideband}"
    )
    csiparams = shell_run(
        csi_param_shell_cmd,
        capture=True,
    ).stdout.strip("\n")

    # Log info
    ant_idxs = receiver.antenna_idxs
    logger.debug(
        "Created nexmon csi params\n"
        + f" -- Channel number     : {channel_number}\n"
        + f" -- Channel bandwidth  : {channel_bw} MHz\n"
        + f" -- Source mac address : {source_addr}\n"
        + f" -- Receive antennas   : {ant_idxs} (mask: {recv_antenna_bitmask})\n"
        + f" --> csiparams         : {csiparams}"
    )

    if not csiparams:
        raise RuntimeError("Failed to create CSI params - EMPTY??")

    return csiparams


def _assign_compatible_ip(receiver: Receiver, tool_config: NexmonConfig):
    """
    Router is available only via specific subnet. We need to configure the
    corresponding interface to have a compatible ip address

    NOTE: Currently not needed since we statically set the address with netplan
    """
    logger.debug("Assigning compatible ip address to interface of Nexmon device ...")
    interface = receiver.interface
    device_ip = tool_config.device_ip
    netmask = tool_config.device_netmask
    shell_run(f"sudo ifconfig {interface} {device_ip} netmask {netmask}")


def setup_interface(receiver: str, interface_settings: NexmonInterfaceSettings):
    """
    Configure network interface on Nexmon device for CSI extraction
    """
    logger.trace(f"Configuring interface on remote : {receiver} ...")
    shell_run(
        "./scripts/nexmon/setup_interface.sh "
        + f"{receiver} "
        + f"{interface_settings.interface} "
        + f"{interface_settings.channel_number} "
        + f"{interface_settings.bandwidth_mhz} "
        + f"{interface_settings.sideband}"
    )


def set_filter_rules(receiver_name: str, csi_interface: str, csiparams: str):
    """
    Args:
        csi_interface: Network interface on which nexmon broadcasts CSI data
        csiparams:     Binary params to set filter rules on Nexmon device
    """
    logger.debug(f"Setting filter rules on remote : {receiver_name}")
    shell_run(
        "./scripts/nexmon/set_filter_rules.sh"
        + f" {receiver_name} {csi_interface} {csiparams}"
    )


class Nexmon(CsiTool):
    """
    Nexmon CSI collection adapter
    """

    def __init__(self):
        """
        Nexmon Device Constructor. Configures host and remote
        """
        super().__init__()

        self.devices: Dict[str, NexmonDeviceConfig] = {}
        self.bg_process = CaptureProcess()

    def __del__(self):
        self._restore(list(self.devices.keys()))

    def _restore(self, receiver_list: list):
        if not receiver_list:
            return

        # Trying my best; If you give me an iterator, that loop below will crash.
        # This would save us from such a case.
        if not isinstance(receiver_list, list):
            receiver_list = list(receiver_list)

        for receiver_name in receiver_list:
            config = self.devices[receiver_name]

            logger.trace(
                "Forgetting Nexmon Config and cleaning up temporary artifacts for: \n"
                + f" -- receiver : {receiver_name}\n"
                + f" -- tmp_file : {config.tmp_file}\n"
            )

            # File cleanup!
            if config.tmp_file:
                config.tmp_file.unlink(missing_ok=True)

            # And of course, forget
            self.devices.pop(receiver_name, None)

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
        current_receivers = set(self.devices.keys())
        requested_receivers = set(receiver.short_name for receiver in receiver_list)
        unused_receivers = current_receivers - requested_receivers

        # Restore the unused ones
        self._restore(list(unused_receivers))

        for receiver in receiver_list:
            # Extract tool specific config
            config = receiver.tool_config
            receiver_name = receiver.short_name

            if not isinstance(config, NexmonConfig):
                raise ValueError(
                    f"Nexmon Config for receiver {receiver_name} has wrong type "
                    + f"{type(config)}. Check config parsing implementation."
                )

            # Assert monitor mode since we dont support anything else for now
            if receiver.mode != NetworkInterfaceMode.MONITOR:
                raise NotImplementedError(
                    "We currently support only monitor mode for Nexmon"
                )

            interface_settings = NexmonInterfaceSettings(
                interface=config.csi_interface,
                channel_number=channel.channel_number,
                bandwidth_mhz=channel.bandwidth_mhz,
                sideband=channel.sideband.value,
            )

            if (
                receiver_name in self.devices
                and self.devices[receiver_name].interface_settings == interface_settings
            ):
                logger.trace(
                    f"Interface settings for {receiver_name} unchanged "
                    + "since last configuration. Nothing to do ..."
                )
            else:
                setup_interface(receiver_name, interface_settings)

            # Set filter rules on nexmon device to prepare for frame reception
            csiparams = create_csi_params(receiver, frame, channel)
            if (
                receiver_name in self.devices
                and self.devices[receiver_name].csiparams == csiparams
            ):
                logger.trace(
                    f"Nexmon CSI parameters for {receiver_name} unchanged "
                    + "since last configuration. Nothing to do ..."
                )
            else:
                set_filter_rules(receiver_name, interface_settings.interface, csiparams)

            # Store current device settings to check for changes in next config run
            self.devices[receiver_name] = NexmonDeviceConfig(
                interface_settings=interface_settings,
                csiparams=csiparams,
                netcat_port=config.netcat_port,
                device_ip=config.device_ip,
                antenna_idxs=receiver.antenna_idxs,
            )

    def run_capture(self, session: ReadySessionConfig):
        """
        Capture with Nexmon.

        NOTE: Nexmon will broadcast CSI on the device used for capture. We open a netcat
        tunnel to route that data back to the host.
        """

        # Start CSI listener instances on host
        for receiver_name, device_config in self.devices.items():
            device_config.tmp_file = session.cache_dir / f"{receiver_name}.pcap"
            tmp_file = device_config.tmp_file

            logger.debug(
                "Starting netcat listen on host to receive CSI ...\n"
                + f" -- receiver ssh name : {receiver_name}\n"
                + f" -- tmp file          : {tmp_file}\n"
                + f" -- destination port  : {device_config.netcat_port}\n"
                + f" -- destination ip    : {device_config.device_ip}\n"
            )

            self.bg_process.start_process(
                shell_command=(
                    "./scripts/nexmon/csi_capture_start.sh"
                    f" {device_config.netcat_port} {tmp_file}"
                ),
                cleanup_command=None,  # Netcat closes when sender is killed.
            )

        # Start CSI forwarding stream on Nexmon devices
        for receiver_name, device_config in self.devices.items():
            self.bg_process.start_process(
                shell_command=(
                    "./scripts/nexmon/csi_stream_start.sh"
                    f" {receiver_name} {device_config.interface_settings.interface} {device_config.netcat_port} {device_config.device_ip}"
                ),
                cleanup_command=f"./scripts/nexmon/csi_stream_stop.sh {receiver_name}",
            )

    def stop_and_reap(self) -> List[CaptureResult]:
        """
        Stop the capture and reap the captured data
        """
        self.bg_process.teardown()

        captures = []

        # Go through all devices that were prepared for this session and reap their data.
        for receiver_name, device_config in self.devices.items():
            tmp_file = device_config.tmp_file
            assert tmp_file, (
                "Nexmon CSI Tool class must maintain a path to a temp file to store "
                + "pcap captures temporarily before reaping!"
            )

            # If the file contains no data or doesnt even exist, we have no data to reap.
            if tmp_file.is_file() and tmp_file.stat().st_size < 10:
                tmp_file.unlink()
                continue
            if not tmp_file.is_file():
                continue

            data = load_nexmon_data(
                tmp_file,
                device_config.antenna_idxs,
                device_config.interface_settings.bandwidth_mhz,
            )
            captures.append(CaptureResult(data=data, receiver_name=receiver_name))

            logger.debug(
                f"Loaded nexmon data for {receiver_name}."
                + f"Found {captures[-1].data.shape} data points"
            )

        # Remove the temp data file on disk to avoid accidental reuse
        for receiver_name, device_config in self.devices.items():
            assert device_config.tmp_file, "Must have registered file to store data to"
            if device_config.tmp_file.is_file():
                device_config.tmp_file.unlink()

        return captures
