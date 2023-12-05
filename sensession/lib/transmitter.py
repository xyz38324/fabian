"""
Transmitter implementation to send frames
"""

from loguru import logger

from sensession.lib.shell import shell_run
from sensession.lib.config import Channel, TransmitterConfig, GeneratedFrameInfo
from sensession.lib.remote_access import Command


class Transmitter:
    """
    Class representing the USRP transmitting a fixed statically configured frame.

    NOTE: Could be extended to be an interface with instances for different types
    of transmitter, if desired.
    """

    def __init__(self, config: TransmitterConfig):
        self.config = config

    def transmit(
        self, frame_info: GeneratedFrameInfo, channel: Channel, gain: int, n_reps: int
    ):
        """
        Transmit

        Args:
            frame_info  : Information on generated frame (group) to transmit
            channel     : Channel on which to transmit samples
            gain        : Transmission gain to use
            n_reps      : Number of times to repeat this transmission
        """
        freq = channel.frequency
        rate = frame_info.base_frame.send_rate
        file = frame_info.frame_file

        # Sanity check file to be used for transmission
        assert (
            file.is_file() and file.stat().st_size > 1e4
        ), f"Sample file {file} does not exist or is broken."

        # If rate is in MS/s, change scale to Samples/s instead
        if rate < 1000:
            rate = int(rate * 1e6)

        filename = str(file)
        if self.config.access:
            # If on remote, rsync the file to the remote and point to it
            remote_name = self.config.access.remote_ssh_hostname
            sync_cmd = f"rsync {file} {remote_name}:/tmp"
            shell_run(sync_cmd)
            filename = f"/tmp/{file.name}"

        # Assemble command, possibly piping it through SSH onto a remote, if the
        # transmitter is not available locally on the current machine
        shell_cmd = Command(
            f"./scripts/transmit_from_sdr.sh {filename} {freq} {gain} {rate} {n_reps}"
        ).script_through_remote(self.config.access)

        logger.debug(
            "Transmitting dedicated frame with SDR.\n"
            + f" -- sample file : {file}\n"
            + f" -- frequency   : {freq}\n"
            + f" -- gain        : {gain}\n"
            + f" -- rate        : {rate}\n"
            + f" -- num repeats : {n_reps}\n"
        )

        shell_run(shell_cmd)

    def trim_usage(self, max_size: int = 10_000_000_000):
        """
        Trim files used for transmissions. This is only relevant if the transmitter
        is on a remote PC to which files are synced for transmission.

        Args:
            max_size : Byte size to trim
        """
        if not self.config.access:
            return

        logger.trace(
            f"Trimming tmp directory to {max_size} bytes to avoid trashing system."
        )

        cmd = Command(f"./scripts/trim_tmp.sh {max_size}").script_through_remote(
            self.config.access
        )
        shell_run(cmd)
