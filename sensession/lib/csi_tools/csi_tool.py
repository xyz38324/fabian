"""
CSI Tool Interface

CSI Tools are other applications that instrument hardware to capture CSI values.
This interface specifies the minimal set of functionalities required for them to
be used within sensession.
"""

from abc import ABCMeta, abstractmethod
from typing import List
from dataclasses import dataclass

import polars as pl

from sensession.lib.config import Channel, Receiver, BaseFrameConfig, ReadySessionConfig


# fmt: off
@dataclass
class CaptureResult:
    """
    Struct to bundle result of a CSI capture. A capture refers to running a tool
    for some amount of time and amassing the collected data.
    """
    data          : pl.DataFrame   # The captured data aggregated into a dataframe
    receiver_name : str            # Name of receiver from which data came
# fmt: on


class CsiTool(metaclass=ABCMeta):
    """
    Abstract interface base class specifying a CSI Capture Tool.

    NOTE: Implementations should take care of setup in their constructor
    """

    @abstractmethod
    def prepare_for(
        self, receiver_list: List[Receiver], frame: BaseFrameConfig, channel: Channel
    ):
        """
        Perform necessary setup required to capture CSI from given frame config on
        specified channel.

        Args:
            receiver_list : List of receivers to capture with
            frame         : Configuration of the frame to capture CSI from
            channel       : Configuration of channel on which frame is sent
        """

    @abstractmethod
    def run_capture(self, session: ReadySessionConfig):
        """
        Start capturing.

        Args:
            session : The current session

        Warning:
            Must not block!
        """

    @abstractmethod
    def stop_and_reap(self) -> List[CaptureResult]:
        """
        Stop the capture and reap the captured data
        """
