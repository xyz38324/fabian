"""
Collection of configurations used throughout the library.

Most importantly, contains all the experiment and sensing session participant
configuration structs.
"""

import json

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[11]

from enum import Enum
from typing import Dict, List, Type, Union, Optional
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import field, asdict, dataclass

import numpy as np


#######################################################################################
## Modes and discrete choices
#######################################################################################
# fmt: off
class NetworkInterfaceMode(str, Enum):
    """
    Mode of network interface
    """
    MONITOR    = "monitor"                # Linux networking monitor mode

class CSICaptureTool(str, Enum):
    """
    Enum of capture tool names
    """
    PICOSCENES = "PicoScenes"
    ESP32      = "ESP32Toolkit"
    NEXMON     = "Nexmon"
    ATH9K      = "Ath9k"

class GuardIntervalMode(str, Enum):
    """
    Mode of 802.11 guard intervals
    """
    SHORT = "Short"   # 0.4 ns guard interval length between symbols
    LONG  = "Long"    # 0.8 ns guard interval length between symbols

class DataRateMode(str, Enum):
    """
    802.11 data rate mode
    """
    HIGH_THROUGHPUT      = "HT"   # 802.11n
    VERY_HIGH_THROUGHPUT = "VHT"  # 802.11ac
    # NON-HT
    # HE

class Sideband(str, Enum):
    """
    Sideband of channel

    This is relevant for larger channels, e.g. 40 MHz. See 802.11 standard.
    """
    UPPER = "u"
    LOWER = "l"
    NONE  = ""


#######################################################################################
## CSI Capture Tool specific options
#######################################################################################
@dataclass(frozen=True)
class PicoScenesConfig:
    """
    PicoScenes Tool specific required config information
    """
    phy_path : int                          # PhyPath (check `array_status` command)


@dataclass(frozen=True)
class NexmonConfig:
    """
    Nexmon Tool specific required config information
    """
    device_ip      : str                    # Ip of the router/device hosting nexmon
    device_netmask : str                    # Netmask of subnet device is in
    csi_interface  : str                    # Interface on router where CSI is broadcasted
    netcat_port    : int                    # Port through which to forward CSI to host via netcat


@dataclass(frozen=True)
class Ath9kConfig:
    """
    Ath9k QCA Tool specific required config information
    """
    repo_path : str                         # Path to modified ath9k driver repository


@dataclass(frozen=True)
class ESP32Config:
    """
    ESP32-CSI Tool specific required config information
    """
    dummy : int


# Type alias
CSIToolConfig = Union[dict, PicoScenesConfig, NexmonConfig, Ath9kConfig]



#######################################################################################
## Assembled config for one receiver
#######################################################################################
@dataclass(frozen=True)
class SshPasswordless:
    """
    Configuration for passwordless reemote access via ssh.

    Requires a corresponding setup in ˜/.ssh/config
    """
    remote_ssh_hostname : str                              # Hostname (per ssh config) for passwordless access

@dataclass
class Receiver:
    """
    Configuration information of a receiver
    """
    name             : str                                 # Full name of the card for information
    short_name       : str                                 # Short name for references
    interface        : str                                 # Corresponding network interface
    mode             : NetworkInterfaceMode                # Mode of network interface
    antenna_idxs     : List[int]                           # The antennas that are used for capturing
    mac_address      : str                                 # Mac address of the card / interface
    tool_config      : CSIToolConfig                       # Spec for tool to collect CSI with
    access           : Optional[SshPasswordless] = None    # If receiver is situated on another machine; an access specification to it
    tool_type        : CSICaptureTool            = field(init=False)  # The capture tool type

    def __post_init__(self):
        if isinstance(self.tool_config, dict):
            assert len(self.tool_config) == 1, "Must specify exactly one csi tool"
            (tool_name, tool_config), = self.tool_config.items()
            self.tool_type = getattr(CSICaptureTool, tool_name.upper())

            # Unwrap config into dataclasses
            if self.tool_type == CSICaptureTool.PICOSCENES:
                self.tool_config = PicoScenesConfig(**tool_config)
            elif self.tool_type == CSICaptureTool.NEXMON:
                self.tool_config = NexmonConfig(**tool_config)
            elif self.tool_type == CSICaptureTool.ATH9K:
                self.tool_config = Ath9kConfig(**tool_config)
            else:
                raise NotImplementedError(f"No config mapping for {self.tool_type} implemented")

        if isinstance(self.mode, str):
            # Convert string to enum
            self.mode = getattr(NetworkInterfaceMode, self.mode.upper())

        if self.access and isinstance(self.access, dict):
            assert "ssh" in self.access, "Only supporting ssh remote access currently"
            self.access = SshPasswordless(**self.access["ssh"])


#######################################################################################
## Transmitter/Transmission Config
## Specifies transmitter SDR and parameters for the transmission from it
#######################################################################################
@dataclass
class TransmitterConfig:
    """
    Description of a transmitter.

    Currently we only allow for USRP transmitters, so this is kept simple
    """
    name     : str                              # Name of transmitter
    access   : Optional[SshPasswordless] = None # If transmitter is situated on another machine; an access specification to it

    def __post_init__(self):
        if self.access and isinstance(self.access, dict):
            assert "ssh" in self.access, "Only supporting ssh remote access currently"
            self.access = SshPasswordless(**self.access["ssh"])


#######################################################################################
## Hardware Setup; Assembling all hardware participant configuration
#######################################################################################
@dataclass
class EnvironmentConfig:
    """
    Environment config, i.e. the bundled information on all receivers and transmitters
    available in the actual physical system.
    """
    receivers: List[Receiver]
    transmitter: List[TransmitterConfig]

    def __post_init__(self):
        if all(isinstance(r, dict) for r in self.receivers):
            self.receivers = [Receiver(**receiver) for receiver in self.receivers] # type: ignore  # ensured to be dict

        if all(isinstance(t, dict) for t in self.transmitter):
            self.transmitter = [TransmitterConfig(**transmitter) for transmitter in self.transmitter] # type: ignore # as above



#######################################################################################
## Configuration of the frame to be generated for and sent with the SDR. This frame is
## detected at the receiver(s) and used to extract CSI. As such, PHY and MAC headers
## contain some relevant information.
#######################################################################################
FrameDelay = Union[int, timedelta] # Either number of samples or a nanosecond-level delay between frames

@dataclass(frozen=True, eq=True)
class BaseFrameConfig:
    """
    Basic frame configuration, i.e. parameters used to specify a single simple
    WiFi 802.11 frame
    """
    index               : int               # Index to allow referencing this specific frame
    receiver_address    : str               # Mac address of receiver of the frame
    transmitter_address : str               # Mac address of the transsmitter (e.g. AP)
    bssid_address       : str               # BSSID address, usually equal transmitter_address
    bandwidth           : int  = int(20e6)  # Bandwidth of underlying channel
    send_rate           : int  = int(20e6)  # The rate with which this frame is to be sent upsampled to avoid congruency issues.
    enable_sounding     : bool = False      # Whether to force sounding bit in PHY preamble
    rescale_factor      : int  = 25000      # Max value to scale int16 sample outputs in file to (applied on base frame before precoding)
    guard_iv_mode       : GuardIntervalMode = GuardIntervalMode.SHORT         # Specifies length of guard interval in PPDU
    data_rate_mode      : DataRateMode      = DataRateMode.HIGH_THROUGHPUT    # HT mode


@dataclass(frozen=True)
class FrameGroupConfig:
    """
    Configuration of a group of (possibly masked) frames, one after the other.
    All frames in a group are based off of the same basic frame, and differ only
    through precoding masking.
    """
    group_id          : str                 # Mask id/hash for identification
    mask_id           : str                 # Identifier/hash of mask group
    mask_group        : np.ndarray          # A [num_mask, num_subcarrier] array of masks
    mask_name         : str                 # Descriptive name
    base_frame        : BaseFrameConfig     # Base frame (unmodified)
    group_repetitions : int                 # Number of times to repeat masked-frame-group for final sample file
    interframe_delay  : FrameDelay          # Number of interframe zero-padding samples


@dataclass(frozen=True)
class GeneratedFrameInfo(FrameGroupConfig):
    """
    Struct to maintain information on a generated frame, i.e. one for which a
    file with IQ-samples exist
    """
    frame_file : Path                       # Full path to generated file
    created_at : datetime                   # Time of creation
    file_size  : int                        # Size of the associated IQ-sample file


#######################################################################################
## Channel configuration, i.e. parameters dictating where the frame is sent
#######################################################################################
@dataclass
class Channel:
    """
    Description of a WiFi channel
    """
    frequency      : int                       # Center frequency in Hz
    channel_number : int                       # WiFi channel number
    channel_spec   : str                       # Channel Specification (e.g. HT20)
    bandwidth_mhz  : int = 20                  # Bandwidth in MHz (defaulting to 20, nothing else supported rn)
    sideband       : Sideband = Sideband.NONE  # For 2.4 GHz 40 MHz channels, we need to specify sideband

    def __post_init__(self):
        if self.sideband != Sideband.NONE:
            assert self.frequency < 2_500_000_000, "Sideband only relevant in 2.4 GHz band"
            assert self.bandwidth_mhz == 40, "Sidebands only exist for bonded 40 MHz channels"

        # Defaulting to use at least some sideband for ease use. We are not interested in this,
        # this is just required to configure the network interfaces properly
        if self.bandwidth_mhz == 40 and self.frequency < 2_500_000_000:
            if self.sideband == Sideband.NONE:
                self.sideband = Sideband.UPPER


#######################################################################################
## Experiment Configuration options
#######################################################################################
@dataclass
class SessionConfig:
    """
    Configuration of a single sensing session
    """
    name          : str                       # Some name for this session for clarity
    channel       : Channel                   # Channel to use
    receivers     : List[str]                 # Short names of receivers participating in session
    frame_group   : FrameGroupConfig          # Frame to send out in this session
    transmitter   : List[str]                 # Name of transmitter to use
    tx_gain       : int               = 5     # Gain of transmitter
    n_repeat      : int               = 1     # Number of transmitter repetitions
    training_reps : int               = 1000  # Number of times to send unmodified frames for equalization training
    session_id    : str               = ""    # ID (automatically generated)

@dataclass
class ReadySessionConfig:
    """
    Extended session configuration for after everything has been set up for it to be
    ready to run
    """
    name             : str                             # Some name for this session for clarity
    channel          : Channel                         # Channel to use
    receivers        : List[Receiver]                  # Short names of receivers participating in session
    transmitter      : List["Transmitter"]             # type: ignore    # Forward ref, Name of transmitter to use
    tools            : Dict[CSICaptureTool, "CsiTool"] # type: ignore    # Forward ref, Tools to be used with receivers
    frame_info       : Optional[GeneratedFrameInfo]    # Config of generated frame
    train_frame_info : Optional[GeneratedFrameInfo]    # Frame info of frame used for equalization training at beginning of session
    cache_dir        : Path                            # A temp session dir to e.g. put data in
    tx_gain          : int                             # Gain of transmitter
    n_repeat         : int                             # Number of transmitter repetitions
    training_reps    : int                             # Number of times to send unmodified frames for equalization training
    session_id       : str                             # Identifier for this session

@dataclass
class ExperimentConfig:
    """
    Configuration of the full experiment
    """
    exp_id            : str                 # Identifier for the whole experiment
    database_path     : Path                # File path for database file
    sessions          : List[SessionConfig] # List of sessions to run
    cache_dir         : Path                # Directory of cache
    matlab_batch_size : int = 10            # Number of frames to generate in parallel during generation time
    cache_trim        : int = int(1e10)     # Byte size after which to trim frame cache to avoid blow up
    performed_at      : datetime = datetime.now() # Time this experiment was started
    inter_session_delay_seconds : int = 0         # Delay between sessions in seconds

    def __post_init__(self):
        self.database_path = Path(self.database_path)
# fmt: on


#######################################################################################
## Convenience helpers
#######################################################################################
def get_pretty_config(config) -> str:
    """
    Convert dataclass config to a pretty printed json object

    Args:
        config : Dataclass object to convert
    """
    return json.dumps(asdict(config), indent=4, default=vars)


def _load_config(config_file_path: Path, config_type: Type):
    """
    Parse config from toml file into dataclass type

    Args:
        config_file_path : Path to configuration file
        config_type      : Type to parse dict into
    """

    if not config_file_path.is_file():
        raise FileNotFoundError(
            f"Config file {config_file_path} not found; Check path again!"
        )

    if not config_file_path.suffix == ".toml":
        raise FileNotFoundError(
            f"Config file {config_file_path} is not toml; Wrong format."
        )

    with open(config_file_path, "rb") as f:
        data = tomllib.load(f)
        config = config_type(**data)

    return config


def load_hardware_setup(
    config_file_path: Path = Path.cwd() / "setup.toml",
) -> EnvironmentConfig:
    """
    Loads config from filepath

    Args:
        config_file_path : Path to configuration file
    """
    return _load_config(config_file_path, config_type=EnvironmentConfig)
