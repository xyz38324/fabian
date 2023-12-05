"""
Generation of WiFi Frames, specifically down to the waveform (IQ-samples)
"""

import shelve
import hashlib
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
from loguru import logger

from sensession.lib.config import (
    DataRateMode,
    BaseFrameConfig,
    FrameGroupConfig,
    GeneratedFrameInfo,
    get_pretty_config,
)

TRIVIAL_MASK_NAME: str = "_unmodified"


# -------------------------------------------------------------------------------------
# Helper functions for frame generation
# Can be used to invoke matlab scripts (employing the WLan Toolbox) to generate the
# frame. Matlabengine is used to process the frames.
# -------------------------------------------------------------------------------------
def matlab_to_numpy_array(arr, dtype=None) -> np.ndarray:
    """
    Convert matlab arrays to numpy native ones
    """
    kwargs = {"dtype": dtype} if dtype else {}
    return np.squeeze(np.asarray(arr, **kwargs).reshape(arr.size, order="F"))


def get_framegroup_hash(
    config: BaseFrameConfig, mask_group: np.ndarray, group_repetitions: int
) -> str:
    """
    Create a hash for a framegroup for identification.

    Args:
        config            : Config of group base frame
        mask_group        : Mask used in the group
        group_repetitions : How often the group is repeated
    """
    hash_id = hashlib.md5()
    mask_id = hashlib.sha256(mask_group.data)
    hash_id.update(
        repr(config).encode("utf-8")
        + mask_id.hexdigest().encode("utf-8")
        + str(group_repetitions).encode("utf-8")
    )
    return hash_id.hexdigest()


def get_mask_hash(mask: np.ndarray) -> str:
    """
    Get a hash for a mask

    Args:
        mask : Mask to generate hash for
    """
    hash_id = hashlib.sha256(mask.data)
    return hash_id.hexdigest()


def create_fg_config(
    base_frame: BaseFrameConfig,
    mask_group: np.ndarray,
    group_repetitions: int,
    mask_name: str = "",
    interframe_delay: "int | timedelta" = 30000,
) -> FrameGroupConfig:
    """
    Construction helper for frame group to automatically set unique IDs
    """
    return FrameGroupConfig(
        group_id=get_framegroup_hash(base_frame, mask_group, group_repetitions),
        mask_id=get_mask_hash(mask_group),
        mask_group=mask_group,
        mask_name=mask_name,
        base_frame=base_frame,
        group_repetitions=group_repetitions,
        interframe_delay=interframe_delay,
    )


def to_training(config: FrameGroupConfig) -> FrameGroupConfig:
    """
    Convert a frame group config to a training frame group, i.e. one with trivial
    mask that can be used for training phases.
    """
    mask_group = np.ones((config.mask_group.shape[0], 1), dtype=np.complex64)
    return create_fg_config(
        base_frame=config.base_frame,
        mask_group=mask_group,
        group_repetitions=1,
        mask_name=TRIVIAL_MASK_NAME,
        interframe_delay=config.interframe_delay,
    )


# -------------------------------------------------------------------------------------
# Frame Cache
# -------------------------------------------------------------------------------------
class FrameCache:
    """
    Frame Cache to manage storing frames in a cache directory for reuse
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._cache_file = cache_dir / "frame_map.shelve"
        self._cached_frames = self._read_from_shelve()

    def __del__(self):
        """
        Destroy frame cache object, persist cached frames.
        """
        self._write_to_shelve()

    def get_cached_bytesize(self) -> int:
        """
        Get size of cached frame files in bytes
        """
        return sum(frame.file_size for frame in self._cached_frames.values())

    def trim_cache(self, max_bytesize: int):
        """
        Delete the oldest frames to clear up to cache

        Args:
            max_bytesize : Maximum number of bytesize before cache should be trimmed
        """
        current_size = self.get_cached_bytesize()

        # If nothing needs to be removed, stop here.
        if current_size < max_bytesize:
            return

        # Calculate how much to remove
        tbremoved = current_size - max_bytesize
        logger.info(f"Trimming cache of {tbremoved} bytes ...")

        # Sort to dictate which frames are removed first
        # We sort by two criteria:
        #  - Primarily, we want "unmodified" frames to be removed last, because they are
        #    reused at the beginning of every session as training frames for shape
        #    equalization
        #  - Secondarily, we sort by time: Want to remove the oldest frames first.
        time_sorted_frames = dict(
            sorted(
                self._cached_frames.items(),
                key=lambda item: (
                    0 if item[1].mask_name != TRIVIAL_MASK_NAME else 1,
                    item[1].created_at,
                ),
            )
        )

        # Remove until enough space is freed
        for frame_id, frame_info in time_sorted_frames.items():
            tbremoved -= frame_info.file_size
            self.remove_cached_file(frame_id)

            logger.debug(
                "Cleared cache of frame file: \n"
                + f" -- id         : {frame_id}\n"
                + f" -- created at : {frame_info.created_at}\n"
                + f" -- mask name  : {frame_info.mask_name}\n"
                + f" -- mask id    : {frame_info.mask_id}\n"
                + f" -- group id   : {frame_info.group_id}\n"
            )

            if tbremoved < 0:
                break

        # We are sure something was removed. To be safe, lets just write the
        # shelve file here as well.
        self._write_to_shelve()

    def get_cached_frame(
        self, frame: "FrameGroupConfig | None"
    ) -> Optional[GeneratedFrameInfo]:
        """
        Retrieve cached frame value, if any.

        Args:
            frame : config of frame group

        Returns:
            Generated Frame Info if frame is present in cache, otherwise None.
        """
        if not frame:
            return None

        digest = frame.group_id

        if digest in self._cached_frames:
            # NOTE: Important here is to understand that we cache frames based on a base frame
            # configuration (MAC address etc.) and masks.
            # Metaparameters, like repetitions, are basically _ignored_ when building hashes and
            # thus irrelevant in caching.
            # We therefore instead merge these into the cached frame from the specified frame
            cached_frame = self._cached_frames[digest]
            return GeneratedFrameInfo(
                group_id=digest,
                base_frame=cached_frame.base_frame,
                mask_group=cached_frame.mask_group,
                mask_id=frame.mask_id,
                mask_name=frame.mask_name,
                group_repetitions=frame.group_repetitions,
                frame_file=cached_frame.frame_file,
                created_at=cached_frame.created_at,
                file_size=cached_frame.file_size,
                interframe_delay=frame.interframe_delay,
            )

        return None

    def is_cached(self, frame: FrameGroupConfig) -> bool:
        """
        Check whether frame is cached or not.

        Args:
            frame : config of frame group
        """
        digest = frame.group_id
        return digest in self._cached_frames

    def register_cached_file(self, frame_info: GeneratedFrameInfo):
        """
        Register frame as cached

        Args:
            frame_info : Frame info struct of generated frame
        """
        digest = frame_info.group_id
        self._cached_frames[digest] = frame_info

    def remove_cached_file(self, frame_hash: str):
        """
        Remove cached file

        Args:
            frame_hash : Hash of frame to remove
        """
        frame_info = self._cached_frames[frame_hash]
        frame_info.frame_file.unlink()
        del self._cached_frames[frame_hash]

    def _read_from_shelve(self) -> Dict[str, GeneratedFrameInfo]:
        """
        Read dictionary values from a shelve.
        """
        with shelve.open(str(self._cache_file)) as db:
            return dict(db)

    def _write_to_shelve(self):
        """
        Write info of generated frames to a shelve.

        NOTE: Will overwrite previous entries, if existent.
        """
        if not all(
            frame_info.frame_file.is_file()
            for frame_info in self._cached_frames.values()
        ):
            raise RuntimeError("configs added to shelf must contain valid filepaths!")

        with shelve.open(str(self._cache_file)) as db:
            db.clear()
            for frame_hash, frame_info in self._cached_frames.items():
                db[frame_hash] = frame_info


# -------------------------------------------------------------------------------------
# Frame generation helper functions
# -------------------------------------------------------------------------------------
def generate_frame_group(
    eng, config: FrameGroupConfig, frame_file: Path
) -> GeneratedFrameInfo:
    """
    Generate an IQ-sample file for a group of WiFi frames

    Args:
        eng     : matlab engine object
        config  : Configuration of frame group to generate
    """
    if not frame_file:
        raise ValueError("Must specify frame file path")

    # Matlab requires format without colons
    frame_config = config.base_frame
    receiver_address = frame_config.receiver_address.replace(":", "")
    transmitter_address = frame_config.transmitter_address.replace(":", "")
    bssid_address = frame_config.bssid_address.replace(":", "")

    logger.trace(
        "Generating frame with configuration: \n"
        + f" -- group id          : {config.group_id}\n"
        + f" -- mask name         : {config.mask_name}\n"
        + f" -- mask shape        : {config.mask_group.shape}\n"
        + f" -- base frame config : {get_pretty_config(frame_config)}\n"
    )

    # Extract Bandwidth in MHz. In case it is specified in Hz, just rescale.
    bandwidth_mhz = frame_config.bandwidth
    if bandwidth_mhz > 1e6:
        bandwidth_mhz //= int(1e6)

    rate_mhz = frame_config.send_rate
    if rate_mhz > 1e6:
        rate_mhz //= int(1e6)

    vht = frame_config.data_rate_mode == DataRateMode.VERY_HIGH_THROUGHPUT
    assert (frame_config.data_rate_mode == DataRateMode.VERY_HIGH_THROUGHPUT) or (
        frame_config.data_rate_mode == DataRateMode.HIGH_THROUGHPUT
    ), "Only HT and VHT supported for frame generation currently.."

    # -------------------------------------------------------------------------
    # Extract interframe padding from delay
    padding = config.interframe_delay
    if isinstance(padding, timedelta):
        # Extract time delta in microsecond
        time_micros = padding / timedelta(microseconds=1)

        # microsecond: 10e-6 seconds
        # megahertz:   10e6  hertz
        num_cmplx_samples = time_micros * bandwidth_mhz
        padding = int(2 * num_cmplx_samples)
    elif not isinstance(padding, int):
        raise ValueError(
            "interframe delay must be either int (num of samples) or timedelta (time)!"
        )

    # -------------------------------------------------------------------------
    # Generate masked frames
    ret = eng.generate_csimasked_frame(
        str(frame_file),
        frame_config.rescale_factor,
        receiver_address,
        transmitter_address,
        bssid_address,
        bandwidth_mhz,
        config.group_repetitions,
        frame_config.enable_sounding,
        config.mask_group,
        padding,
        frame_config.guard_iv_mode.value,
        vht,
        rate_mhz,
        nargout=0,
        background=True,
    )
    ret.result()

    # Ensure frame file actually exists and contains samples as sanity check
    file_size = frame_file.stat().st_size
    if not frame_file.is_file() or file_size == 0:
        raise RuntimeError(
            f"Frame Generation failed?! - {frame_file} is not populated!"
        )

    return GeneratedFrameInfo(
        **config.__dict__,
        frame_file=frame_file,
        created_at=datetime.now(),
        file_size=file_size,
    )
