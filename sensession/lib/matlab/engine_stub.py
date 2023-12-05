"""
Matlab stub

Possibly can be used to perform unittests or something in CI without
Matlab installed

Might be better of just using some mocking capabilities. No idea how
nice these are in python, coming from C++. In case its easy, remove
this file.
"""

# pylint: skip-file
import numpy as np


class MockFrameGenResult:
    def __init__(self):
        pass

    def result(self):
        pass


class MockEngine:
    def __init__(self):
        pass

    def addpath(self, path: str):
        assert isinstance(path, str), "Non-string path encountered"

    def quit(self):
        pass

    def generate_csimasked_frame(
        self,
        frame_file: str,
        rescale_factor: int,
        receiver_address: str,
        transmitter_address: str,
        bssid_address: str,
        bandwidth_mhz: int,
        group_repetitions: int,
        enable_sounding: bool,
        mask_group: np.ndarray,
        padding: int,
        guard_iv_mode: int,
        vht: bool,
        rate_mhz: int,
        nargout: int,
        background: bool,
    ):
        assert isinstance(frame_file, str), "Frame file must be str"
        assert isinstance(rescale_factor, int), "Rescale factor must be int"
        assert isinstance(receiver_address, str), "receiver address must be str"
        assert isinstance(transmitter_address, str), "transmitter address must be str"
        assert isinstance(bssid_address, str), "bssid address must be str"
        assert isinstance(bandwidth_mhz, int), "Bandwidth must be int"
        assert isinstance(group_repetitions, int), "Group reps must be int"
        assert isinstance(enable_sounding, bool), "Sounding is a bool flag"
        assert isinstance(mask_group, np.ndarray), "Mask must be an array"
        assert isinstance(padding, int), "Padding must be int"
        assert isinstance(guard_iv_mode, int), "Padding must be int"
        assert isinstance(vht, bool), "vht must be bool"
        assert isinstance(rate_mhz, int), "rate must be int"
        assert isinstance(nargout, int), "nargout must be int"
        assert isinstance(background, bool), "background must be bool"

    def read_csi(self, file: str, chipset: str, bandwidth: int, nargout: int):
        assert isinstance(file, str), "File must be str"
        assert isinstance(chipset, str), "chipset must be str"
        assert isinstance(bandwidth, int), "Bandwidth must be int"
        assert isinstance(nargout, int), "nargout must be int"

        return [], [], [], [], []


class matlabengine:
    def __init__(self):
        pass

    @staticmethod
    def start_matlab() -> MockEngine:
        return MockEngine()
