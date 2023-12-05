"""
NOTE: Currently not integrated in lib.
TODO: Integrate! also fix this stuff. And of course implement streaming files back from Oscilloscope.
"""

import time
import socket
import struct


def send(sock, text: str):
    # Ensure endline termination character is streamed, encode into little-endian
    # bytearray and send over socket
    text = text + "\n"
    encoded = text.encode()
    packed_bytes = struct.pack(f"<{len(encoded)}s", encoded)
    sock.sendall(packed_bytes)


def receive(sock, size=8 * 1024):
    data = sock.recv(size)
    unpacked_data = struct.unpack(f"<{len(data)}s", data)

    # Unpack always returns a tuple, extract the first and only value
    return unpacked_data[0].decode()


class KeysightOsci:
    """
    Class to represent the keysight oscilloscope
    """

    def __init__(self):
        self.ip = "169.254.152.121"
        self.port = 5025

        # fmt: off
        self.sample_rate  = "250e6"     # 250 MS/s
        self.y_scale      = "5e-3"      # 5 mV
        self.y_max        = "80e-3"     # 80 mV
        self.trig_voltage = "12e-3"     # 10 mV
        self.trig_prepend = "-200e-6"   # capture 200 microseconds before trigger
        self.capt_width   = "10e-3"     # capture window width of 10 miliseconds
        self.channel      = "4"         # Input channel
        # fmt: on

    def capture_once(self, output_file: str = ""):
        """
        Set up oscilloscope with trigger to capture once

        Args:
            output_file : File to save to, path must be valid on oscilloscope
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP) as s:
            s.connect((self.ip, self.port))

            # Turn off system header
            send(s, ":SYSTem:HEADer OFF")

            # Set capture resolution, sample rate and turn off interpolations for
            # faithful real-time capture.
            # realtime high resolution
            send(s, ":ACQuire:MODE HRESolution")
            send(s, ":ACQuire:HRESolution BITS12")
            send(s, ":ACQuire:POINts AUTO")
            send(s, ":ACQuire:INTerpolate OFF")
            send(s, f":ACQuire:SRATe {self.sample_rate}")

            # Set y-axis scales
            send(s, f":CHANnel{self.channel}:SCALe {self.y_scale}")
            send(s, f":CHANnel{self.channel}:RANGe {self.y_max}")

            # x axis
            # Position : How far before the trigger buffer starts
            # Range    : width of capture starting from trigger
            send(s, f":TIMebase:POSition {self.trig_prepend}")
            send(s, f":TIMebase:RANGe {self.capt_width}")
            send(s, ":TIMebase:REFerence LEFT")

            # trigger new
            send(s, f":TRIGger:LEVel CHANnel{self.channel},{self.trig_voltage}")
            send(s, ":TRIGger:MODE EDGE")

            # clear status and capture
            send(s, "*CLS")
            send(s, ":SINGle")

            while True:
                send(s, ":PDER?")

                # Will return a +1 code once finished.
                code = int(receive(s))
                if code == 1:
                    break

                time.sleep(0.2)

            send(s, f":WAVeform:SOURce CHANnel{self.channel}")
            send(s, ":WAVeform:VIEW ALL")

            if output_file:
                path = r"C:\Users\Public"
                send(s, f':DISK:CDIRectory "{path}"')
                send(s, f':DISK:SAVE:WAVeform ALL,"{output_file}",MATlab')
                time.sleep(1)


if __name__ == "__main__":
    scope = KeysightOsci()
    scope.capture_once(output_file="csi_test")
