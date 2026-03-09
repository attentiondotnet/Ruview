"""
CSI data models.

CSI (Channel State Information) describes how a WiFi signal propagates
from transmitter to receiver across all subcarriers.  On an ESP32 running
802.11n in 20 MHz mode the chip reports 52 subcarrier pairs per received
packet.  Each pair is a (imaginary, real) int8 that encodes the complex
channel coefficient H_k = A_k · e^{jφ_k}.
"""

from __future__ import annotations

import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Default number of OFDM subcarriers in 802.11n 20 MHz mode
NUM_SUBCARRIERS = 52


@dataclass
class CSIFrame:
    """One CSI measurement snapshot captured from a single WiFi packet.

    Attributes
    ----------
    timestamp:
        Unix epoch seconds when the frame was received.
    node_id:
        Identifier of the edge node that captured the frame.
    amplitude:
        Per-subcarrier signal amplitude |H_k|, shape ``(num_subcarriers,)``.
    phase:
        Per-subcarrier unwrapped phase ∠H_k (radians), shape ``(num_subcarriers,)``.
    rssi:
        Received signal strength indicator in dBm.
    noise_floor:
        Estimated noise floor in dBm.
    channel:
        WiFi channel number (1–13 for 2.4 GHz).
    """

    timestamp: float = field(default_factory=time.time)
    node_id: str = ""
    amplitude: np.ndarray = field(default_factory=lambda: np.zeros(NUM_SUBCARRIERS))
    phase: np.ndarray = field(default_factory=lambda: np.zeros(NUM_SUBCARRIERS))
    rssi: int = 0
    noise_floor: int = -95
    channel: int = 6

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def complex_csi(self) -> np.ndarray:
        """Return CSI as complex phasors: A_k · e^{jφ_k}."""
        return self.amplitude * np.exp(1j * self.phase)

    @property
    def num_subcarriers(self) -> int:
        return len(self.amplitude)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_esp32_bytes(cls, raw: bytes, node_id: str = "", channel: int = 6) -> "CSIFrame":
        """Parse the raw byte buffer produced by the ESP32 CSI callback.

        The ESP32 ``wifi_csi_info_t.buf`` stores alternating int8 pairs
        ``[imag_0, real_0, imag_1, real_1, …]`` for every subcarrier.

        Parameters
        ----------
        raw:
            Byte string of length ``2 * num_subcarriers`` (int8 pairs).
        node_id:
            Identifier tag for the originating sensor node.
        channel:
            WiFi channel the frame was captured on.
        """
        if len(raw) % 2 != 0:
            raise ValueError("raw CSI buffer must have an even number of bytes")

        pairs = np.frombuffer(raw, dtype=np.int8).reshape(-1, 2).astype(np.float32)
        imag_parts = pairs[:, 0]
        real_parts = pairs[:, 1]

        amplitude = np.sqrt(real_parts**2 + imag_parts**2)
        phase = np.arctan2(imag_parts, real_parts)

        return cls(
            node_id=node_id,
            amplitude=amplitude,
            phase=phase,
            channel=channel,
        )

    @classmethod
    def from_udp_packet(cls, data: bytes) -> "CSIFrame":
        """Deserialise a UDP packet sent by the ESP32 firmware.

        Wire format (little-endian)::

            [ node_id : 8 bytes (null-padded ASCII) ]
            [ timestamp : 8 bytes  (double)         ]
            [ rssi      : 1 byte   (int8)            ]
            [ channel   : 1 byte   (uint8)           ]
            [ n_sub     : 2 bytes  (uint16)          ]
            [ csi_buf   : 2*n_sub bytes (int8 pairs) ]
        """
        header_fmt = "<8sdbbH"
        header_size = struct.calcsize(header_fmt)
        node_id_raw, timestamp, rssi, channel, n_sub = struct.unpack_from(
            header_fmt, data, 0
        )
        node_id = node_id_raw.rstrip(b"\x00").decode("ascii", errors="replace")
        csi_raw = data[header_size : header_size + 2 * n_sub]
        frame = cls.from_esp32_bytes(csi_raw, node_id=node_id, channel=channel)
        frame.timestamp = timestamp
        frame.rssi = rssi
        return frame

    def __len__(self) -> int:
        return self.num_subcarriers

    def __repr__(self) -> str:
        return (
            f"CSIFrame(node={self.node_id!r}, t={self.timestamp:.3f}, "
            f"rssi={self.rssi}, subs={self.num_subcarriers})"
        )


class CSIBuffer:
    """Rolling time-window buffer of :class:`CSIFrame` objects.

    Parameters
    ----------
    max_frames:
        Maximum number of frames to retain.  Oldest frames are discarded
        automatically once the buffer is full.
    """

    def __init__(self, max_frames: int = 500) -> None:
        self.max_frames = max_frames
        self._frames: deque[CSIFrame] = deque(maxlen=max_frames)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, frame: CSIFrame) -> None:
        """Append a new frame, evicting the oldest when the buffer is full."""
        self._frames.append(frame)

    def clear(self) -> None:
        """Remove all frames from the buffer."""
        self._frames.clear()

    # ------------------------------------------------------------------
    # Bulk accessors
    # ------------------------------------------------------------------

    @property
    def frames(self) -> list[CSIFrame]:
        return list(self._frames)

    @property
    def amplitudes(self) -> np.ndarray:
        """Amplitude matrix of shape ``(T, num_subcarriers)``."""
        if not self._frames:
            return np.empty((0, NUM_SUBCARRIERS))
        return np.stack([f.amplitude for f in self._frames])

    @property
    def phases(self) -> np.ndarray:
        """Phase matrix of shape ``(T, num_subcarriers)``."""
        if not self._frames:
            return np.empty((0, NUM_SUBCARRIERS))
        return np.stack([f.phase for f in self._frames])

    @property
    def timestamps(self) -> np.ndarray:
        """1-D array of frame timestamps."""
        return np.array([f.timestamp for f in self._frames])

    @property
    def sample_rate(self) -> Optional[float]:
        """Estimated sample rate in Hz, or *None* if fewer than 2 frames."""
        ts = self.timestamps
        if len(ts) < 2:
            return None
        dt = np.diff(ts)
        return float(1.0 / np.median(dt[dt > 0])) if np.any(dt > 0) else None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._frames)

    def __repr__(self) -> str:
        return f"CSIBuffer(frames={len(self)}, max={self.max_frames})"
