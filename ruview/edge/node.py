"""
Edge node abstraction.

An *edge node* is a single ESP32 (or compatible) sensor that continuously
captures CSI data and streams it to the host over UDP or serial.

This module provides:

* :class:`NodeStatus` — health/connectivity status enum.
* :class:`EdgeNode` — thin wrapper around a transport that converts raw
  packets into :class:`~ruview.csi.models.CSIFrame` objects and feeds them
  into a :class:`~ruview.csi.models.CSIBuffer`.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional

from ruview.csi.models import CSIBuffer, CSIFrame

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Connectivity / health state of an edge node."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    ACTIVE = auto()
    ERROR = auto()


class EdgeNode:
    """Manages a single CSI sensor node.

    Parameters
    ----------
    node_id:
        Human-readable label (e.g. ``'node-01'``).
    buffer_size:
        Maximum number of CSI frames to retain in the rolling buffer.
    stale_timeout_s:
        Seconds since last received frame before the node is marked as
        ``DISCONNECTED``.
    on_frame:
        Optional callback invoked with each new :class:`~ruview.csi.models.CSIFrame`
        immediately after it is added to the buffer.
    """

    def __init__(
        self,
        node_id: str,
        buffer_size: int = 500,
        stale_timeout_s: float = 5.0,
        on_frame: Optional[Callable[[CSIFrame], None]] = None,
    ) -> None:
        self.node_id = node_id
        self.stale_timeout_s = stale_timeout_s
        self.on_frame = on_frame

        self.buffer = CSIBuffer(max_frames=buffer_size)
        self._status = NodeStatus.DISCONNECTED
        self._last_frame_time: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------

    def ingest(self, frame: CSIFrame) -> None:
        """Add a new CSI frame to the node's buffer.

        Thread-safe.  Updates :attr:`status` to :attr:`NodeStatus.ACTIVE`.
        """
        with self._lock:
            self.buffer.add(frame)
            self._last_frame_time = time.monotonic()
            self._status = NodeStatus.ACTIVE

        if self.on_frame is not None:
            try:
                self.on_frame(frame)
            except Exception:
                logger.exception("Error in on_frame callback for node %s", self.node_id)

    def ingest_raw(self, raw: bytes) -> None:
        """Parse a raw ESP32 CSI byte string and ingest the resulting frame."""
        frame = CSIFrame.from_esp32_bytes(raw, node_id=self.node_id)
        self.ingest(frame)

    def ingest_udp_packet(self, data: bytes) -> None:
        """Parse and ingest a full UDP packet produced by the ESP32 firmware."""
        frame = CSIFrame.from_udp_packet(data)
        self.ingest(frame)

    # ------------------------------------------------------------------
    # Status / health
    # ------------------------------------------------------------------

    @property
    def status(self) -> NodeStatus:
        """Current connectivity status.  Updated lazily on property access."""
        if self._status is NodeStatus.ACTIVE and self._last_frame_time is not None:
            elapsed = time.monotonic() - self._last_frame_time
            if elapsed > self.stale_timeout_s:
                self._status = NodeStatus.DISCONNECTED
        return self._status

    @status.setter
    def status(self, value: NodeStatus) -> None:
        self._status = value

    def is_active(self) -> bool:
        """Return ``True`` if the node is currently delivering CSI data."""
        return self.status is NodeStatus.ACTIVE

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EdgeNode(id={self.node_id!r}, status={self.status.name}, "
            f"frames={len(self.buffer)})"
        )
