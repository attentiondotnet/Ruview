"""
Transport receivers that pull raw CSI data from edge nodes.

Two transport backends are provided:

* :class:`UDPReceiver` — listens on a UDP socket for packets sent by the
  ESP32 firmware.  Suitable for LAN deployments.
* :class:`SerialReceiver` — reads from a serial port (USB-to-UART).
  Suitable for direct cable connections during development.

Both receivers run in a background thread and call
:meth:`~ruview.edge.node.EdgeNode.ingest_udp_packet` (or
:meth:`~ruview.edge.node.EdgeNode.ingest_raw` for serial) with each new
packet.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
from typing import Callable

from ruview.edge.node import EdgeNode

logger = logging.getLogger(__name__)


class UDPReceiver:
    """Receive CSI packets from ESP32 nodes over UDP.

    Parameters
    ----------
    host:
        IP address to bind to.  Use ``'0.0.0.0'`` to accept from any
        interface.
    port:
        UDP port to listen on.
    nodes:
        Mapping of ``node_id → EdgeNode`` to route incoming packets to.
        If the node_id embedded in a packet is not found, a new
        :class:`~ruview.edge.node.EdgeNode` is created automatically and
        added to this dict.
    buffer_size:
        Maximum UDP datagram size in bytes.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        nodes: dict[str, EdgeNode] | None = None,
        buffer_size: int = 4096,
    ) -> None:
        self.host = host
        self.port = port
        self.nodes: dict[str, EdgeNode] = nodes or {}
        self.buffer_size = buffer_size

        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind the UDP socket and start the receiver thread."""
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="udp-receiver")
        self._thread.start()
        logger.info("UDPReceiver listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        """Stop the receiver thread and close the socket."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        logger.info("UDPReceiver stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(self.buffer_size)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._dispatch(data)
            except Exception:
                logger.exception("Error dispatching packet from %s", addr)

    def _dispatch(self, data: bytes) -> None:
        from ruview.csi.models import CSIFrame

        frame = CSIFrame.from_udp_packet(data)
        node_id = frame.node_id
        if node_id not in self.nodes:
            logger.info("Auto-registering new node %r", node_id)
            self.nodes[node_id] = EdgeNode(node_id=node_id)
        self.nodes[node_id].ingest(frame)

    def __enter__(self) -> "UDPReceiver":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


class SerialReceiver:
    """Receive CSI data from an ESP32 connected via a serial port.

    The expected wire format over serial is a simple length-prefixed frame::

        [ 0xAA  : 1 byte  (sync byte)   ]
        [ length: 2 bytes (uint16 LE)    ]
        [ payload: length bytes          ]
        [ checksum: 1 byte (XOR of payload) ]

    Parameters
    ----------
    port:
        Serial device path, e.g. ``'/dev/ttyUSB0'`` or ``'COM3'``.
    baudrate:
        Serial baud rate (must match the firmware configuration).
    node:
        The :class:`~ruview.edge.node.EdgeNode` to forward frames to.
    """

    SYNC_BYTE = 0xAA

    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        node: EdgeNode | None = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.node = node or EdgeNode(node_id=port)

        self._thread: threading.Thread | None = None
        self._running = False
        self._ser = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the serial port and start the receiver thread."""
        try:
            import serial  # pyserial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for SerialReceiver. "
                "Install it with: pip install pyserial"
            ) from exc

        if self._running:
            return
        self._ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=1.0)
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="serial-receiver"
        )
        self._thread.start()
        logger.info("SerialReceiver on %s @ %d baud", self.port, self.baudrate)

    def stop(self) -> None:
        """Stop the receiver thread and close the serial port."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            try:
                sync = self._ser.read(1)
                if not sync or sync[0] != self.SYNC_BYTE:
                    continue
                length_bytes = self._ser.read(2)
                if len(length_bytes) < 2:
                    continue
                (length,) = struct.unpack_from("<H", length_bytes)
                payload = self._ser.read(length)
                if len(payload) < length:
                    continue
                checksum_byte = self._ser.read(1)
                if not checksum_byte:
                    continue
                expected_cs = 0
                for b in payload:
                    expected_cs ^= b
                if checksum_byte[0] != expected_cs:
                    logger.debug("Checksum mismatch — dropping frame")
                    continue
                self.node.ingest_raw(payload)
            except Exception:
                logger.exception("Error in serial receiver loop")

    def __enter__(self) -> "SerialReceiver":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
