"""
RuView main perception engine.

The :class:`RuViewEngine` ties together all subsystems:

* :class:`~ruview.edge.node.EdgeNode` objects that receive CSI from sensors.
* :class:`~ruview.csi.processor.CSIProcessor` for preprocessing.
* :class:`~ruview.presence.detector.PresenceDetector` for occupancy detection.
* :class:`~ruview.vitals.breathing.BreathingMonitor` for breathing rate.
* :class:`~ruview.vitals.heart_rate.HeartRateMonitor` for heart rate.
* :class:`~ruview.pose.estimator.PoseEstimator` for body-pose reconstruction.

Quick start
-----------
>>> from ruview import RuViewEngine
>>> from ruview.edge import EdgeNode
>>> node = EdgeNode(node_id="esp32-01")
>>> engine = RuViewEngine(nodes=[node])
>>> engine.start()
>>> # inject a frame from a live sensor or from recorded data:
>>> node.ingest(frame)
>>> result = engine.observe()
>>> print(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ruview.csi.models import CSIBuffer
from ruview.edge.node import EdgeNode
from ruview.pose.estimator import PoseEstimator, PoseResult
from ruview.presence.detector import PresenceDetector, PresenceResult
from ruview.vitals.breathing import BreathingMonitor, BreathingResult
from ruview.vitals.heart_rate import HeartRateMonitor, HeartRateResult

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """A single snapshot of the perceived environment.

    Attributes
    ----------
    presence:
        Human presence detection result.
    breathing:
        Breathing rate estimation.
    heart_rate:
        Heart rate estimation.
    pose:
        Body pose estimation (keypoints).
    node_ids:
        Identifiers of the nodes whose data contributed to this observation.
    """

    presence: PresenceResult
    breathing: BreathingResult
    heart_rate: HeartRateResult
    pose: PoseResult
    node_ids: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        br = (
            f"{self.breathing.rate_bpm:.1f} BPM"
            if self.breathing.rate_bpm is not None
            else "—"
        )
        hr = (
            f"{self.heart_rate.rate_bpm:.1f} BPM"
            if self.heart_rate.rate_bpm is not None
            else "—"
        )
        return (
            f"Observation("
            f"present={self.presence.present}, "
            f"conf={self.presence.confidence:.2f}, "
            f"breathing={br}, "
            f"hr={hr}, "
            f"pose_conf={self.pose.overall_confidence:.2f})"
        )


class RuViewEngine:
    """Orchestrates the full RuView perception pipeline.

    Parameters
    ----------
    nodes:
        List of :class:`~ruview.edge.node.EdgeNode` objects to aggregate.
        If omitted, a single in-memory node ``'default'`` is created.
    presence_threshold:
        Variance-ratio threshold passed to :class:`~ruview.presence.detector.PresenceDetector`.
    """

    def __init__(
        self,
        nodes: Optional[list[EdgeNode]] = None,
        presence_threshold: float = 2.5,
    ) -> None:
        if nodes:
            self.nodes: dict[str, EdgeNode] = {n.node_id: n for n in nodes}
        else:
            default_node = EdgeNode(node_id="default")
            self.nodes = {"default": default_node}

        self._presence = PresenceDetector(threshold=presence_threshold)
        self._breathing = BreathingMonitor()
        self._heart_rate = HeartRateMonitor()
        self._pose = PoseEstimator()
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the engine as running (receivers should be started separately)."""
        self._running = True
        logger.info("RuViewEngine started with %d node(s)", len(self.nodes))

    def stop(self) -> None:
        """Stop the engine."""
        self._running = False
        logger.info("RuViewEngine stopped")

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def observe(self, node_id: Optional[str] = None) -> Observation:
        """Run all perception algorithms and return an :class:`Observation`.

        Parameters
        ----------
        node_id:
            If given, only use data from this specific node.
            Otherwise all active nodes are merged into a single buffer.

        Returns
        -------
        Observation
        """
        buffer, used_nodes = self._get_buffer(node_id)

        presence = self._presence.detect(buffer)
        breathing = self._breathing.estimate(buffer)
        hr = self._heart_rate.estimate(buffer)
        pose = self._pose.estimate(buffer)

        return Observation(
            presence=presence,
            breathing=breathing,
            heart_rate=hr,
            pose=pose,
            node_ids=used_nodes,
        )

    def calibrate(self, node_id: Optional[str] = None) -> None:
        """Calibrate the presence detector using the current (empty-room) data.

        Parameters
        ----------
        node_id:
            Node to calibrate from.  Defaults to all active nodes merged.
        """
        buffer, _ = self._get_buffer(node_id)
        self._presence.calibrate(buffer)
        logger.info("Presence detector calibrated")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_buffer(
        self, node_id: Optional[str]
    ) -> tuple[CSIBuffer, list[str]]:
        """Return a merged CSIBuffer and list of contributing node IDs."""
        if node_id is not None:
            node = self.nodes.get(node_id)
            if node is None:
                raise KeyError(f"Node {node_id!r} not registered")
            return node.buffer, [node_id]

        active = [n for n in self.nodes.values() if n.is_active()]
        if not active:
            active = list(self.nodes.values())

        if len(active) == 1:
            n = active[0]
            return n.buffer, [n.node_id]

        # Merge frames from all active nodes, sorted by timestamp
        merged = CSIBuffer(max_frames=sum(n.buffer.max_frames for n in active))
        all_frames = sorted(
            [frame for n in active for frame in n.buffer.frames],
            key=lambda f: f.timestamp,
        )
        for frame in all_frames:
            merged.add(frame)

        return merged, [n.node_id for n in active]

    def __repr__(self) -> str:
        return f"RuViewEngine(nodes={list(self.nodes.keys())}, running={self._running})"
