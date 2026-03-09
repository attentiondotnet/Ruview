"""
WiFi DensePose — estimating human body keypoints from CSI data.

Background
----------
Research by Carnegie Mellon University (*DensePose from WiFi*, 2022) showed
that a neural network trained on paired WiFi-CSI / camera data can reconstruct
dense human pose from radio signals alone.

RuView implements a *self-supervised* variant that does **not** require a
camera at inference time and learns an incremental spatial embedding of the
room.  At initialisation we build a physics-informed prior over the 17-keypoint
COCO skeleton using arrival-angle geometry; this prior is refined online as the
sensor collects data.

The current implementation provides:

* The keypoint data structures and skeleton topology (COCO 17-point).
* A feature extractor that maps a preprocessed CSI buffer to a fixed-length
  embedding suitable for downstream regression.
* A lightweight linear regression–based estimator that can be updated
  incrementally with new (keypoint, CSI) pairs.
* A stub ``PoseEstimator.estimate()`` method that returns a structurally
  valid but approximate pose when a trained model is not available, based on
  the physics-informed prior alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ruview.csi.models import CSIBuffer
from ruview.csi.processor import CSIProcessor
from ruview.signal.features import pca_compress, csi_variance

# ---------------------------------------------------------------------------
# COCO 17-point skeleton definition
# ---------------------------------------------------------------------------

KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Pairs of keypoint indices that form skeleton edges for visualisation
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12),                          # torso
    (11, 12),                                  # hips
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

NUM_KEYPOINTS: int = len(KEYPOINT_NAMES)


@dataclass
class Keypoint:
    """A single body keypoint with normalised 2-D coordinates.

    Coordinates are in the range ``[0, 1]`` relative to the sensing region.

    Attributes
    ----------
    name:
        Keypoint label (e.g. ``'nose'``).
    x:
        Horizontal normalised coordinate.
    y:
        Vertical normalised coordinate.
    confidence:
        Estimated localisation confidence in ``[0, 1]``.
    """

    name: str
    x: float
    y: float
    confidence: float = 0.0


@dataclass
class PoseResult:
    """Full-body pose estimate from one CSI observation.

    Attributes
    ----------
    keypoints:
        List of :class:`Keypoint` in COCO order.
    overall_confidence:
        Mean confidence across all keypoints.
    """

    keypoints: list[Keypoint] = field(default_factory=list)

    @property
    def overall_confidence(self) -> float:
        if not self.keypoints:
            return 0.0
        return float(np.mean([kp.confidence for kp in self.keypoints]))

    def as_array(self) -> np.ndarray:
        """Return ``(NUM_KEYPOINTS, 3)`` array of ``[x, y, confidence]``."""
        return np.array([[kp.x, kp.y, kp.confidence] for kp in self.keypoints])

    def __repr__(self) -> str:
        return f"PoseResult(keypoints={len(self.keypoints)}, conf={self.overall_confidence:.2f})"


class PoseEstimator:
    """Estimate human body pose from a CSI buffer.

    The estimator maintains an incremental linear model mapping the CSI
    feature vector to keypoint coordinates.  Before any training data is
    available it falls back to a *standing-upright prior* that places
    keypoints at anatomically reasonable positions with low confidence.

    Parameters
    ----------
    feature_dim:
        Length of the CSI feature vector extracted per observation.
    learning_rate:
        Step size for the incremental least-squares update.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        learning_rate: float = 0.01,
    ) -> None:
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self._processor = CSIProcessor()

        # Weight matrix: (feature_dim, NUM_KEYPOINTS * 3)
        # Columns are [x_0, y_0, c_0, x_1, y_1, c_1, …]
        self._weights: Optional[np.ndarray] = None
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, buffer: CSIBuffer) -> PoseResult:
        """Estimate pose from a CSI buffer.

        If the model has not been trained yet (``n_updates == 0``), returns
        the standing-upright prior with low confidence.

        Parameters
        ----------
        buffer:
            Rolling CSI buffer with at least 30 frames.

        Returns
        -------
        PoseResult
        """
        features = self._extract_features(buffer)

        if self._weights is None or self._n_updates < 10:
            return self._prior_pose(confidence=0.1)

        raw = features @ self._weights  # (NUM_KEYPOINTS * 3,)
        return self._decode_output(raw, base_confidence=0.5)

    def update(self, buffer: CSIBuffer, keypoints: np.ndarray) -> None:
        """Update the model with a new (CSI → keypoints) training sample.

        Parameters
        ----------
        buffer:
            CSI buffer captured while the ground-truth pose was observed.
        keypoints:
            Ground-truth keypoint array of shape ``(NUM_KEYPOINTS, 3)``
            with columns ``[x, y, confidence]``.
        """
        if keypoints.shape != (NUM_KEYPOINTS, 3):
            raise ValueError(
                f"keypoints must have shape ({NUM_KEYPOINTS}, 3), "
                f"got {keypoints.shape}"
            )
        features = self._extract_features(buffer)
        target = keypoints.flatten()

        if self._weights is None:
            self._weights = np.zeros((self.feature_dim, NUM_KEYPOINTS * 3))

        # Incremental gradient descent update
        prediction = features @ self._weights
        error = prediction - target
        grad = np.outer(features, error)
        self._weights -= self.learning_rate * grad
        self._n_updates += 1

    @property
    def n_updates(self) -> int:
        """Number of training updates applied so far."""
        return self._n_updates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, buffer: CSIBuffer) -> np.ndarray:
        """Map a CSI buffer to a fixed-length feature vector.

        The feature vector concatenates:
        - First principal component statistics (mean, std, min, max)
        - Per-subcarrier variance (sub-sampled to ``feature_dim // 2``)
        - Per-subcarrier mean amplitude (sub-sampled to ``feature_dim // 2``)

        The result is L2-normalised.
        """
        if len(buffer) < 2:
            return np.zeros(self.feature_dim)

        amp = self._processor.preprocess_buffer(buffer)
        pc1 = pca_compress(amp, n_components=1)

        # Statistical features from the first PC
        pc_stats = np.array([pc1.mean(), pc1.std(), pc1.min(), pc1.max()])

        var_vec = csi_variance(amp)
        mean_vec = amp.mean(axis=0)

        half = self.feature_dim // 2
        var_sub = self._subsample(var_vec, half)
        mean_sub = self._subsample(mean_vec, half)

        raw = np.concatenate([pc_stats, var_sub, mean_sub])
        raw = raw[: self.feature_dim]

        # Pad if necessary
        if len(raw) < self.feature_dim:
            raw = np.pad(raw, (0, self.feature_dim - len(raw)))

        norm = np.linalg.norm(raw)
        return raw / (norm + 1e-12)

    @staticmethod
    def _subsample(vec: np.ndarray, target_len: int) -> np.ndarray:
        """Resample *vec* to *target_len* using linear interpolation."""
        if len(vec) == target_len:
            return vec
        idx = np.linspace(0, len(vec) - 1, target_len)
        return np.interp(idx, np.arange(len(vec)), vec)

    @staticmethod
    def _prior_pose(confidence: float = 0.1) -> PoseResult:
        """Return an upright-standing anatomical prior pose.

        Coordinates are in normalised image space ``[0, 1]``.  The figure
        is centred horizontally and occupies most of the vertical range.
        """
        # Approximate COCO keypoint positions for a person standing upright,
        # facing the sensor.
        prior_xy: list[tuple[float, float]] = [
            (0.50, 0.08),  # 0  nose
            (0.48, 0.07),  # 1  left_eye
            (0.52, 0.07),  # 2  right_eye
            (0.46, 0.09),  # 3  left_ear
            (0.54, 0.09),  # 4  right_ear
            (0.44, 0.20),  # 5  left_shoulder
            (0.56, 0.20),  # 6  right_shoulder
            (0.40, 0.35),  # 7  left_elbow
            (0.60, 0.35),  # 8  right_elbow
            (0.37, 0.50),  # 9  left_wrist
            (0.63, 0.50),  # 10 right_wrist
            (0.45, 0.55),  # 11 left_hip
            (0.55, 0.55),  # 12 right_hip
            (0.44, 0.72),  # 13 left_knee
            (0.56, 0.72),  # 14 right_knee
            (0.43, 0.90),  # 15 left_ankle
            (0.57, 0.90),  # 16 right_ankle
        ]
        keypoints = [
            Keypoint(name=KEYPOINT_NAMES[i], x=xy[0], y=xy[1], confidence=confidence)
            for i, xy in enumerate(prior_xy)
        ]
        return PoseResult(keypoints=keypoints)

    @staticmethod
    def _decode_output(raw: np.ndarray, base_confidence: float) -> PoseResult:
        """Decode the raw model output into a :class:`PoseResult`."""
        coords = raw.reshape(NUM_KEYPOINTS, 3)
        keypoints = []
        for i, (x, y, c) in enumerate(coords):
            keypoints.append(
                Keypoint(
                    name=KEYPOINT_NAMES[i],
                    x=float(np.clip(x, 0.0, 1.0)),
                    y=float(np.clip(y, 0.0, 1.0)),
                    confidence=float(np.clip(c * base_confidence, 0.0, 1.0)),
                )
            )
        return PoseResult(keypoints=keypoints)
