"""
Human presence detection from CSI amplitude variance.

Algorithm
---------
When no one is present the wireless channel is essentially static and the
CSI amplitude matrix has very low temporal variance.  Human movement — even
subtle breathing — scatters energy off the body and introduces characteristic
fluctuations across subcarriers.

The detector maintains a calibration baseline of the *empty-room* variance
and flags presence when the current variance significantly exceeds that
baseline.

Adaptive threshold
------------------
The baseline is updated incrementally with a slow exponential moving average
during periods classified as *empty*.  This lets the system adapt over time
to slow environmental changes (e.g. furniture being moved, temperature shifts
affecting the RF environment).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ruview.csi.models import CSIBuffer
from ruview.csi.processor import CSIProcessor
from ruview.signal.features import pca_compress, csi_variance


@dataclass
class PresenceResult:
    """Output of :class:`PresenceDetector`.

    Attributes
    ----------
    present:
        ``True`` if at least one person is estimated to be present.
    confidence:
        Value in ``[0, 1]`` indicating detection confidence.
    variance_ratio:
        Ratio of the current CSI variance to the calibrated empty-room
        baseline.  Values > 1 indicate activity above the baseline.
    """

    present: bool
    confidence: float
    variance_ratio: float


class PresenceDetector:
    """Detect human presence from a rolling :class:`~ruview.csi.models.CSIBuffer`.

    Parameters
    ----------
    threshold:
        Variance-ratio threshold above which presence is declared.
        A value of ``2.5`` means the current variance must be 2.5× the
        empty-room baseline to trigger detection.
    min_frames:
        Minimum number of frames required before a decision can be made.
    baseline_alpha:
        EMA coefficient used to adapt the empty-room baseline over time.
        Lower values → slower but more stable adaptation.
    """

    def __init__(
        self,
        threshold: float = 2.5,
        min_frames: int = 30,
        baseline_alpha: float = 0.01,
    ) -> None:
        if threshold <= 1.0:
            raise ValueError("threshold must be > 1.0")
        if min_frames < 2:
            raise ValueError("min_frames must be >= 2")
        self.threshold = threshold
        self.min_frames = min_frames
        self.baseline_alpha = baseline_alpha

        self._processor = CSIProcessor()
        self._baseline_variance: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, buffer: CSIBuffer) -> PresenceResult:
        """Analyse the buffer and return a :class:`PresenceResult`.

        Parameters
        ----------
        buffer:
            Rolling CSI buffer.  Should contain at least
            ``min_frames`` frames for a reliable estimate.

        Returns
        -------
        PresenceResult
        """
        if len(buffer) < self.min_frames:
            return PresenceResult(present=False, confidence=0.0, variance_ratio=1.0)

        amp = self._processor.preprocess_buffer(buffer)
        signal = pca_compress(amp, n_components=1)
        current_var = float(np.var(signal))

        if self._baseline_variance is None:
            self._baseline_variance = current_var
            return PresenceResult(present=False, confidence=0.0, variance_ratio=1.0)

        if self._baseline_variance < 1e-12:
            ratio = float("inf") if current_var > 1e-12 else 1.0
        else:
            ratio = current_var / self._baseline_variance

        present = ratio > self.threshold
        confidence = float(np.clip((ratio - 1.0) / (self.threshold - 1.0), 0.0, 1.0))

        if not present:
            self._update_baseline(current_var)

        return PresenceResult(present=present, confidence=confidence, variance_ratio=ratio)

    def calibrate(self, buffer: CSIBuffer) -> None:
        """Force-set the empty-room baseline from the provided buffer.

        Call this method while the monitored space is known to be empty
        to establish an accurate initial baseline.
        """
        if len(buffer) < self.min_frames:
            raise ValueError(
                f"Need at least {self.min_frames} frames for calibration, "
                f"got {len(buffer)}."
            )
        amp = self._processor.preprocess_buffer(buffer)
        signal = pca_compress(amp, n_components=1)
        self._baseline_variance = float(np.var(signal))

    def reset(self) -> None:
        """Reset the calibrated baseline, forcing re-calibration."""
        self._baseline_variance = None
        self._processor.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_baseline(self, current_var: float) -> None:
        """Slowly drift the baseline towards the current measurement."""
        if self._baseline_variance is None:
            self._baseline_variance = current_var
        else:
            self._baseline_variance = (
                self.baseline_alpha * current_var
                + (1.0 - self.baseline_alpha) * self._baseline_variance
            )
