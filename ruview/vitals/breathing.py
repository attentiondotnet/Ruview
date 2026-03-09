"""
Breathing rate estimation from CSI amplitude time series.

Physical basis
--------------
During respiration the chest wall expands and contracts at roughly 0.1–0.5 Hz
(6–30 breaths per minute).  These slow mechanical oscillations modulate the
WiFi channel and appear as a low-frequency sinusoidal component in the CSI
amplitude variance.

Algorithm
---------
1. Project the CSI buffer onto its first principal component (PCA) to obtain
   a single robust time series.
2. Apply a bandpass filter centred on the breathing band (0.1–0.5 Hz).
3. Estimate the dominant frequency using Welch's power spectral density method.
4. Convert the frequency to breaths per minute (BPM) = frequency × 60.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ruview.csi.models import CSIBuffer
from ruview.csi.processor import CSIProcessor
from ruview.signal.features import pca_compress
from ruview.signal.filters import bandpass_filter, dominant_frequency

# Physiological breathing frequency range (Hz)
BREATHING_LOW_HZ: float = 0.1
BREATHING_HIGH_HZ: float = 0.5

# Minimum sampling rate required for breathing analysis (Hz)
MIN_FS_BREATHING: float = 2.0

# Minimum number of frames (at typical 10 Hz → 5 s of data)
MIN_FRAMES: int = 50


@dataclass
class BreathingResult:
    """Output of :class:`BreathingMonitor`.

    Attributes
    ----------
    rate_bpm:
        Estimated breathing rate in breaths per minute, or ``None`` if
        insufficient data was available.
    confidence:
        Normalised spectral-peak prominence in ``[0, 1]``.
    frequency_hz:
        Dominant frequency in the breathing band (Hz).
    """

    rate_bpm: Optional[float]
    confidence: float
    frequency_hz: float


class BreathingMonitor:
    """Estimate breathing rate from a rolling CSI buffer.

    Parameters
    ----------
    min_frames:
        Minimum number of frames needed before an estimate is produced.
    """

    def __init__(self, min_frames: int = MIN_FRAMES) -> None:
        self.min_frames = min_frames
        self._processor = CSIProcessor()

    def estimate(self, buffer: CSIBuffer) -> BreathingResult:
        """Return a breathing-rate estimate from *buffer*.

        Parameters
        ----------
        buffer:
            Rolling CSI buffer.  The :attr:`~ruview.csi.models.CSIBuffer.sample_rate`
            must be at least :data:`MIN_FS_BREATHING` Hz.

        Returns
        -------
        BreathingResult
        """
        if len(buffer) < self.min_frames:
            return BreathingResult(rate_bpm=None, confidence=0.0, frequency_hz=0.0)

        fs = buffer.sample_rate
        if fs is None or fs < MIN_FS_BREATHING:
            return BreathingResult(rate_bpm=None, confidence=0.0, frequency_hz=0.0)

        amp = self._processor.preprocess_buffer(buffer)
        signal = pca_compress(amp, n_components=1)

        try:
            filtered = bandpass_filter(signal, BREATHING_LOW_HZ, BREATHING_HIGH_HZ, fs=fs)
        except ValueError:
            return BreathingResult(rate_bpm=None, confidence=0.0, frequency_hz=0.0)

        freq_hz, power = dominant_frequency(
            filtered, fs=fs, freq_min=BREATHING_LOW_HZ, freq_max=BREATHING_HIGH_HZ
        )

        if freq_hz < BREATHING_LOW_HZ or power < 1e-12:
            return BreathingResult(rate_bpm=None, confidence=0.0, frequency_hz=0.0)

        rate_bpm = freq_hz * 60.0
        confidence = self._peak_prominence_confidence(filtered, fs)

        return BreathingResult(
            rate_bpm=round(rate_bpm, 1),
            confidence=confidence,
            frequency_hz=round(freq_hz, 4),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _peak_prominence_confidence(signal: np.ndarray, fs: float) -> float:
        """Quantify how dominant the spectral peak is vs. background noise."""
        from scipy.signal import welch

        nperseg = min(len(signal), max(64, len(signal) // 4))
        _, psd = welch(signal, fs=fs, nperseg=nperseg)

        if psd.max() < 1e-12:
            return 0.0

        peak_power = psd.max()
        mean_power = psd.mean()
        snr = peak_power / (mean_power + 1e-12)
        return float(np.clip((snr - 1.0) / 10.0, 0.0, 1.0))
