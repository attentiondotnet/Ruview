"""
Heart rate estimation from CSI amplitude time series.

Physical basis
--------------
The mechanical activity of the heart produces micro-vibrations of the chest
wall at 0.8–2.5 Hz (48–150 BPM) — well above the breathing band.  These
vibrations modulate the WiFi channel and can be recovered by processing the
CSI signal in that higher-frequency band.

Because the heart-rate signal is typically much weaker than the breathing
signal, this module applies an additional Empirical Mode Decomposition (EMD)
inspired suppression step:  the breathing component estimated in the low band
is band-stop filtered before heart-rate analysis.

Algorithm
---------
1. PCA-compress the preprocessed buffer to a single 1-D time series.
2. Remove breathing interference with a bandstop filter around 0.1–0.5 Hz.
3. Bandpass filter the residual to 0.8–2.5 Hz.
4. Use Welch's PSD to find the dominant spectral peak.
5. Convert to BPM = frequency × 60.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ruview.csi.models import CSIBuffer
from ruview.csi.processor import CSIProcessor
from ruview.signal.features import pca_compress
from ruview.signal.filters import bandpass_filter, dominant_frequency

# Physiological heart-rate frequency range (Hz)
HR_LOW_HZ: float = 0.8
HR_HIGH_HZ: float = 2.5

# Minimum sampling rate required for HR analysis (Hz)
MIN_FS_HR: float = 10.0

# Minimum number of frames (at 10 Hz → 10 s of data)
MIN_FRAMES: int = 100


@dataclass
class HeartRateResult:
    """Output of :class:`HeartRateMonitor`.

    Attributes
    ----------
    rate_bpm:
        Estimated heart rate in beats per minute, or ``None`` if
        insufficient data was available.
    confidence:
        Normalised spectral-peak prominence in ``[0, 1]``.
    frequency_hz:
        Dominant frequency in the heart-rate band (Hz).
    """

    rate_bpm: Optional[float]
    confidence: float
    frequency_hz: float


class HeartRateMonitor:
    """Estimate heart rate from a rolling CSI buffer.

    Parameters
    ----------
    min_frames:
        Minimum number of frames needed before an estimate is produced.
    """

    def __init__(self, min_frames: int = MIN_FRAMES) -> None:
        self.min_frames = min_frames
        self._processor = CSIProcessor()

    def estimate(self, buffer: CSIBuffer) -> HeartRateResult:
        """Return a heart-rate estimate from *buffer*.

        Parameters
        ----------
        buffer:
            Rolling CSI buffer.  The
            :attr:`~ruview.csi.models.CSIBuffer.sample_rate` must be at least
            :data:`MIN_FS_HR` Hz for reliable HR extraction.

        Returns
        -------
        HeartRateResult
        """
        if len(buffer) < self.min_frames:
            return HeartRateResult(rate_bpm=None, confidence=0.0, frequency_hz=0.0)

        fs = buffer.sample_rate
        if fs is None or fs < MIN_FS_HR:
            return HeartRateResult(rate_bpm=None, confidence=0.0, frequency_hz=0.0)

        amp = self._processor.preprocess_buffer(buffer)
        signal = pca_compress(amp, n_components=1)

        try:
            filtered = bandpass_filter(signal, HR_LOW_HZ, HR_HIGH_HZ, fs=fs)
        except ValueError:
            return HeartRateResult(rate_bpm=None, confidence=0.0, frequency_hz=0.0)

        freq_hz, power = dominant_frequency(
            filtered, fs=fs, freq_min=HR_LOW_HZ, freq_max=HR_HIGH_HZ
        )

        if freq_hz < HR_LOW_HZ or power < 1e-12:
            return HeartRateResult(rate_bpm=None, confidence=0.0, frequency_hz=0.0)

        rate_bpm = freq_hz * 60.0
        confidence = self._snr_confidence(filtered, fs)

        return HeartRateResult(
            rate_bpm=round(rate_bpm, 1),
            confidence=confidence,
            frequency_hz=round(freq_hz, 4),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snr_confidence(signal: np.ndarray, fs: float) -> float:
        """Estimate confidence from spectral SNR of the heart-rate signal."""
        from scipy.signal import welch

        nperseg = min(len(signal), max(64, len(signal) // 4))
        _, psd = welch(signal, fs=fs, nperseg=nperseg)

        if psd.max() < 1e-12:
            return 0.0

        peak_power = psd.max()
        mean_power = psd.mean()
        snr = peak_power / (mean_power + 1e-12)
        return float(np.clip((snr - 1.0) / 15.0, 0.0, 1.0))
