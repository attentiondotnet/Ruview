"""
Digital signal filtering helpers used throughout RuView.

All filters are implemented with :func:`scipy.signal.butter` (IIR Butterworth)
and applied forward-backward (``sosfiltfilt``) to achieve zero phase shift,
which is important for accurate peak detection in vital-sign waveforms.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def bandpass_filter(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    signal:
        1-D input signal.
    low_hz:
        Lower cut-off frequency in Hz.
    high_hz:
        Upper cut-off frequency in Hz.
    fs:
        Sampling frequency in Hz.
    order:
        Filter order (higher → steeper roll-off, more ringing).

    Returns
    -------
    numpy.ndarray
        Filtered signal of the same length.

    Raises
    ------
    ValueError
        If the frequency bounds are outside the Nyquist range or are invalid.
    """
    nyquist = 0.5 * fs
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Invalid bandpass bounds: low={low_hz} Hz, high={high_hz} Hz "
            f"(Nyquist = {nyquist} Hz)"
        )
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def lowpass_filter(
    signal: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    signal:
        1-D input signal.
    cutoff_hz:
        Cut-off frequency in Hz.
    fs:
        Sampling frequency in Hz.
    order:
        Filter order.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """
    nyquist = 0.5 * fs
    if not (0 < cutoff_hz < nyquist):
        raise ValueError(
            f"Invalid low-pass cutoff: {cutoff_hz} Hz (Nyquist = {nyquist} Hz)"
        )
    sos = butter(order, cutoff_hz / nyquist, btype="low", output="sos")
    return sosfiltfilt(sos, signal)


def dominant_frequency(
    signal: np.ndarray,
    fs: float,
    freq_min: float = 0.0,
    freq_max: float | None = None,
) -> tuple[float, float]:
    """Estimate the dominant frequency in a signal using Welch's method.

    Parameters
    ----------
    signal:
        1-D time-domain signal.
    fs:
        Sampling frequency in Hz.
    freq_min:
        Lower bound of the frequency range to search (Hz).
    freq_max:
        Upper bound of the frequency range to search (Hz).
        Defaults to the Nyquist frequency.

    Returns
    -------
    (frequency_hz, power)
        Frequency of the dominant spectral peak and its power.
    """
    if freq_max is None:
        freq_max = 0.5 * fs

    nperseg = min(len(signal), max(64, len(signal) // 4))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(mask):
        return 0.0, 0.0

    restricted_psd = psd[mask]
    restricted_freqs = freqs[mask]
    peak_idx = np.argmax(restricted_psd)
    return float(restricted_freqs[peak_idx]), float(restricted_psd[peak_idx])
