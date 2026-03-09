"""
CSI preprocessing pipeline.

Raw CSI from the ESP32 contains several artefacts that must be removed
before any higher-level analysis:

1. **DC offset** — constant per-antenna hardware offset on amplitude.
2. **Phase ambiguity** — the receiver timestamp offset introduces a
   linear phase ramp across subcarriers; we remove it via linear regression.
3. **Outlier frames** — occasional burst errors produce implausibly large
   amplitude values; we clamp them with a soft median-based filter.
4. **Temporal smoothing** — a lightweight exponential moving average
   reduces high-frequency noise without introducing significant lag.
"""

from __future__ import annotations

import numpy as np

from ruview.csi.models import CSIBuffer, CSIFrame, NUM_SUBCARRIERS


class CSIProcessor:
    """Stateless / stateful CSI preprocessing helpers.

    Parameters
    ----------
    ema_alpha:
        Smoothing factor for the exponential moving average applied to
        per-subcarrier amplitudes.  Range ``(0, 1]``; lower → smoother.
    amplitude_clip_sigma:
        Frames whose mean amplitude deviates more than this many standard
        deviations from the buffer median are considered outliers and
        their amplitudes are clipped.
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        amplitude_clip_sigma: float = 4.0,
    ) -> None:
        if not (0 < ema_alpha <= 1):
            raise ValueError("ema_alpha must be in (0, 1]")
        self.ema_alpha = ema_alpha
        self.amplitude_clip_sigma = amplitude_clip_sigma
        self._ema_state: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Per-frame operations
    # ------------------------------------------------------------------

    def sanitize_phase(self, frame: CSIFrame) -> CSIFrame:
        """Remove the linear phase ramp across subcarriers.

        The residual phase is computed by subtracting a least-squares
        linear fit over subcarrier indices, eliminating the Sampling
        Frequency Offset (SFO) component.

        Returns a *new* :class:`CSIFrame` with corrected phase.
        """
        k = np.arange(len(frame.phase))
        slope, intercept = np.polyfit(k, frame.phase, 1)
        corrected_phase = frame.phase - (slope * k + intercept)
        import dataclasses
        return dataclasses.replace(frame, phase=corrected_phase)

    def apply_ema(self, amplitude: np.ndarray) -> np.ndarray:
        """Apply exponential moving average to a single amplitude vector.

        Maintains internal state so successive calls produce a smooth
        output stream.  Call :meth:`reset` to discard accumulated state.
        """
        if self._ema_state is None or self._ema_state.shape != amplitude.shape:
            self._ema_state = amplitude.copy()
        else:
            self._ema_state = (
                self.ema_alpha * amplitude + (1.0 - self.ema_alpha) * self._ema_state
            )
        return self._ema_state.copy()

    def reset(self) -> None:
        """Clear the EMA internal state."""
        self._ema_state = None

    # ------------------------------------------------------------------
    # Buffer-level operations
    # ------------------------------------------------------------------

    @staticmethod
    def remove_dc_offset(amplitudes: np.ndarray) -> np.ndarray:
        """Subtract the temporal mean from each subcarrier column.

        Parameters
        ----------
        amplitudes:
            Array of shape ``(T, num_subcarriers)``.

        Returns
        -------
        numpy.ndarray
            Mean-removed amplitude matrix of the same shape.
        """
        return amplitudes - amplitudes.mean(axis=0, keepdims=True)

    def remove_outlier_frames(self, amplitudes: np.ndarray) -> np.ndarray:
        """Replace outlier frames with the column-wise median.

        A frame is considered an outlier if its per-subcarrier mean amplitude
        is more than ``amplitude_clip_sigma`` standard deviations away from
        the overall buffer mean.

        Parameters
        ----------
        amplitudes:
            Array of shape ``(T, num_subcarriers)``.

        Returns
        -------
        numpy.ndarray
            Cleaned amplitude matrix of the same shape.
        """
        if amplitudes.shape[0] < 3:
            return amplitudes
        row_means = amplitudes.mean(axis=1)
        mu = np.median(row_means)
        sigma = row_means.std()
        if sigma < 1e-9:
            return amplitudes
        outliers = np.abs(row_means - mu) > self.amplitude_clip_sigma * sigma
        if not np.any(outliers):
            return amplitudes
        result = amplitudes.copy()
        col_medians = np.median(amplitudes[~outliers], axis=0)
        result[outliers] = col_medians
        return result

    def preprocess_buffer(self, buffer: CSIBuffer) -> np.ndarray:
        """Run the full preprocessing pipeline on a :class:`CSIBuffer`.

        Steps:
          1. Extract the amplitude matrix ``(T, K)``.
          2. Remove outlier frames.
          3. Remove DC offset (temporal mean per subcarrier).
          4. Normalise each subcarrier to unit variance.

        Parameters
        ----------
        buffer:
            Buffer to process.

        Returns
        -------
        numpy.ndarray
            Preprocessed amplitude matrix ``(T, num_subcarriers)``.
        """
        amp = buffer.amplitudes.copy()
        if amp.shape[0] == 0:
            return amp
        amp = self.remove_outlier_frames(amp)
        amp = self.remove_dc_offset(amp)
        std = amp.std(axis=0)
        std[std < 1e-9] = 1.0
        return amp / std
