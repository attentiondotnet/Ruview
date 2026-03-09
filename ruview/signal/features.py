"""
Feature extraction from pre-processed CSI matrices.

These compact representations of the raw CSI are fed into the
higher-level presence, pose, and vitals modules.
"""

from __future__ import annotations

import numpy as np


def csi_energy(amplitudes: np.ndarray) -> np.ndarray:
    """Compute per-subcarrier signal energy across the time axis.

    Parameters
    ----------
    amplitudes:
        Array of shape ``(T, K)`` (time × subcarrier).

    Returns
    -------
    numpy.ndarray
        1-D energy vector of shape ``(K,)``.
    """
    return np.mean(amplitudes**2, axis=0)


def csi_variance(amplitudes: np.ndarray) -> np.ndarray:
    """Compute per-subcarrier temporal variance.

    Parameters
    ----------
    amplitudes:
        Array of shape ``(T, K)``.

    Returns
    -------
    numpy.ndarray
        1-D variance vector of shape ``(K,)``.
    """
    return np.var(amplitudes, axis=0)


def pca_compress(amplitudes: np.ndarray, n_components: int = 1) -> np.ndarray:
    """Project the CSI amplitude matrix onto its leading principal components.

    Reduces the subcarrier dimension to ``n_components``, retaining the
    directions of maximum variance.  This single time-series is very effective
    for breathing and heart-rate extraction because human body movement tends
    to project strongly onto the first PC.

    Parameters
    ----------
    amplitudes:
        Array of shape ``(T, K)`` after DC-offset removal.
    n_components:
        Number of principal components to retain.

    Returns
    -------
    numpy.ndarray
        Projected signal of shape ``(T, n_components)``.  If
        ``n_components == 1``, a 1-D array of shape ``(T,)`` is returned.
    """
    if amplitudes.shape[0] < 2:
        return amplitudes[:, :n_components].squeeze()

    # Centre each column (should already be done by the preprocessor, but
    # we apply it here defensively)
    centered = amplitudes - amplitudes.mean(axis=0, keepdims=True)

    try:
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return centered[:, :n_components].squeeze()

    projected = U[:, :n_components] * s[:n_components]
    if n_components == 1:
        return projected.squeeze()
    return projected
