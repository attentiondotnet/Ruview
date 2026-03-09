"""Shared test fixtures and helpers."""
from __future__ import annotations

import time
import numpy as np
import pytest

from ruview.csi.models import CSIFrame, CSIBuffer

FS = 20.0       # simulated sample rate (Hz)
N_SUB = 52      # subcarriers


def make_frame(
    amplitude: np.ndarray | None = None,
    phase: np.ndarray | None = None,
    t: float = 0.0,
    node_id: str = "test",
) -> CSIFrame:
    if amplitude is None:
        amplitude = np.ones(N_SUB, dtype=np.float32) * 20.0
    if phase is None:
        phase = np.zeros(N_SUB, dtype=np.float32)
    return CSIFrame(timestamp=t, node_id=node_id, amplitude=amplitude, phase=phase)


def make_buffer(
    n_frames: int = 200,
    fs: float = FS,
    breath_hz: float = 0.25,
    hr_hz: float = 1.2,
    noise_std: float = 0.5,
    rng: np.random.Generator | None = None,
) -> CSIBuffer:
    """Build a synthetic CSI buffer with breathing and heart-rate modulations."""
    if rng is None:
        rng = np.random.default_rng(0)
    buf = CSIBuffer(max_frames=n_frames + 10)
    for i in range(n_frames):
        t = i / fs
        base = rng.normal(loc=40.0, scale=2.0, size=N_SUB).astype(np.float32)
        breath = 5.0 * np.sin(2 * np.pi * breath_hz * t)
        hr = 1.5 * np.sin(2 * np.pi * hr_hz * t)
        amp = np.clip(base + breath + hr + rng.normal(scale=noise_std, size=N_SUB), 0, None)
        phase = rng.uniform(-np.pi, np.pi, N_SUB).astype(np.float32)
        buf.add(make_frame(amp.astype(np.float32), phase, t=t))
    return buf
