"""Tests for breathing rate and heart rate estimation."""
from __future__ import annotations

import numpy as np
import pytest

from ruview.csi.models import CSIBuffer, CSIFrame
from ruview.vitals.breathing import BreathingMonitor, BreathingResult, BREATHING_LOW_HZ, BREATHING_HIGH_HZ
from ruview.vitals.heart_rate import HeartRateMonitor, HeartRateResult, HR_LOW_HZ, HR_HIGH_HZ
from tests.conftest import make_buffer, make_frame, N_SUB


# ---------------------------------------------------------------------------
# BreathingMonitor
# ---------------------------------------------------------------------------

class TestBreathingMonitor:
    def test_insufficient_frames_returns_none(self):
        monitor = BreathingMonitor(min_frames=50)
        buf = CSIBuffer()
        for i in range(10):
            buf.add(make_frame(t=float(i) / 20.0))
        result = monitor.estimate(buf)
        assert result.rate_bpm is None
        assert result.confidence == pytest.approx(0.0)

    def test_detects_breathing_rate(self):
        BREATH_HZ = 0.25  # 15 BPM
        buf = make_buffer(n_frames=300, fs=20.0, breath_hz=BREATH_HZ, noise_std=0.3)
        monitor = BreathingMonitor(min_frames=50)
        result = monitor.estimate(buf)
        assert result.rate_bpm is not None
        # Allow ±3 BPM tolerance
        assert abs(result.rate_bpm - BREATH_HZ * 60) < 3.0

    def test_breathing_rate_in_valid_range(self):
        buf = make_buffer(n_frames=300, fs=20.0, breath_hz=0.3)
        monitor = BreathingMonitor()
        result = monitor.estimate(buf)
        if result.rate_bpm is not None:
            assert BREATHING_LOW_HZ * 60 <= result.rate_bpm <= BREATHING_HIGH_HZ * 60

    def test_confidence_is_in_unit_interval(self):
        buf = make_buffer(n_frames=300, fs=20.0)
        monitor = BreathingMonitor()
        result = monitor.estimate(buf)
        assert 0.0 <= result.confidence <= 1.0

    def test_result_dataclass_fields(self):
        result = BreathingResult(rate_bpm=15.0, confidence=0.8, frequency_hz=0.25)
        assert result.rate_bpm == pytest.approx(15.0)
        assert result.frequency_hz == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# HeartRateMonitor
# ---------------------------------------------------------------------------

class TestHeartRateMonitor:
    def test_insufficient_frames_returns_none(self):
        monitor = HeartRateMonitor(min_frames=100)
        buf = CSIBuffer()
        for i in range(20):
            buf.add(make_frame(t=float(i) / 20.0))
        result = monitor.estimate(buf)
        assert result.rate_bpm is None

    def test_detects_heart_rate(self):
        HR_HZ = 1.2   # 72 BPM
        buf = make_buffer(n_frames=400, fs=25.0, breath_hz=0.2, hr_hz=HR_HZ, noise_std=0.2)
        monitor = HeartRateMonitor(min_frames=100)
        result = monitor.estimate(buf)
        assert result.rate_bpm is not None
        # Allow ±6 BPM tolerance
        assert abs(result.rate_bpm - HR_HZ * 60) < 6.0

    def test_heart_rate_in_valid_range(self):
        buf = make_buffer(n_frames=400, fs=25.0, hr_hz=1.0)
        monitor = HeartRateMonitor()
        result = monitor.estimate(buf)
        if result.rate_bpm is not None:
            assert HR_LOW_HZ * 60 <= result.rate_bpm <= HR_HIGH_HZ * 60

    def test_confidence_in_unit_interval(self):
        buf = make_buffer(n_frames=400, fs=25.0)
        monitor = HeartRateMonitor()
        result = monitor.estimate(buf)
        assert 0.0 <= result.confidence <= 1.0

    def test_low_sample_rate_returns_none(self):
        # Build buffer with sample rate below the minimum (< 10 Hz)
        buf = make_buffer(n_frames=200, fs=3.0)
        monitor = HeartRateMonitor()
        result = monitor.estimate(buf)
        assert result.rate_bpm is None

    def test_result_dataclass_fields(self):
        result = HeartRateResult(rate_bpm=72.0, confidence=0.7, frequency_hz=1.2)
        assert result.rate_bpm == pytest.approx(72.0)
        assert result.frequency_hz == pytest.approx(1.2)
