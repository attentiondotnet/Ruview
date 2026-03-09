"""Tests for presence detection."""
from __future__ import annotations

import numpy as np
import pytest

from ruview.csi.models import CSIBuffer
from ruview.presence.detector import PresenceDetector, PresenceResult
from tests.conftest import make_frame, make_buffer, N_SUB, FS


class TestPresenceDetector:
    def test_insufficient_frames_no_detection(self):
        detector = PresenceDetector(min_frames=30)
        buf = CSIBuffer()
        for i in range(5):
            buf.add(make_frame(t=float(i) / FS))
        result = detector.detect(buf)
        assert result.present is False
        assert result.confidence == pytest.approx(0.0)

    def test_empty_room_no_presence(self):
        """Static CSI (no motion) should not trigger presence."""
        rng = np.random.default_rng(10)
        buf = CSIBuffer()
        # Very low-variance signal simulating empty room
        base_amp = np.ones(N_SUB, dtype=np.float32) * 30.0
        for i in range(60):
            amp = base_amp + rng.normal(scale=0.05, size=N_SUB).astype(np.float32)
            buf.add(make_frame(amp, t=float(i) / FS))

        detector = PresenceDetector(threshold=2.5, min_frames=30)
        # Calibrate with same static data
        detector.calibrate(buf)

        # Second batch of equally static data — should not detect presence
        buf2 = CSIBuffer()
        for i in range(60):
            amp = base_amp + rng.normal(scale=0.05, size=N_SUB).astype(np.float32)
            buf2.add(make_frame(amp, t=float(i + 60) / FS))
        result = detector.detect(buf2)
        assert result.present is False

    def test_high_variance_triggers_presence(self):
        """Highly variable CSI (simulating movement) should trigger presence."""
        rng = np.random.default_rng(11)

        # Build empty-room baseline
        static_buf = CSIBuffer()
        base_amp = np.ones(N_SUB, dtype=np.float32) * 30.0
        for i in range(60):
            amp = base_amp + rng.normal(scale=0.05, size=N_SUB).astype(np.float32)
            static_buf.add(make_frame(amp, t=float(i) / FS))

        detector = PresenceDetector(threshold=2.5, min_frames=30)
        detector.calibrate(static_buf)

        # Now inject highly variable signal
        motion_buf = make_buffer(n_frames=60, fs=FS, breath_hz=0.25, noise_std=3.0)
        result = detector.detect(motion_buf)
        assert result.present is True
        assert result.confidence > 0.0

    def test_variance_ratio_positive(self):
        buf = make_buffer(n_frames=50)
        detector = PresenceDetector(min_frames=30)
        result = detector.detect(buf)
        assert result.variance_ratio >= 0.0

    def test_calibrate_with_too_few_frames_raises(self):
        detector = PresenceDetector(min_frames=30)
        buf = CSIBuffer()
        for i in range(5):
            buf.add(make_frame(t=float(i)))
        with pytest.raises(ValueError, match="frames"):
            detector.calibrate(buf)

    def test_reset_clears_baseline(self):
        buf = make_buffer(n_frames=50)
        detector = PresenceDetector(min_frames=30)
        detector.calibrate(buf)
        assert detector._baseline_variance is not None
        detector.reset()
        assert detector._baseline_variance is None

    def test_confidence_in_unit_interval(self):
        buf = make_buffer(n_frames=60, fs=FS)
        detector = PresenceDetector(min_frames=30)
        result = detector.detect(buf)
        assert 0.0 <= result.confidence <= 1.0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            PresenceDetector(threshold=0.5)

    def test_result_fields(self):
        r = PresenceResult(present=True, confidence=0.9, variance_ratio=4.0)
        assert r.present is True
        assert r.confidence == pytest.approx(0.9)
        assert r.variance_ratio == pytest.approx(4.0)
