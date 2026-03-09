"""Tests for CSIFrame, CSIBuffer, and CSIProcessor."""
from __future__ import annotations

import struct
import time

import numpy as np
import pytest

from ruview.csi.models import CSIFrame, CSIBuffer, NUM_SUBCARRIERS
from ruview.csi.processor import CSIProcessor
from tests.conftest import make_frame, make_buffer, N_SUB


# ---------------------------------------------------------------------------
# CSIFrame
# ---------------------------------------------------------------------------

class TestCSIFrame:
    def test_defaults(self):
        f = CSIFrame()
        assert f.amplitude.shape == (NUM_SUBCARRIERS,)
        assert f.phase.shape == (NUM_SUBCARRIERS,)
        assert f.num_subcarriers == NUM_SUBCARRIERS

    def test_complex_csi(self):
        amp = np.array([1.0, 2.0], dtype=np.float32)
        phase = np.array([0.0, np.pi / 2], dtype=np.float32)
        f = CSIFrame(amplitude=amp, phase=phase)
        c = f.complex_csi
        assert c.shape == (2,)
        np.testing.assert_allclose(np.abs(c), amp, atol=1e-5)
        np.testing.assert_allclose(np.angle(c), phase, atol=1e-5)

    def test_from_esp32_bytes_round_trip(self):
        rng = np.random.default_rng(1)
        raw = rng.integers(-127, 127, size=NUM_SUBCARRIERS * 2, dtype=np.int8).tobytes()
        f = CSIFrame.from_esp32_bytes(raw, node_id="esp32-01", channel=11)
        assert f.node_id == "esp32-01"
        assert f.channel == 11
        assert f.amplitude.shape == (NUM_SUBCARRIERS,)
        assert f.phase.shape == (NUM_SUBCARRIERS,)
        assert np.all(f.amplitude >= 0)

    def test_from_esp32_bytes_odd_length_raises(self):
        with pytest.raises(ValueError, match="even"):
            CSIFrame.from_esp32_bytes(b"\x01\x02\x03")

    def test_from_udp_packet(self):
        node_id = "n1"
        timestamp = 1234567890.123
        rssi = -55
        channel = 6
        n_sub = 52
        raw_csi = np.zeros(n_sub * 2, dtype=np.int8).tobytes()

        header = struct.pack(
            "<8sdbbH",
            node_id.encode("ascii").ljust(8, b"\x00"),
            timestamp,
            rssi,
            channel,
            n_sub,
        )
        packet = header + raw_csi
        f = CSIFrame.from_udp_packet(packet)
        assert f.node_id == node_id
        assert abs(f.timestamp - timestamp) < 1e-3
        assert f.rssi == rssi
        assert f.channel == channel

    def test_repr(self):
        f = make_frame(node_id="x", t=1.0)
        assert "x" in repr(f)


# ---------------------------------------------------------------------------
# CSIBuffer
# ---------------------------------------------------------------------------

class TestCSIBuffer:
    def test_add_and_len(self):
        buf = CSIBuffer(max_frames=5)
        for i in range(5):
            buf.add(make_frame(t=float(i)))
        assert len(buf) == 5

    def test_max_frames_eviction(self):
        buf = CSIBuffer(max_frames=3)
        for i in range(5):
            buf.add(make_frame(t=float(i)))
        assert len(buf) == 3
        # Oldest frames should be evicted
        assert buf.frames[0].timestamp == pytest.approx(2.0)

    def test_amplitudes_shape(self):
        buf = make_buffer(n_frames=10)
        assert buf.amplitudes.shape == (10, N_SUB)

    def test_phases_shape(self):
        buf = make_buffer(n_frames=10)
        assert buf.phases.shape == (10, N_SUB)

    def test_timestamps_shape(self):
        buf = make_buffer(n_frames=10)
        ts = buf.timestamps
        assert ts.shape == (10,)
        assert ts[0] < ts[-1]

    def test_sample_rate(self):
        buf = make_buffer(n_frames=50, fs=20.0)
        sr = buf.sample_rate
        assert sr is not None
        assert abs(sr - 20.0) < 2.0

    def test_sample_rate_none_when_empty(self):
        buf = CSIBuffer()
        assert buf.sample_rate is None

    def test_clear(self):
        buf = make_buffer(n_frames=10)
        buf.clear()
        assert len(buf) == 0


# ---------------------------------------------------------------------------
# CSIProcessor
# ---------------------------------------------------------------------------

class TestCSIProcessor:
    def test_remove_dc_offset(self):
        proc = CSIProcessor()
        amp = np.ones((10, N_SUB)) * 5.0
        result = proc.remove_dc_offset(amp)
        np.testing.assert_allclose(result, np.zeros((10, N_SUB)), atol=1e-10)

    def test_remove_outlier_frames(self):
        proc = CSIProcessor(amplitude_clip_sigma=3.0)
        rng = np.random.default_rng(2)
        amp = rng.normal(loc=10.0, scale=1.0, size=(50, N_SUB))
        # Inject a clear outlier row
        amp[25] = 1000.0
        cleaned = proc.remove_outlier_frames(amp)
        assert cleaned[25].mean() < 100.0

    def test_sanitize_phase_removes_ramp(self):
        proc = CSIProcessor()
        k = np.arange(N_SUB, dtype=float)
        phase = 0.1 * k + 1.5  # linear ramp
        frame = make_frame(phase=phase.astype(np.float32))
        corrected = proc.sanitize_phase(frame)
        # After removing the linear ramp the variance should drop dramatically
        assert corrected.phase.var() < phase.var() * 0.1

    def test_apply_ema_smoothing(self):
        proc = CSIProcessor(ema_alpha=0.1)
        rng = np.random.default_rng(3)
        noisy = rng.normal(loc=10.0, scale=5.0, size=N_SUB).astype(np.float32)
        result = proc.apply_ema(noisy)
        # EMA output on first call should equal the input
        np.testing.assert_array_equal(result, noisy)
        # Second call should be a blend
        noisy2 = rng.normal(loc=10.0, scale=5.0, size=N_SUB).astype(np.float32)
        result2 = proc.apply_ema(noisy2)
        # The output should NOT equal noisy2 (it is blended with the previous state)
        assert not np.allclose(result2, noisy2)

    def test_preprocess_buffer_shape(self):
        proc = CSIProcessor()
        buf = make_buffer(n_frames=50)
        out = proc.preprocess_buffer(buf)
        assert out.shape == (50, N_SUB)

    def test_preprocess_buffer_empty(self):
        proc = CSIProcessor()
        buf = CSIBuffer()
        out = proc.preprocess_buffer(buf)
        assert out.shape[0] == 0

    def test_reset_clears_ema(self):
        proc = CSIProcessor(ema_alpha=0.1)
        amp = np.ones(N_SUB, dtype=np.float32) * 10.0
        proc.apply_ema(amp)
        proc.reset()
        assert proc._ema_state is None
