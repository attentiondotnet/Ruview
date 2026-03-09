"""Tests for the WiFi DensePose estimator."""
from __future__ import annotations

import numpy as np
import pytest

from ruview.pose.estimator import (
    PoseEstimator,
    PoseResult,
    Keypoint,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    SKELETON_EDGES,
)
from tests.conftest import make_buffer


class TestKeypoint:
    def test_fields(self):
        kp = Keypoint(name="nose", x=0.5, y=0.1, confidence=0.9)
        assert kp.name == "nose"
        assert kp.x == pytest.approx(0.5)
        assert kp.confidence == pytest.approx(0.9)


class TestPoseResult:
    def _make_result(self, conf: float = 0.5) -> PoseResult:
        kps = [Keypoint(name=n, x=0.5, y=float(i) / NUM_KEYPOINTS, confidence=conf)
               for i, n in enumerate(KEYPOINT_NAMES)]
        return PoseResult(keypoints=kps)

    def test_overall_confidence(self):
        pr = self._make_result(conf=0.7)
        assert pr.overall_confidence == pytest.approx(0.7)

    def test_overall_confidence_empty(self):
        pr = PoseResult()
        assert pr.overall_confidence == pytest.approx(0.0)

    def test_as_array_shape(self):
        pr = self._make_result()
        arr = pr.as_array()
        assert arr.shape == (NUM_KEYPOINTS, 3)

    def test_as_array_xy_range(self):
        pr = self._make_result()
        arr = pr.as_array()
        assert np.all(arr[:, :2] >= 0.0)
        assert np.all(arr[:, :2] <= 1.0)

    def test_repr(self):
        pr = self._make_result()
        assert "PoseResult" in repr(pr)


class TestPoseEstimator:
    def test_prior_pose_structure(self):
        estimator = PoseEstimator()
        buf = make_buffer(n_frames=40)
        result = estimator.estimate(buf)
        assert len(result.keypoints) == NUM_KEYPOINTS
        for kp in result.keypoints:
            assert 0.0 <= kp.x <= 1.0
            assert 0.0 <= kp.y <= 1.0
            assert 0.0 <= kp.confidence <= 1.0

    def test_prior_contains_all_keypoints(self):
        estimator = PoseEstimator()
        buf = make_buffer(n_frames=5)  # too few for model, uses prior
        result = estimator.estimate(buf)
        names = [kp.name for kp in result.keypoints]
        assert set(names) == set(KEYPOINT_NAMES)

    def test_update_increments_counter(self):
        estimator = PoseEstimator(feature_dim=64)
        buf = make_buffer(n_frames=100)
        gt = np.random.default_rng(7).random((NUM_KEYPOINTS, 3)).astype(np.float32)
        gt[:, :2] = np.clip(gt[:, :2], 0, 1)  # ensure x,y in [0,1]
        gt[:, 2] = np.clip(gt[:, 2], 0, 1)    # ensure confidence in [0,1]
        assert estimator.n_updates == 0
        estimator.update(buf, gt)
        assert estimator.n_updates == 1

    def test_update_wrong_shape_raises(self):
        estimator = PoseEstimator()
        buf = make_buffer(n_frames=100)
        wrong_shape = np.zeros((NUM_KEYPOINTS, 2))
        with pytest.raises(ValueError, match="shape"):
            estimator.update(buf, wrong_shape)

    def test_estimate_after_training(self):
        rng = np.random.default_rng(8)
        estimator = PoseEstimator(feature_dim=64)
        for _ in range(20):
            buf = make_buffer(n_frames=100, rng=rng)
            gt = rng.random((NUM_KEYPOINTS, 3)).astype(np.float32)
            estimator.update(buf, gt)
        buf = make_buffer(n_frames=100, rng=rng)
        result = estimator.estimate(buf)
        assert len(result.keypoints) == NUM_KEYPOINTS

    def test_skeleton_edges_valid_indices(self):
        for a, b in SKELETON_EDGES:
            assert 0 <= a < NUM_KEYPOINTS
            assert 0 <= b < NUM_KEYPOINTS

    def test_keypoint_names_count(self):
        assert len(KEYPOINT_NAMES) == 17
        assert "nose" in KEYPOINT_NAMES
        assert "left_ankle" in KEYPOINT_NAMES
