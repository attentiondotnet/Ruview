"""
RuView — Edge AI Perception System
====================================
Sense presence, pose, breathing rate, and heart rate entirely from
WiFi/radio Channel State Information (CSI) — no cameras, no cloud.

Quick start
-----------
>>> from ruview import RuViewEngine
>>> engine = RuViewEngine()
>>> engine.start()           # connects to edge nodes and begins streaming
>>> result = engine.observe()
>>> print(result.presence, result.breathing_rate, result.heart_rate)
"""

from ruview.engine import RuViewEngine
from ruview.csi.models import CSIFrame, CSIBuffer
from ruview.presence.detector import PresenceDetector
from ruview.vitals.breathing import BreathingMonitor
from ruview.vitals.heart_rate import HeartRateMonitor
from ruview.pose.estimator import PoseEstimator
from ruview.edge.node import EdgeNode

__version__ = "0.1.0"
__all__ = [
    "RuViewEngine",
    "CSIFrame",
    "CSIBuffer",
    "PresenceDetector",
    "BreathingMonitor",
    "HeartRateMonitor",
    "PoseEstimator",
    "EdgeNode",
]
