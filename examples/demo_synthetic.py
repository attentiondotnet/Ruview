"""
demo_synthetic.py — Run the full RuView pipeline on synthetic CSI data.

No hardware or WiFi sensor is required for this demo.
"""

import time
import numpy as np
from ruview import RuViewEngine
from ruview.csi.models import CSIFrame
from ruview.edge.node import EdgeNode

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
FS           = 20.0   # simulated sample rate (Hz)
N_FRAMES     = 400    # number of frames to generate
N_SUB        = 52     # WiFi subcarriers (802.11n 20 MHz)
BREATH_HZ    = 0.25   # ~15 breaths / min
HR_HZ        = 1.2    # ~72 BPM

# ---------------------------------------------------------------------------
# Generate synthetic CSI data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

node = EdgeNode(node_id="demo-node")
engine = RuViewEngine(nodes=[node])
engine.start()

print(f"Generating {N_FRAMES} synthetic CSI frames at {FS} Hz …")
for i in range(N_FRAMES):
    t = i / FS
    base  = rng.normal(loc=40.0, scale=2.0, size=N_SUB).astype(np.float32)
    breath = 5.0 * np.sin(2 * np.pi * BREATH_HZ * t)
    hr_mod  = 1.5 * np.sin(2 * np.pi * HR_HZ   * t)
    amp = np.clip(base + breath + hr_mod + rng.normal(scale=0.4, size=N_SUB), 0, None)
    phase = rng.uniform(-np.pi, np.pi, N_SUB).astype(np.float32)

    frame = CSIFrame(
        timestamp=time.time() - (N_FRAMES - i) / FS,
        node_id="demo-node",
        amplitude=amp.astype(np.float32),
        phase=phase,
    )
    node.ingest(frame)

# ---------------------------------------------------------------------------
# Run perception
# ---------------------------------------------------------------------------
obs = engine.observe()

print()
br = (f"{obs.breathing.rate_bpm:.1f} BPM"
      if obs.breathing.rate_bpm is not None else "insufficient data")
hr = (f"{obs.heart_rate.rate_bpm:.1f} BPM"
      if obs.heart_rate.rate_bpm is not None else "insufficient data")
presence_str = f"{'YES' if obs.presence.present else 'NO'}  (conf={obs.presence.confidence:.2f})"

print("╔══════════════════════════════════════════╗")
print("║      RuView  —  Perception Result        ║")
print("╠══════════════════════════════════════════╣")
print(f"║  Presence       : {presence_str:<24}║")
print(f"║  Breathing rate : {br:<24}║")
print(f"║  Heart rate     : {hr:<24}║")
print(f"║  Pose confidence: {obs.pose.overall_confidence:<24.2f}║")
print("╚══════════════════════════════════════════╝")

engine.stop()
