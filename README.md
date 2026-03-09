# RuView

> **Perceive the world through signals. No cameras. No wearables. No Internet. Just physics.**

RuView is an **edge AI perception system** that learns directly from the environment around it.  Instead of relying on cameras or cloud models, it observes the signals that already fill a space — WiFi, radio waves, motion, vibration — and builds a local understanding of what is happening.

Built on the physics of **Channel State Information (CSI)**, RuView reconstructs **human presence**, **body pose**, **breathing rate**, and **heart rate** in real time using signal processing and machine learning — all running at the edge on hardware as inexpensive as an ESP32 (~$1/node).

---

## Key Features

| Capability | Details |
|---|---|
| 🧍 **Presence detection** | Detects human occupancy from CSI variance — no motion trigger needed |
| 🫀 **Vital signs** | Estimates breathing rate (0.1–0.5 Hz band) and heart rate (0.8–2.5 Hz band) |
| 🦴 **WiFi DensePose** | Reconstructs 17-point COCO skeleton keypoints from radio signals |
| 📡 **Multi-node fusion** | Aggregates CSI from multiple ESP32 sensor nodes |
| 🔒 **Fully offline** | No cloud, no cameras, no labeled data required |
| 🔋 **Edge-first** | Runs on ESP32 hardware; Python host runs on Raspberry Pi or any PC |
| 🧠 **Self-learning** | Presence baseline and pose model adapt over time to local RF environment |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          RuView Host (Python)                        │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────┐  │
│  │  EdgeNode   │→  │ CSIProcessor │→  │    RuViewEngine           │  │
│  │  (UDP/Serial│   │ (preprocess) │   │  ┌────────┐ ┌─────────┐  │  │
│  │   receiver) │   └──────────────┘   │  │Presence│ │Breathing│  │  │
│  └─────────────┘                      │  └────────┘ └─────────┘  │  │
│         ▲                             │  ┌─────────┐ ┌─────────┐ │  │
│         │  UDP / Serial               │  │HeartRate│ │  Pose   │ │  │
│         │                             │  └─────────┘ └─────────┘ │  │
└─────────┼───────────────────────────────────────────────────────────┘
          │
┌─────────┴────────────────────────┐
│  ESP32 Sensor Mesh (Firmware)    │
│  ┌────────┐  ┌────────┐          │
│  │ node-01│  │ node-02│  …       │
│  │ CSI tap│  │ CSI tap│          │
│  └────────┘  └────────┘          │
└──────────────────────────────────┘
```

---

## Quick Start

### 1 — Install the Python package

```bash
pip install -e ".[dev]"
```

### 2 — Run the demo (no hardware needed)

```bash
ruview demo
```

or directly:

```bash
python examples/demo_synthetic.py
```

Expected output:

```
╔══════════════════════════════════════╗
║      RuView  —  Perception Result    ║
╠══════════════════════════════════════╣
║  Presence       :  YES  (conf=0.98)  ║
║  Breathing rate : 15.0 BPM           ║
║  Heart rate     : 72.0 BPM           ║
║  Pose confidence: 0.10               ║
╚══════════════════════════════════════╝
```

### 3 — Flash the ESP32 firmware

Open `firmware/csi_node/csi_node.ino` in the Arduino IDE (or use PlatformIO):

1. Edit the `USER CONFIGURATION` block at the top of the sketch:
   - Set `WIFI_SSID` / `WIFI_PASSWORD`
   - Set `HOST_IP` to the IP of your Python host
   - Give each node a unique `NODE_ID`
2. Flash via **Arduino IDE** or `pio run --target upload`

### 4 — Start the live engine

```bash
ruview run --port 5005
```

For serial (USB cable, no WiFi):

```bash
ruview run --serial /dev/ttyUSB0 --baud 921600
```

---

## Python API

```python
from ruview import RuViewEngine
from ruview.edge import EdgeNode, UDPReceiver

# Create a node and a UDP receiver
node = EdgeNode(node_id="esp32-living-room")
receiver = UDPReceiver(port=5005, nodes={node.node_id: node})

# Start the engine
engine = RuViewEngine(nodes=[node])
engine.start()
receiver.start()

# Calibrate the empty-room baseline
engine.calibrate()  # call while room is empty

# Poll for observations
import time
while True:
    obs = engine.observe()
    if obs.presence.present:
        print(f"Someone is here! Breathing: {obs.breathing.rate_bpm} BPM")
    time.sleep(1)
```

---

## Signal Processing Pipeline

### Presence Detection

1. Preprocess CSI buffer (remove DC offset, outlier frames, normalise).
2. Project to first principal component (PCA).
3. Compute temporal variance of the PC1 signal.
4. Compare to calibrated empty-room baseline; raise a flag when `variance_ratio > threshold`.

### Breathing Rate Estimation

1. PCA-compress CSI to a single time series.
2. Butterworth bandpass filter: **0.1 – 0.5 Hz** (6 – 30 breaths/min).
3. Welch PSD → dominant frequency → multiply by 60 to get BPM.

### Heart Rate Estimation

1. PCA-compress CSI to a single time series.
2. Butterworth bandpass filter: **0.8 – 2.5 Hz** (48 – 150 BPM).
3. Welch PSD → dominant frequency → multiply by 60 to get BPM.

### WiFi DensePose (Pose Estimation)

Inspired by [*DensePose from WiFi* (CMU, 2022)](https://arxiv.org/abs/2301.00250).

1. Extract per-observation feature vector:
   - PC1 statistics (mean, std, min, max)
   - Per-subcarrier variance (sub-sampled)
   - Per-subcarrier mean amplitude (sub-sampled)
2. L2-normalise the feature vector.
3. Linear regression maps the feature vector to 17 COCO keypoint coordinates.
4. Model is updated incrementally via gradient descent on new (CSI → keypoints) pairs.
5. Prior to any training, an anatomically correct upright-standing pose is returned with low confidence.

---

## Hardware

| Component | Cost | Notes |
|---|---|---|
| ESP32-DevKitC | ~$4 | Recommended dev board |
| ESP32-WROOM-32 module | ~$1 | For production deployments |
| USB-C cable | — | For serial / flashing |

A single node is sufficient for presence and vitals detection.  For accurate pose estimation, place **3–4 nodes** at different positions around the monitored space.

---

## Repository Layout

```
Ruview/
├── ruview/                  Python package
│   ├── csi/                 CSI data models & preprocessing
│   ├── signal/              Bandpass filters, PSD, feature extraction
│   ├── presence/            Presence detector
│   ├── vitals/              Breathing & heart-rate monitors
│   ├── pose/                WiFi DensePose estimator (COCO 17-point)
│   ├── edge/                Edge node + UDP/Serial transport
│   ├── engine.py            Top-level RuViewEngine orchestrator
│   └── cli.py               `ruview` command-line tool
├── firmware/
│   └── csi_node/
│       ├── csi_node.ino     Arduino sketch for ESP32
│       └── platformio.ini   PlatformIO config
├── tests/                   pytest test suite
├── examples/
│   └── demo_synthetic.py    No-hardware demo
├── pyproject.toml
└── requirements.txt
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check ruview tests
```

---

## References

- He, J., et al. *DensePose from WiFi*. arXiv:2301.00250 (2023).  
- Adib, F., et al. *See Through Walls with WiFi!* ACM SIGCOMM (2013).  
- ESP32 CSI API: https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi.html

---

## License

MIT © Attention.net

