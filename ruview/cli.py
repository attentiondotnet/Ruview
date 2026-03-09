"""
RuView command-line interface.

Usage
-----
    ruview --help
    ruview run --port 5005
    ruview run --serial /dev/ttyUSB0 --baud 921600
    ruview demo
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ruview.cli")


def _cmd_run(args: argparse.Namespace) -> None:
    """Start the RuView perception engine and print live observations."""
    from ruview.edge.node import EdgeNode
    from ruview.edge.protocol import UDPReceiver, SerialReceiver
    from ruview.engine import RuViewEngine

    node = EdgeNode(node_id="node-01")
    engine = RuViewEngine(nodes=[node])
    engine.start()

    if args.serial:
        logger.info("Using serial transport on %s @ %d baud", args.serial, args.baud)
        receiver = SerialReceiver(port=args.serial, baudrate=args.baud, node=node)
    else:
        logger.info("Using UDP transport on 0.0.0.0:%d", args.port)
        receiver = UDPReceiver(host="0.0.0.0", port=args.port, nodes={node.node_id: node})

    with receiver:
        logger.info("Waiting for sensor data …  (Ctrl-C to quit)")
        try:
            while True:
                time.sleep(1.0)
                obs = engine.observe()
                print(obs)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
    engine.stop()


def _cmd_demo(args: argparse.Namespace) -> None:
    """Inject synthetic CSI data and print observations (no hardware required)."""
    import numpy as np
    from ruview.csi.models import CSIFrame, CSIBuffer
    from ruview.edge.node import EdgeNode
    from ruview.engine import RuViewEngine

    rng = np.random.default_rng(42)

    node = EdgeNode(node_id="demo")
    engine = RuViewEngine(nodes=[node])
    engine.start()

    FS = 20.0          # simulated sampling rate (Hz)
    N = 300            # number of frames to inject
    N_SUB = 52         # subcarriers
    BREATH_HZ = 0.25   # ~15 breaths/min
    HR_HZ = 1.2        # ~72 BPM

    print(f"Injecting {N} synthetic CSI frames at {FS} Hz …")
    for i in range(N):
        t = i / FS
        # Simulate CSI amplitude: background noise + breathing + heart-rate modulation
        base = rng.normal(loc=40.0, scale=2.0, size=N_SUB).astype(np.float32)
        breath_signal = 5.0 * np.sin(2 * np.pi * BREATH_HZ * t)
        hr_signal = 1.5 * np.sin(2 * np.pi * HR_HZ * t)
        amp = base + breath_signal + hr_signal + rng.normal(scale=0.5, size=N_SUB)
        amp = np.clip(amp, 0, None).astype(np.float32)
        phase = rng.uniform(-np.pi, np.pi, N_SUB).astype(np.float32)

        frame = CSIFrame(
            timestamp=time.time() - (N - i) / FS,
            node_id="demo",
            amplitude=amp,
            phase=phase,
        )
        node.ingest(frame)

    obs = engine.observe()
    print("\n── RuView Demo Result ──────────────────────────")
    print(f"  Presence       : {obs.presence.present} (conf={obs.presence.confidence:.2f})")
    br = f"{obs.breathing.rate_bpm:.1f} BPM" if obs.breathing.rate_bpm else "insufficient data"
    hr = f"{obs.heart_rate.rate_bpm:.1f} BPM" if obs.heart_rate.rate_bpm else "insufficient data"
    print(f"  Breathing rate : {br}")
    print(f"  Heart rate     : {hr}")
    print(f"  Pose (conf)    : {obs.pose.overall_confidence:.2f}")
    print("────────────────────────────────────────────────\n")
    engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ruview",
        description="RuView — edge AI perception from WiFi/radio signals",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────────
    run_parser = sub.add_parser("run", help="Start the live perception engine")
    run_parser.add_argument(
        "--port", type=int, default=5005, help="UDP port to listen on (default: 5005)"
    )
    run_parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Serial device path (e.g. /dev/ttyUSB0 or COM3)",
    )
    run_parser.add_argument(
        "--baud", type=int, default=921600, help="Serial baud rate (default: 921600)"
    )
    run_parser.set_defaults(func=_cmd_run)

    # ── demo ─────────────────────────────────────────────────────────────
    demo_parser = sub.add_parser(
        "demo", help="Run a demo with synthetic CSI data (no hardware required)"
    )
    demo_parser.set_defaults(func=_cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
