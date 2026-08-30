"""Generate a tiny synthetic sample video for offline HypeReel demos/tests.

Produces a short, low-resolution clip with a moving colored rectangle over a
changing background, plus a couple of brighter/faster "spike" segments so the
audio-peak/motion-intensity proposer signals have something to detect. This
is NOT real game footage — it exists purely so the pipeline can be exercised
end-to-end without needing a real video file.

Requires numpy + opencv-python (``cv2``). Both are optional dependencies: if
either is missing, this script prints a clear message and exits cleanly
rather than crashing, since the full HypeReel pipeline also runs in mock mode
with no sample video at all.

Usage:
    python samples/generate_sample.py
    python samples/generate_sample.py --seconds 20 --out samples/sample_game.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds", type=float, default=30.0, help="length of the generated clip (default: 30)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "sample_game.mp4"),
        help="output path for the generated .mp4 (default: samples/sample_game.mp4)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import numpy as np
        import cv2
    except ImportError:
        print(
            "samples/generate_sample.py: numpy and/or opencv-python (cv2) are not "
            "installed, so no sample video was generated. This is fine — the "
            "HypeReel pipeline runs fully in mock mode with no sample video needed. "
            "To generate a real sample, run: pip install numpy opencv-python"
        )
        sys.exit(0)

    width, height, fps = 320, 240, 15
    total_frames = int(args.seconds * fps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # A handful of "spike" segments (in frames) simulate crowd-noise/motion
    # bursts — e.g. a made basket — for the audio-peak/motion signals to find.
    spike_ranges = [
        (int(total_frames * 0.25), int(total_frames * 0.30)),
        (int(total_frames * 0.65), int(total_frames * 0.72)),
    ]

    def in_spike(frame_idx: int) -> bool:
        return any(lo <= frame_idx < hi for lo, hi in spike_ranges)

    for i in range(total_frames):
        t = i / fps
        spiking = in_spike(i)

        # Changing background: a slow color drift, brighter during a spike.
        bg_level = 40 + int(20 * (1 + __import__("math").sin(t * 0.5)))
        if spiking:
            bg_level = min(255, bg_level + 100)
        frame = np.full((height, width, 3), bg_level, dtype=np.uint8)

        # Moving colored rectangle ("the player"): faster during a spike.
        speed = 60 if spiking else 20
        rect_w, rect_h = 30, 50
        cx = int((width - rect_w) * (0.5 + 0.5 * __import__("math").sin(t * speed / 30)))
        cy = int((height - rect_h) * (0.5 + 0.5 * __import__("math").cos(t * speed / 45)))
        color = (30, 30, 220) if spiking else (200, 120, 30)  # BGR: red spike, orange normal
        cv2.rectangle(frame, (cx, cy), (cx + rect_w, cy + rect_h), color, thickness=-1)

        writer.write(frame)

    writer.release()
    print(f"wrote {total_frames} frames ({args.seconds:.1f}s @ {fps}fps) to {out_path}")


if __name__ == "__main__":
    main()
