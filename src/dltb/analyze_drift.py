#!/usr/bin/env python3
"""analyze_drift: quantify convergence/drift of a dltb run from its frames.

Runs locally on saved frames - no GPU, no torch. Only numpy + Pillow.

For each frame it computes two mean-absolute-pixel-difference metrics (0-255 scale):

  delta_prev     : |frame_t - frame_{t-1}|   -> per-pass change.
                   -> 0 means the loop reached a FIXED POINT (converged).
                   Plateau > 0 means permanent chatter (no fixed point).
  delta_original : |frame_t - frame_0|       -> cumulative departure from the
                   input image. Slope = drift rate; plateau = bounded change.

Usage:
    python analyze_drift.py output_sdxl_turbo/free-running/frames
    python analyze_drift.py output_flux_schnell/free-running/frames --every 1

Outputs drift_metrics.csv next to the frames dir and prints a summary.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames_dir", type=Path, help="Directory with frame_XXXX.png files")
    p.add_argument("--every", type=int, default=1,
                   help="Analyze every Nth frame (speeds up long runs)")
    args = p.parse_args()

    frames = sorted(args.frames_dir.glob("frame_*.png"))[:: args.every]
    if len(frames) < 2:
        raise SystemExit(f"Need at least 2 frames in {args.frames_dir}")

    size = Image.open(frames[0]).size
    original_path = args.frames_dir.parent / "frame_0000_original.png"
    original = (
        load_gray(original_path, size) if original_path.exists() else load_gray(frames[0], size)
    )

    rows = []
    prev = load_gray(frames[0], size)
    for f in frames:
        cur = load_gray(f, size)
        rows.append(
            (
                f.stem,
                float(np.abs(cur - prev).mean()),       # delta_prev
                float(np.abs(cur - original).mean()),   # delta_original
            )
        )
        prev = cur

    out_csv = args.frames_dir.parent / "drift_metrics.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "delta_prev", "delta_original"])
        w.writerows(rows)

    tail = rows[-10:]
    mean_tail_prev = np.mean([r[1] for r in tail])
    mean_tail_orig = np.mean([r[2] for r in tail])
    peak_orig = max(r[2] for r in rows)

    print(f"Analyzed {len(rows)} frames -> {out_csv}")
    print(f"  mean delta_prev over last 10 frames : {mean_tail_prev:8.3f}  "
          f"({'CONVERGED to fixed point' if mean_tail_prev < 0.5 else 'still drifting'})")
    print(f"  mean delta_original, last 10 frames : {mean_tail_orig:8.3f}")
    print(f"  peak delta_original                 : {peak_orig:8.3f}")


if __name__ == "__main__":
    main()
