# Permission to use, copy, modify, and/or distribute this software for
# any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""dltb-assemble: encode mp4s from a run's saved PNG frames.

CPU-only post-processing for the split workflow: frames are computed on a
CUDA pod (dltb-continuous / dltb-klein / dltb-iterate; --save-every 1 saves
the full sequence), the run directory is copied home, and the videos are
encoded locally -- no torch, no CUDA (the same local-only idea as
analyze_drift.py). The pod already encodes its own mp4s as it goes, so this
tool is for re-encoding: different --fps, --every subsampling, or frames
copied without the videos.

Frame layout as written by the tools:

    <run>/frames/frame_0001.png ...            main loop
    <run>/frames/<mode> frame_0001.png ...     tail phases -- one prefix per
                                               --tail-modes entry (the space
                                               comes from continuous.py's
                                               per-mode save tag)
    <run>/end_state.png, frame_0000_source.png  (never part of any video)

The main frames become <run>/assembled.mp4; each tail prefix becomes
<run>/assembled_tail_<mode>.mp4. Accepts either the run directory or the
frames directory itself.

CAUTION: frames saved with --save-every N > 1 form a timelapse -- at any
fps the result plays N times faster than the source run. The detected
stride is printed per video; pass a proportionally lower --fps (e.g.
--fps 3 for stride 10 saved from 30 fps source) for real-time pacing.

Example:
    uv run dltb-assemble output_flux2_klein_4b/<run tag> --fps 30
    uv run dltb-assemble output_sdxl_turbo/<run tag>/frames --fps 12 --every 2
    python3 src/dltb/assemble.py <run-or-frames-dir> --fps 24   # no install
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# continuous.py saves the main loop as frame_NNNN.png and each tail phase as
# "<mode> frame_NNNN.png" (the tag passed to one_pass is f"{mode} ").
_FRAME_RE = re.compile(r"^(?:(?P<mode>[a-z]+) )?frame_(?P<idx>\d+)\.png$")


def discover_groups(frames_dir: Path) -> dict[str | None, list[tuple[int, Path]]]:
    """Group frame files by tail-mode prefix: {mode | None: [(idx, path)]},
    each list sorted by frame index (numeric -- robust past 9999 frames,
    unlike a plain filename sort). None is the main loop."""
    groups: dict[str | None, list[tuple[int, Path]]] = {}
    for p in frames_dir.iterdir():
        m = _FRAME_RE.match(p.name)
        if m:
            groups.setdefault(m.group("mode"), []).append((int(m.group("idx")), p))
    for frames in groups.values():
        frames.sort(key=lambda ip: ip[0])
    return groups


def uniform_stride(indices: list[int]) -> int | None:
    """The common index gap if it is uniform (e.g. 10 for --save-every 10),
    else None."""
    if len(indices) < 2:
        return 1
    gaps = {b - a for a, b in zip(indices, indices[1:])}
    return gaps.pop() if len(gaps) == 1 else None


def encode(frames: list[Path], video_path: Path, fps: float) -> None:
    """libx264-encode the given frame files (CPU-only; imageio-ffmpeg's
    bundled binary, the same encoder the pod-side tools use)."""
    import imageio.v2 as imageio

    try:
        with imageio.get_writer(video_path, fps=fps, codec="libx264") as w:
            for f in frames:
                w.append_data(imageio.imread(f))
    except Exception as e:  # encoding is this tool's whole job: fail loudly
        raise SystemExit(f"video encoding failed for {video_path}: {e}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir", type=Path,
                   help="Run directory (containing frames/) or the frames "
                        "directory itself")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Encoding framerate (fractional ok, e.g. 29.97). With "
                        "--save-every N frames this is the TIMELAPSE playback "
                        "speed -- see the stride note printed per video")
    p.add_argument("--every", type=int, default=1,
                   help="Use every Nth of the found frames (like "
                        "analyze_drift's --every)")
    args = p.parse_args(argv)

    frames_dir = args.run_dir
    if (frames_dir / "frames").is_dir():
        frames_dir = frames_dir / "frames"
    if not frames_dir.is_dir():
        raise SystemExit(f"{args.run_dir} is neither a run directory nor a frames "
                         "directory")

    groups = discover_groups(frames_dir)
    if not groups:
        raise SystemExit(f"no frame_*.png / '<mode> frame_*.png' files in {frames_dir}")

    every = max(1, args.every)
    for mode, frames in sorted(groups.items(),
                               key=lambda kv: (kv[0] is not None, kv[0] or "")):
        selected = frames[::every]
        indices = [idx for idx, _ in selected]
        stride = uniform_stride(indices)
        video = frames_dir.parent / ("assembled.mp4" if mode is None
                                     else f"assembled_tail_{mode}.mp4")
        print(f"{'main' if mode is None else f'tail {mode!r}'}: "
              f"{len(selected)} frames -> {video}"
              + (f" (frame stride {stride}: timelapse, {stride}x faster than "
                 f"the source run)" if stride and stride > 1 else ""))
        encode([path for _, path in selected], video, args.fps)

    print("Done.")


if __name__ == "__main__":
    main()
