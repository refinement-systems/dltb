# Permission to use, copy, modify, and/or distribute this software for
# any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
# AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Output-directory layout, input-type sniffing, and timelapse assembly.

All tools write runs under output_<model>/ (or an explicit --output-dir) in
one subdirectory per run, tagged so that different configurations never
collide:

    output_<model>/<input stem>_<run tag>/frames/frame_NNNN.png
"""

from __future__ import annotations

from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"}


def is_video(path: str | Path) -> bool:
    """Sniff the input type from the file extension."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def run_dir(output_dir: str | None, model: str, tag: str) -> Path:
    """Create and return <output_dir | output_<model>>/<tag>."""
    base = Path(output_dir or f"output_{model.replace('-', '_')}")
    d = base / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def assemble_timelapse(frames_dir: Path, video_path: Path, fps: int) -> None:
    """Assemble frame_*.png from frames_dir into an mp4 (best effort)."""
    try:
        import imageio.v2 as imageio
        frames = sorted(frames_dir.glob("frame_*.png"))
        if frames:
            with imageio.get_writer(video_path, fps=fps, codec="libx264") as w:
                for f in frames:
                    w.append_data(imageio.imread(f))
            print(f"Video written to {video_path}")
    except Exception as e:  # video is a convenience, not a requirement
        print(f"Video assembly skipped ({e}); frames are in {frames_dir}")
