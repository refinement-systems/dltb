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

"""dltb-continuous: video pipeline simulation (real motion + carried state).

Each source frame of the input video is processed exactly once, optionally
carrying the model's previous output forward -- DLSS 5's documented loop,
conceptually:

    P_n = f(state(P_{n-1}), N_n, motion_vectors_n, artistic_direction)

where P is the model's own previous output ("carried temporal state") and N
the freshly rendered frame from the engine.

  --mode anchored : P_n = f(N_n)  -- independent per-frame = BOIL TEST
  --mode stateful : P_n = f(blend(R(P_{n-1}), N_n))  -- healthy pipeline
                    (--anchor-blend alpha = weight of the fresh frame;
                     0.0 = free-running, 1.0 = fully re-anchored per frame)

  REPROJECTION (--reproject, default ON for stateful):
  Real temporal pipelines (TAA, DLSS 2-5) never blend raw history: motion
  vectors warp the carried state before blending (here estimated with
  optical flow; see imaging.make_reprojector). Without it, blending
  superimposes old- and new-position content (ghosting). --no-reproject
  gives the naive history blend.

  TAIL PHASES (--tail-frames N --tail-modes freeze,free,black):
  After the last source frame, the carried state P_end is kept in memory
  (and saved as end_state.png), then EACH requested tail mode branches from
  a COPY of that same state into its own video (tail_<mode>.mp4):

    freeze : engine keeps re-submitting the LAST REAL FRAME (blend continues;
             with reprojection the flow is zero -> identity warp, matching a
             static menu with valid zero motion vectors)
    free   : anchor dropped entirely; source = P_{n-1}  (buffer-echo bug:
             mathematically free-running at ANY blend, since blend(P,P,a) = P)
    black  : engine submits BLACK frames; source = blend(P_{n-1}, black)
             (renderer-crash scenario; a decay driver, NOT free-running)

  CONDITIONING STRATEGIES: run() accepts make_conditioning, a factory
  (args, estimate_flow, warp) -> (combine, tail_source), so tools can replace
  HOW carried state and fresh frame become the model input:

    combine(carried, new_frame, prev_source) -> source   (main loop)
    tail_source(mode, current, last_source, black) -> source   (tail phases)

  The default is the pixel-blend above (_blend_conditioning). dltb-klein uses
  the hook for dual-reference conditioning ([P, N] as two clean reference
  images instead of one blended one; see NOTES.md, "klein dual-reference
  conditioning"). When args.conditioning == "dual-ref" the output tag is
  dualref[-norepro] (no blend component).

Example:
    uv run dltb-continuous --model flux-schnell --input clip.mp4 --mode stateful \\
        --anchor-blend 0.3 --max-frames 300 \\
        --tail-frames 60 --tail-modes freeze,free,black
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .args import add_common_args
from .imaging import (PassSettings, make_generator, make_reprojector,
                      prepare_frame, run_pass)
from .models import MODELS, check_requirements, load_pipeline, resolve_geometry
from .output import run_dir

TAIL_MODES = ("freeze", "free", "black")


def _parse_tail_modes(value: str) -> list[str]:
    modes = [m.strip() for m in value.split(",") if m.strip()]
    bad = [m for m in modes if m not in TAIL_MODES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown tail mode(s): {bad}; choose from {list(TAIL_MODES)}"
        )
    return modes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p)
    p.add_argument("--input", required=True,
                   help="Input video (mp4/mov/mkv/webm/avi)")
    p.add_argument("--mode", choices=["anchored", "stateful"], default="stateful",
                   help="Loop topology (see module docstring). anchored = "
                        "independent per-frame boil test; stateful = carried "
                        "state blended with each new frame. (The anchor-loss "
                        "scenario is covered by --tail-modes free.)")
    p.add_argument("--anchor-blend", type=float, default=0.3,
                   help="stateful mode: weight alpha of the fresh frame in the blend "
                        "(1-a)*previous_output + a*new_frame. 0.0 = free-running, "
                        "1.0 = fully re-anchored every frame")
    p.add_argument("--reproject", action=argparse.BooleanOptionalAction, default=True,
                   help="stateful mode only: warp the carried state by optical flow "
                        "estimated between consecutive source frames before blending "
                        "(emulates engine motion vectors; needs opencv-python-headless). "
                        "--no-reproject gives the naive history blend (ghosting).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many SOURCE frames (tail phases, if "
                        "any, come after)")
    p.add_argument("--tail-frames", type=int, default=0,
                   help="Generate this many extra frames per tail mode after "
                        "the source video ends")
    p.add_argument("--tail-modes", type=_parse_tail_modes, default=_parse_tail_modes("freeze"),
                   help="comma-separated list of tail scenarios, each branching from "
                        "the same end-of-video state: freeze (re-submit last frame), "
                        "free (anchor dropped), black (black frames)")
    p.add_argument("--save-every", type=int, default=10,
                   help="Also save every Nth processed frame as PNG (10 keeps disk "
                        "usage sane on video runs)")
    return p.parse_args(argv)


def _blend_conditioning(args, estimate_flow, warp):
    """Default conditioning: pixel-blend the (optionally reprojected) carried
    state with the fresh frame into ONE model input."""
    from PIL import Image

    def combine(carried, new_frame, prev_source):
        c = carried
        if estimate_flow is not None and prev_source is not None:
            c = warp(c, estimate_flow(prev_source, new_frame))
        return Image.blend(c, new_frame, args.anchor_blend)

    def tail_source(mode, current, last_source, black):
        if mode == "free":
            return current                                # anchor dropped
        if mode == "black":
            return Image.blend(current, black, args.anchor_blend)
        return Image.blend(current, last_source, args.anchor_blend)  # freeze

    return combine, tail_source


def run(args: argparse.Namespace, make_conditioning=None) -> None:
    """Run the video loop. make_conditioning (optional) is a factory
    (args, estimate_flow, warp) -> (combine, tail_source) replacing the
    default pixel-blend conditioning; see the module docstring."""
    spec = MODELS[args.model]
    width, height = resolve_geometry(spec, args.width, args.height)
    check_requirements(spec, args.num_inference_steps, args.strength)
    if not 0.0 <= args.anchor_blend <= 1.0:
        raise SystemExit("--anchor-blend must be in [0, 1]")

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    dual_ref = getattr(args, "conditioning", "blend") == "dual-ref"
    mode_tag = args.mode
    if args.mode == "stateful":
        if dual_ref:
            mode_tag = "dualref"                    # no blend component
        else:
            mode_tag = f"stateful-a{args.anchor_blend:g}"
        if not args.reproject:
            mode_tag += "-norepro"
    if args.tail_frames:
        mode_tag += f"_tails{'-'.join(args.tail_modes)}{args.tail_frames}"
    tag = f"{Path(args.input).stem}_{mode_tag}"
    out_dir = run_dir(args.output_dir, args.model, tag)

    use_reproject = args.reproject and args.mode == "stateful"
    estimate_flow = warp = None
    if use_reproject:
        estimate_flow, warp = make_reprojector()

    if make_conditioning is None:
        combine, tail_source = _blend_conditioning(args, estimate_flow, warp)
    else:
        combine, tail_source = make_conditioning(args, estimate_flow, warp)

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(spec, args.offload)

    reader = imageio.get_reader(args.input, "ffmpeg")
    meta = reader.get_meta_data()
    src_fps = meta.get("fps", 30)
    total = meta.get("nframes")
    if total in (None, float("inf")):
        total = "?"
    print(f"source: {args.input} | fps={src_fps} | frames={total} | mode={args.mode} | "
          f"conditioning={'dual-ref' if dual_ref else 'blend'} | "
          f"reproject={'on' if use_reproject else 'off'} | "
          f"tails={args.tail_frames}x{','.join(args.tail_modes) if args.tail_frames else 'off'}")

    out_video = out_dir / f"processed_{args.mode}.mp4"
    writer = imageio.get_writer(out_video, fps=src_fps, codec="libx264")

    settings = PassSettings(prompt=args.prompt,
                            num_inference_steps=args.num_inference_steps,
                            strength=args.strength,
                            guidance_scale=args.guidance_scale)

    current = None      # previous PROCESSED frame (carried state)
    prev_source = None  # previous SOURCE frame (for flow estimation)
    last_source = None  # last real source frame (for freeze tails)
    n = 0

    def one_pass(source, idx, sink, tag=""):
        nonlocal current
        result = run_pass(pipe, spec, settings, source,
                          make_generator(args.seed, args.fixed_seed, idx),
                          width, height)
        current = result
        sink.append_data(np.asarray(result))
        if idx % args.save_every == 0:
            result.save(frames_dir / f"{tag}frame_{idx:04d}.png")
        if idx % 10 == 0 or idx == 1:
            print(f"{tag or 'main '}frame {idx}", flush=True)

    try:
        for n, frame in enumerate(reader, start=1):
            if args.max_frames is not None and n > args.max_frames:
                break
            new_frame = prepare_frame(Image.fromarray(np.asarray(frame)), width, height)
            last_source = new_frame
            if n == 1:
                new_frame.save(out_dir / "frame_0000_source.png")

            if args.mode == "anchored" or current is None:
                source = new_frame          # independent per frame -> boil test
            else:                           # stateful: carry processed state forward
                source = combine(current, new_frame, prev_source)
            prev_source = new_frame
            one_pass(source, n, writer)
    finally:
        writer.close()
        reader.close()

    print(f"Processed video written to {out_video}")

    # --- tail phases: branch each mode from the SAME end-of-video state ---
    if args.tail_frames and current is not None:
        end_state = current.copy()
        end_state.save(out_dir / "end_state.png")
        black = Image.new("RGB", (width, height), (0, 0, 0))

        for mode in args.tail_modes:
            print(f"--- tail '{mode}': {args.tail_frames} frames from shared end state")
            current = end_state.copy()
            tail_video = out_dir / f"tail_{mode}.mp4"
            tail_writer = imageio.get_writer(tail_video, fps=src_fps, codec="libx264")
            try:
                for t in range(1, args.tail_frames + 1):
                    source = tail_source(mode, current, last_source, black)
                    one_pass(source, t, tail_writer, tag=f"{mode} ")
            finally:
                tail_writer.close()
            print(f"Tail video written to {tail_video}")

    print("Done.")


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
