"""dltb-klein: video pipeline simulation with the FLUX.2 klein editors.

Klein (FLUX.2-klein-4B / -9B) is a reference-image EDITOR, not a partial-noise
img2img model: it regenerates from pure noise attending to pristine reference
tokens, so there is NO --strength, and per-pass change scales with the blend
alpha (roughly linear -- no constant rewrite bias like sd/sdxl-turbo and
flux-schnell). Consequences for the loop:

  --anchor-blend    klein's active range is far below the img2img models
                    (default here 0.1; try 0.03 / 0.05 / 0.2)
  --prompt          the de-facto per-pass edit-strength knob: an empty or
                    neutral prompt preserves, an edit instruction compounds
                    every pass (scripts/sweep-klein.sh walks that ladder)
  --num-inference-steps  real scheduler steps (klein card default 4) -- the
                    direct per-pass edit-intensity knob alongside the prompt
                    (sweep-klein probes 2 / 8 around it)

  Guidance: --guidance-scale > 1 is INERT for klein -- step-wise distilled
  models get no CFG (hard-disabled) and no guidance embedding; diffusers
  warns and ignores the value. dltb-klein prints its own warning; see
  NOTES.md, "FLUX.2 klein: --guidance-scale is inert".

This runs the same loop topologies as dltb-continuous (anchored boil test /
stateful blend with optical-flow reprojection / failure tails), restricted to
the klein pair, with the klein-appropriate blend default. It shares
continuous.run() -- only argument defaults and model choice differ.

Refinement idea: pass [P_{n-1}, N] as two separate reference images instead
of pixel-blending.

Example:
    uv run dltb-klein --model flux2-klein-4b --input clip.mp4 \\
        --prompt "slightly enhance the fine details" \\
        --tail-frames 60 --tail-modes freeze
"""

from __future__ import annotations

import argparse

from .args import add_geometry_args, add_output_args, add_pass_args
from .continuous import _parse_tail_modes, run as run_continuous
from .models import model_key

KLEIN_MODELS = ("flux2-klein-4b", "flux2-klein-9b")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=model_key, choices=KLEIN_MODELS,
                   default="flux2-klein-4b",
                   help="Klein model to run (4b: Apache-2.0, ungated; 9b: FLUX "
                        "Non-Commercial, gated -- accept the license and set HF_TOKEN)")
    add_pass_args(p)
    add_geometry_args(p)
    add_output_args(p)
    p.add_argument("--input", required=True,
                   help="Input video (mp4/mov/mkv/webm/avi)")
    p.add_argument("--mode", choices=["anchored", "stateful"], default="stateful",
                   help="Loop topology (see module docstring). anchored = "
                        "independent per-frame boil test; stateful = carried "
                        "state blended with each new frame.")
    p.add_argument("--anchor-blend", type=float, default=0.1,
                   help="stateful mode: weight alpha of the fresh frame in the blend "
                        "(1-a)*previous_output + a*new_frame. Klein's active range "
                        "is far below the img2img models (try 0.03/0.05/0.2); "
                        "0.0 = free-running, 1.0 = fully re-anchored every frame")
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


def run(args: argparse.Namespace) -> None:
    if args.guidance_scale is not None and args.guidance_scale > 1.0:
        print(
            f"dltb-klein: WARNING: --guidance-scale {args.guidance_scale:g} > 1 is "
            "ignored by step-wise distilled klein models (CFG is hard-disabled "
            "and there is no guidance embedding); the run would be identical to "
            "the default-guidance one. See NOTES.md, 'FLUX.2 klein: "
            "--guidance-scale is inert'.",
            flush=True,
        )
    # Identical loop to dltb-continuous; only the argument surface above
    # differs (model restriction, klein blend default, guidance emphasis).
    run_continuous(args)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
