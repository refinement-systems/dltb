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

"""dltb-klein: video pipeline simulation with the FLUX.2 klein editors.

Klein (FLUX.2-klein-4B / -9B) is a reference-image EDITOR, not a partial-noise
img2img model: it regenerates from pure noise attending to pristine reference
tokens, so there is NO --strength, and per-pass change scales with the blend
alpha (roughly linear -- no constant rewrite bias like sd/sdxl-turbo and
flux-schnell). Consequences for the loop:

  --anchor-blend    klein's active range is far below the img2img models
                    (default here 0.1; try 0.03 / 0.05 / 0.2). Applies to
                    --conditioning blend only; IGNORED under dual-ref.
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

CONDITIONING (--conditioning, stateful mode only):

  blend    (default) pixel-blend the (reprojected) carried state and the fresh
           frame into ONE reference image: image = (1-a)*R(P_{n-1}) + a*N_n

  dual-ref pass the carried state and the fresh frame as TWO SEPARATE clean
           reference images: image = [R(P_{n-1}), N_n]. Flux2KleinPipeline
           accepts a list natively (each reference is preprocessed and packed
           onto the sequence axis), and each reference stays in-distribution --
           no ghosted blend mush for the editor to parse. The model itself
           decides how to weight state vs. anchor through attention, which is
           categorically closer to DLSS 5's multi-input conditioning than the
           pixel blend. --anchor-blend is ignored in this mode.

           --ref-order {state-first,frame-first} (default state-first) selects
           [P, N] vs [N, P]; whether klein treats reference order as
           "primary vs target" is an open question -- A/B it (see NOTES.md).

           Tails under dual-ref: freeze = [P, last_source],
           free = [P] ALONE (single-reference regeneration from state -- the
           pure buffer-echo case), black = [P, black].

           Reprojection stays probeable: with --reproject (default) the
           CARRIED reference is warped by the source-frame flow before
           pairing; A/B with --no-reproject.

Example:
    uv run dltb-klein --model flux2-klein-4b --input clip.mp4 \\
        --conditioning dual-ref \\
        --prompt "image 2 is the current frame; keep the appearance of image 1" \\
        --tail-frames 60 --tail-modes freeze,free
"""

from __future__ import annotations

import argparse

from .args import add_geometry_args, add_output_args, add_pass_args
from .continuous import _parse_tail_modes, run as run_continuous
from .models import model_key

KLEIN_MODELS = ("flux2-klein-4b", "flux2-klein-9b")


def _dual_ref_conditioning(args, estimate_flow, warp):
    """Klein dual-reference conditioning: carried state and fresh frame as two
    separate clean references (a list flows through imaging.run_pass unchanged).
    If reprojection is on, the CARRIED reference is warped by the source-frame
    flow before pairing; the fresh frame is never warped."""

    def ordered(state_img, frame_img):
        if args.ref_order == "state-first":
            return [state_img, frame_img]
        return [frame_img, state_img]

    def combine(carried, new_frame, prev_source):
        c = carried
        if estimate_flow is not None and prev_source is not None:
            c = warp(c, estimate_flow(prev_source, new_frame))
        return ordered(c, new_frame)

    def tail_source(mode, current, last_source, black):
        if mode == "free":
            return [current]   # anchor dropped: single-reference regeneration
        return ordered(current, black if mode == "black" else last_source)

    return combine, tail_source


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
                        "state conditioned with each new frame.")
    p.add_argument("--conditioning", choices=["blend", "dual-ref"], default="blend",
                   help="stateful mode only: how carried state + fresh frame become "
                        "model input. blend = one pixel-blended reference "
                        "(--anchor-blend applies); dual-ref = two separate clean "
                        "references, model-weighted (blend ignored)")
    p.add_argument("--ref-order", choices=["state-first", "frame-first"],
                   default="state-first",
                   help="dual-ref only: reference order [P, N] vs [N, P]; whether "
                        "klein treats order as primary/target is an open question "
                        "-- A/B it")
    p.add_argument("--anchor-blend", type=float, default=0.1,
                   help="stateful+blend only: weight alpha of the fresh frame in "
                        "the blend (1-a)*previous_output + a*new_frame. Klein's "
                        "active range is far below the img2img models (try "
                        "0.03/0.05/0.2); 0.0 = free-running, 1.0 = fully "
                        "re-anchored every frame. IGNORED under dual-ref.")
    p.add_argument("--reproject", action=argparse.BooleanOptionalAction, default=True,
                   help="stateful mode only: warp the carried state by optical flow "
                        "estimated between consecutive source frames before "
                        "combining (emulates engine motion vectors; needs "
                        "opencv-python-headless). Under dual-ref only the CARRIED "
                        "reference is warped. --no-reproject gives the naive variant.")
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
    if args.conditioning == "dual-ref":
        if args.anchor_blend != 0.1:
            print("dltb-klein: NOTE: --anchor-blend is ignored under "
                  "--conditioning dual-ref (state/anchor weighting is done by "
                  "the model, not by pixel blending).", flush=True)
        run_continuous(args, make_conditioning=_dual_ref_conditioning)
    else:
        # Identical loop to dltb-continuous; only the argument surface above
        # differs (model restriction, klein blend default, guidance emphasis).
        run_continuous(args)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
