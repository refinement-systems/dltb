"""argparse flag groups shared by the dltb-* tools.

Tool-specific flags (--input, --iterations, --mode, tails, ...) are added by
each tool on top of these; help texts here apply verbatim everywhere.
dltb-klein composes the same groups but swaps in its own --model (restricted
to the klein pair, with a default), which is why model selection lives in
its own adder.
"""

from __future__ import annotations

import argparse

from .models import MODELS, model_key


def add_model_args(p: argparse.ArgumentParser) -> None:
    """--model over the full table (required)."""
    p.add_argument("--model", required=True, type=model_key, choices=sorted(MODELS),
                   help="Which model to run")


def add_pass_args(p: argparse.ArgumentParser) -> None:
    """Per-pass inference settings."""
    p.add_argument("--strength", type=float, default=0.4,
                   help="img2img denoising strength per pass (sd/sdxl-turbo and "
                        "flux-schnell only). Keep num_inference_steps * strength >= 1.")
    p.add_argument("--num-inference-steps", type=int, default=4,
                   help="Denoise schedule length (all models distilled for 1-4 steps)")
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="Override the model's default guidance scale. Experimental: "
                        "klein takes guidance as an embedded conditioning signal "
                        "(card default 1.0), so raising it may strengthen prompt "
                        "adherence per pass.")
    p.add_argument("--prompt", default="",
                   help="Optional text prompt. A faithful description acts as a "
                        "semantic anchor (analogue of DLSS 5's artistic-direction "
                        "conditioning).")


def add_geometry_args(p: argparse.ArgumentParser) -> None:
    """Output geometry and seeding."""
    p.add_argument("--width", type=int, default=None,
                   help="Output width (multiple of 16; frames center-cropped to this "
                        "aspect, then resized)")
    p.add_argument("--height", type=int, default=None,
                   help="Output height (multiple of 16)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--fixed-seed", action=argparse.BooleanOptionalAction, default=True,
                   help="Same noise every pass/frame (deterministic, DLSS 5-like). "
                        "--no-fixed-seed draws fresh noise per pass/frame.")


def add_output_args(p: argparse.ArgumentParser) -> None:
    """Output location and device placement."""
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: output_<model>)")
    p.add_argument("--offload", action="store_true",
                   help="CPU model offloading for smaller GPUs (slower)")


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Model selection, per-pass inference, geometry, seeding, offloading."""
    add_model_args(p)
    add_pass_args(p)
    add_geometry_args(p)
    add_output_args(p)
