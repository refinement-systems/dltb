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

"""argparse flag groups shared by every dltb-* tool.

Tool-specific flags (--input, --iterations, --mode, tails, ...) are added by
each tool on top of these; help texts here apply verbatim everywhere.
"""

from __future__ import annotations

import argparse

from .models import MODELS, model_key


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Model selection, per-pass inference, geometry, seeding, offloading."""
    p.add_argument("--model", required=True, type=model_key, choices=sorted(MODELS),
                   help="Which model to run")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: output_<model>)")
    p.add_argument("--strength", type=float, default=0.4,
                   help="img2img denoising strength per pass (sd/sdxl-turbo and "
                        "flux-schnell only). Keep num_inference_steps * strength >= 1.")
    p.add_argument("--num-inference-steps", type=int, default=4,
                   help="Denoise schedule length (all models distilled for 1-4 steps)")
    p.add_argument("--prompt", default="",
                   help="Optional text prompt. A faithful description acts as a "
                        "semantic anchor (analogue of DLSS 5's artistic-direction "
                        "conditioning).")
    p.add_argument("--width", type=int, default=None,
                   help="Output width (multiple of 16; frames center-cropped to this "
                        "aspect, then resized)")
    p.add_argument("--height", type=int, default=None,
                   help="Output height (multiple of 16)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--fixed-seed", action=argparse.BooleanOptionalAction, default=True,
                   help="Same noise every pass/frame (deterministic, DLSS 5-like). "
                        "--no-fixed-seed draws fresh noise per pass/frame.")
    p.add_argument("--offload", action="store_true",
                   help="CPU model offloading for smaller GPUs (slower)")
