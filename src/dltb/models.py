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

"""Model table, pipeline loading, and shared fast-fail checks.

Every dltb-* tool selects a model by key (the --model flag); ModelSpec
records the loading options, defaults, and call conventions that differ
between families:

  classic partial-noise img2img (AutoPipelineForImage2Image)
      exposes --strength (the denoising strength per pass)
  flux2klein (Flux2KleinPipeline)
      reference-image editors: regenerate from pure noise, NO --strength

NOTE - FLUX.2 klein models: stateful loops can pass [P_{n-1}, N] as two
separate reference images instead of pixel-blending (dltb-klein
--conditioning dual-ref; the pipeline resizes/packs each reference).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Per-model loading options, defaults, and call conventions."""

    model_id: str
    pipeline: str           # "auto" -> AutoPipelineForImage2Image, "flux2klein" -> Flux2KleinPipeline
    dtype: str
    variant: str | None
    default_width: int
    default_height: int
    guidance_scale: float
    uses_strength: bool     # classic partial-noise img2img exposes --strength
    pass_size: bool = False # pipeline needs explicit width/height (FluxImg2ImgPipeline
                            # defaults to 1024x1024 otherwise; SD/SDXL take the
                            # input image's size and reject those kwargs)
    gated: bool = False     # requires accepting terms on HF + HF_TOKEN


MODELS: dict[str, ModelSpec] = {
    "sd-turbo": ModelSpec(
        model_id="stabilityai/sd-turbo",  # U-Net/CNN, 512-native, ungated
        pipeline="auto",
        dtype="float16",
        variant="fp16",
        default_width=512,
        default_height=512,
        guidance_scale=0.0,
        uses_strength=True,
    ),
    "sdxl-turbo": ModelSpec(
        model_id="stabilityai/sdxl-turbo",  # U-Net/CNN, 1024-native, ungated
        pipeline="auto",
        dtype="float16",
        variant="fp16",
        default_width=768,   # 1024-native; 768 matches the FLUX runs
        default_height=768,
        guidance_scale=0.0,
        uses_strength=True,
    ),
    "flux-schnell": ModelSpec(
        model_id="black-forest-labs/FLUX.1-schnell",  # MM-DiT, Apache-2.0
        pipeline="auto",
        dtype="bfloat16",
        variant=None,
        default_width=768,
        default_height=768,
        guidance_scale=0.0,
        uses_strength=True,
        pass_size=True,   # FluxImg2ImgPipeline defaults to 1024x1024
    ),
    "flux2-klein-4b": ModelSpec(
        model_id="black-forest-labs/FLUX.2-klein-4B",  # Apache-2.0, ungated
        pipeline="flux2klein",
        dtype="bfloat16",
        variant=None,
        default_width=768,
        default_height=768,
        guidance_scale=1.0,  # NOTE: 1.0, not 0.0, per the model card
        uses_strength=False, # reference-based editing: no partial noising
        pass_size=True,      # Flux2KleinPipeline needs explicit width/height
        gated=False,
    ),
    "flux2-klein-9b": ModelSpec(
        model_id="black-forest-labs/FLUX.2-klein-9B",  # FLUX Non-Commercial, gated
        pipeline="flux2klein",
        dtype="bfloat16",
        variant=None,
        default_width=768,
        default_height=768,
        guidance_scale=1.0,
        uses_strength=False,
        pass_size=True,      # Flux2KleinPipeline needs explicit width/height
        gated=True,
    ),
}


def model_key(value: str) -> str:
    """Accept sd_turbo / SD-Turbo as aliases for sd-turbo."""
    return value.strip().lower().replace("_", "-")


def resolve_geometry(spec: ModelSpec, width: int | None,
                     height: int | None) -> tuple[int, int]:
    """Fall back to the model's native size and enforce the /16 alignment rule."""
    width = spec.default_width if width is None else width
    height = spec.default_height if height is None else height
    if width % 16 or height % 16:
        raise SystemExit("width and height must be multiples of 16")
    return width, height


def check_requirements(spec: ModelSpec, num_inference_steps: int,
                       strength: float) -> None:
    """Fail fast: gated models without HF_TOKEN, degenerate denoise schedules."""
    if spec.gated and not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            f"{spec.model_id} is gated on Hugging Face: accept the license terms on "
            "the model page, then re-run with HF_TOKEN set."
        )
    if spec.uses_strength and num_inference_steps * strength < 1:
        raise SystemExit(
            "num_inference_steps * strength < 1: diffusers would run 0 denoise steps "
            "per pass. Raise --strength or --num-inference-steps."
        )


def load_pipeline(spec: ModelSpec, offload: bool):
    """Load the diffusers pipeline for a spec onto the GPU (or CPU-offloaded)."""
    import torch
    from diffusers import AutoPipelineForImage2Image, Flux2KleinPipeline

    load_kwargs = {"dtype": getattr(torch, spec.dtype)}
    if spec.variant:
        load_kwargs["variant"] = spec.variant

    if spec.pipeline == "flux2klein":
        # Requires a recent diffusers (>=0.40); if the import fails:
        #   uv add git+https://github.com/huggingface/diffusers.git
        pipe = Flux2KleinPipeline.from_pretrained(spec.model_id, **load_kwargs)
    else:
        pipe = AutoPipelineForImage2Image.from_pretrained(spec.model_id, **load_kwargs)

    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe
