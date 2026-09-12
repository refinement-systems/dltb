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

"""Frame handling and single-model-pass execution shared by the dltb-* tools.

Heavy imports (torch, PIL, cv2) stay inside functions so that --help and
argument validation never require a CUDA environment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PassSettings:
    """Loop-topology-independent inference settings for one model pass."""

    prompt: str = ""
    num_inference_steps: int = 4
    strength: float = 0.4


def make_generator(seed: int, fixed_seed: bool, i: int = 0):
    """Fresh CUDA generator; i varies the seed when fixed_seed is off.

    fixed_seed=True reuses the same noise every pass (deterministic,
    DLSS 5-like: drift is then purely model bias); False draws fresh noise
    per pass index i.
    """
    import torch
    return torch.Generator("cuda").manual_seed(seed if fixed_seed else seed + i)


def run_pass(pipe, spec, settings: PassSettings, source, generator,
             width: int, height: int):
    """One pass through the model: source image in, processed PIL image out."""
    call_kwargs = dict(
        prompt=settings.prompt,
        image=source,
        num_inference_steps=settings.num_inference_steps,
        guidance_scale=spec.guidance_scale,
        generator=generator,
        output_type="pil",
    )
    if spec.uses_strength:
        call_kwargs["strength"] = settings.strength
    # SD/SDXL img2img infer the output size from the input image and do not
    # accept width/height; FLUX and FLUX.2 need them explicitly.
    if spec.pass_size:
        call_kwargs["width"] = width
        call_kwargs["height"] = height
    return pipe(**call_kwargs).images[0]


def center_crop_to_aspect(img, width: int, height: int):
    """Crop the largest centered region with the target aspect ratio."""
    w, h = img.size
    target = width / height
    if w / h > target:
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, h))
    new_h = int(w / target)
    y0 = (h - new_h) // 2
    return img.crop((0, y0, w, y0 + new_h))


def prepare_frame(img, width: int, height: int):
    """Normalize an input frame: RGB, center-cropped to aspect, resized."""
    from PIL import Image
    img = center_crop_to_aspect(img.convert("RGB"), width, height)
    return img.resize((width, height), Image.LANCZOS)


def make_reprojector():
    """Lazy OpenCV import with a helpful error. Returns (flow_fn, warp_fn).

    Used by dltb-continuous for stateful loops: real temporal pipelines
    (TAA, DLSS 2-5) never blend raw history -- the engine's motion vectors
    warp the carried state so stale content lands where the object has moved
    to, and only then is it combined with the new frame. Engine motion
    vectors are unavailable here, so they are estimated with optical flow
    (Farneback) between consecutive SOURCE frames:

        source = (1-a)*warp(P_{n-1}, flow N_{n-1} -> N_n) + a*N_n

    Requires: uv add opencv-python-headless
    Refinement not implemented: disocclusion rejection (forward-backward flow
    consistency; real TAA drops history where content was just revealed).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise SystemExit(
            "--reproject needs OpenCV: run 'uv add opencv-python-headless' "
            "(or pass --no-reproject for the naive history blend)."
        )

    def estimate_flow(prev_pil, next_pil):
        """Dense forward flow prev -> next (Farneback)."""
        prev = cv2.cvtColor(np.asarray(prev_pil), cv2.COLOR_RGB2GRAY)
        nxt = cv2.cvtColor(np.asarray(next_pil), cv2.COLOR_RGB2GRAY)
        return cv2.calcOpticalFlowFarneback(
            prev, nxt, None,
            pyr_scale=0.5, levels=4, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

    def warp(state_pil, flow):
        """Reproject a carried-state frame along the flow field."""
        from PIL import Image
        h, w = flow.shape[:2]
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        # state pixel at (x - dx, y - dy) moves to (x, y) in the new frame
        map_x = gx - flow[..., 0]
        map_y = gy - flow[..., 1]
        warped = cv2.remap(np.asarray(state_pil), map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(warped)

    return estimate_flow, warp
