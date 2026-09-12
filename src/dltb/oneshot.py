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

"""dltb-oneshot: single image, single model pass (the anchored fixed point).

This is the "anchored" image regime reduced to its essence. In anchored mode
every pass consumes the SAME original frame N, and with --fixed-seed (the
default) the noise is identical too:

    P_n = f(N, same_noise) = P_1 for every n

-- iterating would just write N copies of one image. So this tool runs
exactly one pass and saves it:

    P = f(N)

Use it as the per-model baseline: what the model does to a frame in the
healthy, no-feedback regime. The drift experiments live in dltb-iterate
(free-running self-iteration) and dltb-continuous (video pipelines).

Example:
    uv run dltb-oneshot --model sd-turbo --input menu.png --prompt "..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .args import add_common_args
from .imaging import PassSettings, make_generator, prepare_frame, run_pass
from .models import MODELS, check_requirements, load_pipeline, resolve_geometry
from .output import run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p)
    p.add_argument("--input", required=True,
                   help="Path to the input image (png/jpg/...)")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    spec = MODELS[args.model]
    width, height = resolve_geometry(spec, args.width, args.height)
    check_requirements(spec, args.num_inference_steps, args.strength)

    tag = f"{Path(args.input).stem}_oneshot"
    out_dir = run_dir(args.output_dir, args.model, tag)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(spec, args.offload)

    from PIL import Image
    source = prepare_frame(Image.open(args.input), width, height)
    source.save(out_dir / "frame_0000_original.png")

    settings = PassSettings(prompt=args.prompt,
                            num_inference_steps=args.num_inference_steps,
                            strength=args.strength,
                            guidance_scale=args.guidance_scale)
    result = run_pass(pipe, spec, settings, source,
                      make_generator(args.seed, args.fixed_seed), width, height)
    result_path = frames_dir / "frame_0001.png"
    result.save(result_path)
    print(f"Result written to {result_path}")
    print("Done.")


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
