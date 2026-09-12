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

"""dltb-iterate: free-running image self-iteration (the drift experiment).

Each pass consumes the model's PREVIOUS OUTPUT as its input:

    P_n = f(P_{n-1})     (the anchor is dropped)

This simulates an inference loop whose engine stopped re-supplying the frame
-- the generator runs on its own output, so progressive deformation (drift)
is the expected signature. Every pass is saved (per --save-every) and the
saved frames are assembled into an mp4 timelapse.

For contrast, `dltb-oneshot` is the anchored counterpart reduced to one pass
(anchored iteration with a fixed seed repeats the same image, so a single
pass is already its fixed point).

DLSS 5's documented inference loop is, conceptually:

    P_n = f(state(P_{n-1}), N_n, motion_vectors_n, artistic_direction)

where P is the model's own previous output ("carried temporal state") and N
the freshly rendered frame. Free-running drops N entirely -- the hypothesized
buffer-echo bug. Video inputs (real N_n streams, motion, tails) are handled
by dltb-continuous.

Example:
    uv run dltb-iterate --model flux-schnell --input menu.png --iterations 200

    # Budget mode for a 24 GB card (RTX 4090/3090): slower, but fits:
    uv run dltb-iterate --model flux-schnell --input menu.png --offload
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .args import add_common_args
from .imaging import PassSettings, make_generator, prepare_frame, run_pass
from .models import MODELS, check_requirements, load_pipeline, resolve_geometry
from .output import assemble_timelapse, run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p)
    p.add_argument("--input", required=True,
                   help="Path to the input image (png/jpg/...)")
    p.add_argument("--iterations", type=int, default=200,
                   help="Number of passes through the model")
    p.add_argument("--save-every", type=int, default=1,
                   help="Save every Nth frame (1 = save all)")
    p.add_argument("--video-fps", type=int, default=30,
                   help="FPS of the assembled mp4 timelapse")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    spec = MODELS[args.model]
    width, height = resolve_geometry(spec, args.width, args.height)
    check_requirements(spec, args.num_inference_steps, args.strength)

    tag = f"{Path(args.input).stem}_free-running"
    out_dir = run_dir(args.output_dir, args.model, tag)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(spec, args.offload)
    settings = PassSettings(prompt=args.prompt,
                            num_inference_steps=args.num_inference_steps,
                            strength=args.strength,
                            guidance_scale=args.guidance_scale)

    from PIL import Image
    original = prepare_frame(Image.open(args.input), width, height)
    original.save(out_dir / "frame_0000_original.png")
    current = original

    for i in range(1, args.iterations + 1):
        result = run_pass(pipe, spec, settings, current,
                          make_generator(args.seed, args.fixed_seed, i),
                          width, height)
        current = result

        if i % args.save_every == 0:
            result.save(frames_dir / f"frame_{i:04d}.png")
        if i % 10 == 0 or i == 1:
            print(f"pass {i}/{args.iterations}", flush=True)

    assemble_timelapse(frames_dir, out_dir / "timelapse.mp4", args.video_fps)
    print("Done.")


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
