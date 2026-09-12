# DLTB - Deep Learning Tripping Balls

Warning: everything in this repo is vibecoded slop, proceed with caution.

This is an attempt to have DLSS5 at home, but weirder.

Trying to test what happens when a generative loop is fed its own output,
and what a healthy pipeline (fresh frame + carried state + motion vectors)
does instead.

## Tools

Three scripts share the library code in `src/dltb/` (`models`, `imaging`,
`output`, `args`):

- `dltb-oneshot` — single image, single model pass: the anchored regime
  reduced to its fixed point. (Anchored *iteration* with `--fixed-seed` just
  repeats the same image every pass, so one pass already says everything.)
- `dltb-iterate` — free-running image self-iteration: each pass consumes the
  model's previous output (`P_n = f(P_{n-1})`); saves every `--save-every`th
  frame and assembles an mp4 timelapse.
- `dltb-continuous` — video pipeline simulation: each source frame processed
  once, either independently (`--mode anchored`, the boil test) or with carried
  state blended into each new frame (`--mode stateful`, optical-flow
  reprojection on by default), plus failure tails that branch from the shared
  end-of-video state (`--tail-modes freeze,free,black`).
- `dltb-klein` — the same video loop, restricted to the FLUX.2 klein editors
  (`flux2-klein-4b`, ungated, default / `flux2-klein-9b`, gated). Klein is a
  reference-image editor: no `--strength`, per-pass change scales ~linearly
  with the blend (default `0.1`, far below the img2img models), and the prompt
  is the de-facto per-pass edit-strength knob (`--num-inference-steps` is the
  other). `scripts/sweep-klein.sh` walks its prompt ladder and steps probes
  (guidance is inert for klein — CFG is disabled and there is no guidance
  embedding in the distilled checkpoints).

## Models

| `--model` | Backbone | Default resolution | Notes |
| --- | --- | --- | --- |
| `sd-turbo` | U-Net / CNN | 512x512 | fp16, low VRAM |
| `sdxl-turbo` | U-Net / CNN | 768x768 | fp16 (1024-native) |
| `flux-schnell` | transformer (MM-DiT) | 768x768 | bf16 needs ~34 GB VRAM; fits 24 GB with `--offload` |
| `flux2-klein-4b` | transformer | 768x768 | Apache-2.0, ~9 GB VRAM, no `--strength` |
| `flux2-klein-9b` | transformer | 768x768 | gated, `HF_TOKEN` required, ~20-29 GB VRAM |

## Setup (NVIDIA GPU machine)

`uv` installs a CUDA-enabled PyTorch; no separate `requirements.txt` is needed.

```bash
uv sync
```

For running on Runpod (GPU sizing, storage layout, cache management), see
[README_RUNPOD.md](README_RUNPOD.md).

## Requirements

`flux2-klein-9b` requires you to log in with your HuggingFace account and
approve its license at
https://huggingface.co/black-forest-labs/FLUX.2-klein-9B, then set an
`HF_TOKEN` environment variable with an access token from
https://huggingface.co/settings/tokens (read-only is enough).

## Usage

```bash
# Single-pass baseline
uv run dltb-oneshot --model sd-turbo --input menu.png

# Free-running drift experiment (frames + timelapse.mp4)
uv run dltb-iterate --model sd-turbo --input menu.png --iterations 200

# Video pipeline simulation: stateful blend + all three failure tails
uv run dltb-continuous --model flux-schnell --input clip.mp4 --mode stateful \
    --anchor-blend 0.3 --max-frames 300 --tail-frames 60 --tail-modes freeze,free,black

# Klein editors: prompt is the per-pass edit-strength knob
uv run dltb-klein --input clip.mp4 --prompt "slightly enhance the fine details" \
    --tail-frames 60 --tail-modes freeze

# FLUX.1-schnell, CPU-offloaded to fit a 24 GB card
uv run dltb-iterate --model flux-schnell --input menu.png --offload
```

Runs land in `output_<model>/<input stem>_<tag>/` (e.g. `menu_free-running/`,
`clip_stateful-a0.3_tailsfreeze-free-black60/`): the untouched input frame,
the saved frames, and the output video(s).

Run `uv run <tool> --help` for all options (strength, steps, seed handling,
resolution, prompt, save frequency, video FPS, ...).
