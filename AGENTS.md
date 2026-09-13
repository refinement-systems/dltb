# AGENTS.md

Instructions for AI agents working in this repository (repo: `imgiter`, package: `dltb`).

## What this is

DLTB ("Deep Learning Tripping Balls") — experiments in feeding diffusion models
their own output: free-running image self-iteration, video pipeline simulation
with carried state, and failure tails. Research/hobby code, not a product.
Two domains exist:

- **Local (macOS)**: editing, bundling (`just bundle`), image building. No GPU
  needed for `--help`/arg checks (heavy imports are deliberately lazy).
- **Pod (Runpod, linux/amd64)**: actual GPU runs via the bundle workflow.

Read `NOTES.md` before changing anything nontrivial — it records hard-won
investigations (pod SSH, AppleDouble tar pollution, klein guidance being inert)
and design sketches (e.g. klein dual-reference conditioning, not yet implemented).

## Commands

Package management is `uv` only (Python 3.13, CUDA-enabled torch comes via
`uv sync`; there is no requirements.txt). Never edit `uv.lock` casually; the
Dockerfile bakes it and rebuilds are only needed when it changes.

```bash
uv sync                                      # set up env
uv run dltb-oneshot --help                   # cheap local sanity check (no CUDA)
uv run dltb-iterate --model sd-turbo --input input_example/test_512.png --iterations 3
just bundle                                  # stage pod-ready tarball in bundle/
just image-build <registry>/<name>:<tag>     # build pod image (linux/amd64)
just hf-status | hf-keep <model> | hf-clean  # HF cache management
```

Experiment drivers (bash, env-var configurable, all support `DRY_RUN=1`):

```bash
DRY_RUN=1 scripts/sweep.sh                   # video sweep: boil test, blends, tails
DRY_RUN=1 scripts/sweep-klein.sh             # klein prompt ladder + steps probes +
                                            # reproject A/B (default under dual-ref)
DRY_RUN=1 scripts/sweep-prompt.sh            # free-running prompt x strength sweep on one image
scripts/smoke.sh                             # tiny run of every tool; needs GPU
                                            # (SMOKE_KLEIN=1 adds dltb-klein, both conditionings)
python3 src/dltb/analyze_drift.py <frames_dir>   # CPU-only drift metrics
```

The drivers read their input files (`IMG`/`CLIP`) from `input/inputs.env`,
falling back to the tracked `input_example/inputs.env`; environment variables
win (see `scripts/inputs.sh`).

There is **no test suite and no linter**. Verification ladder:
1. `uv run <tool> --help` locally (catches import/arg breakage without CUDA).
2. `scripts/smoke.sh` on the pod after deploying a bundle.
3. `analyze_drift.py` on produced frames for sanity of results.

## Architecture

Shared library in `src/dltb/` + four console scripts (see `[project.scripts]`
in `pyproject.toml`):

| File | Role |
| --- | --- |
| `models.py` | `ModelSpec`/`MODELS` table (5 models), `load_pipeline`, fast-fail checks, geometry rules |
| `imaging.py` | `run_pass` (one model pass), `prepare_frame`, optical-flow reprojection |
| `output.py` | run-directory layout (`output_<model>/<stem>_<tag>/`), timelapse assembly |
| `args.py` | argparse flag groups shared by the tools |
| `oneshot.py` / `iterate.py` | single pass / free-running self-iteration |
| `continuous.py` | video loop: anchored boil test vs stateful blend, reproject, freeze/free/black tails |
| `assemble.py` | CPU-only local mp4 assembly from a run's saved frames (split workflow; `dltb-assemble`) |
| `klein.py` | klein-restricted wrapper; dual-ref conditioning hook; delegates to `continuous.run()` |

Key invariants:

- **Heavy imports (torch, PIL, cv2, numpy) stay inside functions** so `--help`
  and arg validation never require a CUDA environment. Do not hoist them.
- Model differences are encoded in `ModelSpec` (`uses_strength`, `pass_size`,
  `gated`, dtype/variant), not in `if model == ...` at call sites. Adding a
  model = adding a `MODELS` entry (plus docs/tables in README.md).
- `dltb-klein` shares `continuous.run()` wholesale but customizes conditioning
  via the `run(args, make_conditioning=...)` hook: `--conditioning dual-ref`
  passes `[state, frame]` as two clean reference images instead of the pixel
  blend (`--conditioning blend`, the default, exercises the default
  `_blend_conditioning`). Its argument surface also restricts models, defaults
  blend to 0.1, and warns about inert guidance.
- Run-directory tags encode only mode/blend-or-conditioning/tails
  (dual-ref: `..._dualref[-norepro]_tails…`). Runs differing in other knobs
  (prompt, strength, steps, `--ref-order`) must use `--output-dir` subtrees or
  they silently overwrite earlier results (see how `sweep.sh` does it).
- User-facing errors fail fast via `raise SystemExit("message")`.
- Most modules and scripts carry the ISC license header; keep it on new files
  in `src/dltb/` and `scripts/`.
- Long module docstrings double as `--help` text
  (`RawDescriptionHelpFormatter`); update them with behavior.

## Model gotchas (verified, see NOTES.md)

- `--guidance-scale > 1` is **inert** for the step-distilled klein models (CFG
  hard-disabled, no guidance embedding); `--negative-prompt` equally so.
  dltb-klein warns once; never "fix" this by removing the warning.
- `num_inference_steps * strength >= 1` is enforced (diffusers would run 0
  denoise steps). Defaults 4 × 0.4 = exactly 1 actual step for img2img models;
  klein runs all 4 steps (no strength).
- `--width/--height` must be multiples of 16; SD/SDXL reject explicit
  width/height while FLUX pipelines require them (`pass_size` flag).
- `flux2-klein-9b` is gated: needs `HF_TOKEN` (checked at startup).
- Klein is a reference-image editor: no `--strength`; the prompt is the
  per-pass edit-strength knob; blend active range is far below img2img models.

## Local (macOS) gotchas

- **Any tar creation needs `COPYFILE_DISABLE=1`** or macOS pollutes archives
  with `._*` AppleDouble members (`bundle.sh` does this + a python3 `tarfile`
  verification guard; never verify with `tar -t`, it hides them).
- Bundles ship **tracked files only**, with one exception: gitignored `input/`
  (user inputs) is packed explicitly by `bundle.sh`. `git add` new scripts
  before `just bundle`, or the pod silently misses them (both scripts warn);
  image builds remain tracked-only.
- `.gitignore` covers `output/`, `bundle/`, `._*`, `.DS_Store`, `.pi/` — keep
  generated artifacts out of git.

## Pod workflow (see README_RUNPOD.md for the full runbook)

```bash
just bundle                     # local
runpodctl send bundle/imgiter-<stamp>.tar.gz
# on pod:
cd /workspace && tar xzf imgiter-<stamp>.tar.gz && cd imgiter-<stamp>
uv sync --frozen                # re-points baked venv (/opt/imgiter/.venv) at new code
scripts/smoke.sh && scripts/sweep.sh
```

- Pod image (`image/Dockerfile`) = runpod/base + uv-locked deps; rebuild only
  when `uv.lock` changes. Code ships via bundles.
- `just` and `runpodctl` are NOT in the pod image — use `scripts/*.sh` directly.
- 48 GB VRAM recommended; `--offload` for the two big models on smaller cards.
- Container disk is **ephemeral on stop AND restart** — copy `output/` out
  before stopping. Pod sshd isn't started by default (NOTES.md has the fix).
- HF cache: swept models are kept by default (all five ≈ 87.5 GB fit the 150 GB
  pod disk); `EVICT_CACHE=1` (env or `inputs.env`, like `IMG`/`CLIP`) restores
  keep-one eviction at sweep model boundaries. `hf-cache.sh keep/clean` refuse
  to run without `HF_HOME` set — keep that safety guard.
