# Notes

Loose ends and follow-ups for imgiter.

## Pod SSH: sshd is NOT running by default (2026-09-12)

**Symptom:** every non-interactive access path fails while the pod itself is
healthy — `runpodctl ssh info` ip:port gives *connection refused*, `exec`
python tunnels over that same mapping, and croc relays are independently
flaky (see 2026-09-12 transfer session: two dead relays, one Go panic in
runpodctl's croc client, then a 10-min receive timeout on a third code).
Only the console **web terminal** and the `ssh <pod-id>-<token>@ssh.runpod.io`
gateway (PTY required — plain `ssh host cmd` is rejected with "Your SSH client
doesn't support PTY"; scripted use needs a pty wrapper + fed commands) work.

**Cause:** the pod image does not start sshd, and ships without host keys.

**Fix (on the pod, via web terminal or the gateway):**

```bash
ssh-keygen -A          # generates /etc/ssh/ssh_host_* keys
service ssh start      # "no hostkeys available -- exiting" without the line above
ss -tlnp | grep :22
```

Then the `runpodctl ssh info` ip:port mapping answers. Second gotcha: the
gateway authenticates against the *Runpod account*, the in-container sshd
against `/root/.ssh/authorized_keys` — append the pubkey you connect with
(`echo '<pubkey line>' >> /root/.ssh/authorized_keys`) or publickey auth
still fails. With that, plain `rsync -P -e "ssh -i <key> -p <port>"` works
and is the preferred bulk-transfer path (measured ~14 MB/s).

**All of it is ephemeral** — host keys, authorized_keys, and the running sshd
live on the container disk and vanish on stop/restart/recreate. Redo the
two-liner after every pod start.

## macOS `._*` AppleDouble files in pod bundles

**Status: fixed** (2026-09-12) in `scripts/bundle.sh`; `.gitignore` updated.
`scripts/image-build.sh` is **not** affected — it stages the same tree but never
creates a tar archive.

**Symptom:** extracting a bundle on a pod lists `._Justfile`, `._.gitignore`,
`.___init__.py`, `._imgiter-<stamp>`, etc. next to the real files.

**Investigation** (all verified on the Mac against
`bundle/imgiter-202609121612.tar.gz`):

1. Not committed: `git ls-files | grep -c '\._'` → `0`.
2. Not the staging step: after running bundle.sh's exact staging pipeline,
   `find "$stage" -name '._*'` → `0` files.
3. The staged files carry `com.apple.provenance`; `xattr -l` on a staged file
   shows it. macOS attaches this to files created during the extraction.
4. The **final `tar --no-xattrs -czf`** turns that xattr into AppleDouble
   members: Python `tarfile` counted **26 `._*` members out of 52**.
5. `--no-xattrs` does not prevent this. `COPYFILE_DISABLE=1` → `0` members;
   `--no-mac-metadata` → `0` members.
6. macOS `tar -tzf` **hides** `._*` members when listing, which is why the first
   "the archive is clean" check was wrong.

**Impact:** inert on Linux (ignored by `uv`, Python imports, and git), but they
bloat the archive and clutter extracted trees. They must not be committed.

**Fix applied:**

- `scripts/bundle.sh`: `export COPYFILE_DISABLE=1`, plus a post-build guard that
  verifies the finished archive with `python3`'s `tarfile`, and on failure
  deletes it and exits non-zero. The guard skips with a warning if `python3` is
  unavailable.
- `.gitignore`: added `._*`.

**Lessons:**

- Never verify AppleDouble members on macOS with `tar -t`/`tar -tzf`; use Python
  `tarfile`.
- Any future tar creation run on macOS needs `COPYFILE_DISABLE=1`.

## Runtime log noise (benign, optional cleanup)

Observed during the first full sweep on the RTX 6000 Ada. None of these affect
results; they are candidates for a cleanup pass on the next CLI/bundle change.
Note that suppressing any of them requires a new bundle and a sweep restart.

### 1. `There are modules in AutoencoderKL that should be kept in float32: []`

**Verdict: harmless diffusers false positive.** Fires roughly twice per decoded
frame for the SD/SDXL models (thousands of lines per run); does not appear for
the FLUX/FLUX-2 models.

Mechanism:

- `diffusers/models/modeling_utils.py` (~line 1523 in the installed version) has
  a buggy guard:
  ```python
  fp32_modules = self._keep_in_fp32_modules or []      # never None
  if dtype_present_in_args and fp32_modules is not None:  # always true
      logger.warning(f"... should be kept in float32: {fp32_modules} ...")
  ```
  `AutoencoderKL` does not define `_keep_in_fp32_modules`, so the message prints
  `[]` — nothing actually needs special handling.
- The warnings come from the SD/SDXL pipeline's intentional VAE upcast around
  decode: `pipeline_stable_diffusion_xl_img2img.py` (~lines 1447–1475) does
  `self.upcast_vae()` (`vae.to(dtype=torch.float32)`) before `vae.decode(...)`
  and `self.vae.to(dtype=torch.float16)` after, because the VAE overflows in
  fp16. Both `.to(dtype=...)` calls hit the buggy guard.
- `sd-turbo` and `sdxl-turbo` both have `"force_upcast": true` in their VAE
  `config.json`.

Outputs were verified healthy (frame mean/stddev in normal ranges, no black or
NaN frames). To silence later, add a targeted filter in `dltb/models.py`
(inside `load_pipeline`, before the pipeline loads):

```python
import logging

class _DropFp32FalsePositive(logging.Filter):
    def filter(self, record):
        return "should be kept in float32" not in record.getMessage()

logging.getLogger("diffusers.models.modeling_utils").addFilter(_DropFp32FalsePositive())
```

### 2. `torch.jit.script` is deprecated (FutureWarning)

From diffusers internals (`torch/jit/_script.py` triggered inside the
pipelines). Benign, no action planned beyond upstream updates.

### 3. `requires torchvision (not installed); falling back to CLIPImageProcessorPil`

`torchvision` is not in the image, so transformers falls back to the PIL image
processor (`CLIPImageProcessorPil`, `SiglipImageProcessorPil`). Slightly slower
preprocessing, same results. Two cautions:

- Reproducibility: stay on the same image for a given experiment. Adding
  `torchvision` later would switch the preprocessing backend back, so do not mix
  runs from before/after such a change.
- Adding `torchvision` to `pyproject.toml` is the clean long-term fix if the
  fallback ever becomes a bottleneck or a correctness question.

### 4. `upcast_vae` deprecation

The SDXL pipeline calls the deprecated `upcast_vae()` helper internally; if the
`deprecate` line shows up it is internal diffusers churn, not our call site.

### 5. `Siglip2ImageProcessorFast is deprecated`

Transformers deprecation of the `Fast` suffix on image processors, emitted
while loading pipelines that use Siglip/Siglip2 encoders (the FLUX.2 klein
models). Internal, benign; fixed by a transformers upgrade.

### 6. `hf_hub_download ... local_dir_use_symlinks is deprecated and ignored`

`UserWarning` from `huggingface_hub/utils/_validators.py`. The argument is a
no-op in current huggingface_hub; something further up the pipeline stack still
passes it. Benign; disappears when that caller is updated.

### 7. `You have disabled the safety checker ... safety_checker=None`

Printed once per SD/SDXL pipeline load: those model repos ship
`safety_checker: null` in `model_index.json` and the CLI never requests one, so
diffusers emits the license reminder. Benign, expected for `sd-turbo` and
`sdxl-turbo`; it is not something the CLI can (or should) silence.

### 8. `Guidance scale 2.0 is ignored for step-wise distilled models.`

**Not benign — it means the run is a no-op duplicate.** Emitted once per pass
by `Flux2KleinPipeline.check_inputs` whenever `guidance_scale > 1.0` with a
klein checkpoint (360 lines per probe run in the sweep log). The value is
dropped on the floor: see
[FLUX.2 klein: --guidance-scale is inert](#flux2-klein---guidance-scale-is-inert-step-wise-distilled)
below.

## flux2-klein-9b anchor-blend calibration (paused — needs a redesign)

Context: for `sd-turbo` / `sdxl-turbo` / `flux-schnell`, the stateful sweep at
`--anchor-blend 0.1/0.3/0.5` produced an effect judged too strong (a
"cartoonish" degradation), so the re-sweep grid was set to
`BLENDS="0.6 0.7 0.8"` (`BASELINE=0.7`) for every model.

`flux2-klein-9b` did not follow that pattern:

- preview at 0.6/0.7/0.8: **too weak** to be useful;
- rerun at 0.1: a **noticeable effect but qualitatively different** — "melting"
  rather than the cartoonish degradation the other models show — and it drops
  off.

Status: experiments with this model are paused. The stateful-loop topology needs
a return to the drawing board for `flux2-klein-9b` before it can go into the
cross-model comparison. No conclusion yet on whether the model is suitable at
all, or whether a different conditioning/topology is needed (`flux2-klein-9b`
is a reference-image editor: no `--strength`, full 4-step regeneration).

2026-09-21: that topology redesign is now in the tree as `dltb-klein` +
`scripts/sweep-klein.sh` (prompt-as-strength ladder, guidance probes). The
pre-restructure scripts they were derived from live under `reference/`.
Later the same day the guidance-probe leg turned out to be inert for klein —
see the next section.

Preview settings for reference: stateful, `--reproject`,
`--max-frames 30 --tail-frames 10 --tail-modes freeze`.

## FLUX.2 klein: `--guidance-scale` is inert (step-wise distilled)

**Found 2026-09-12, mid klein sweep:** `scripts/sweep-klein.sh` reached its
guidance probes (`enhance-slight` at `--guidance-scale 2.0` / `4.0`) and the
log flooded with one `Guidance scale 2.0 is ignored for step-wise distilled
models.` warning per pass. Verified against the deployed diffusers (0.40.0,
`diffusers/pipelines/flux2/pipeline_flux2_klein.py`) — the value provably
never reaches the model, via three independent points:

1. `check_inputs` warns exactly when `guidance_scale > 1.0 and
   self.config.is_distilled` — klein checkpoints ship `is_distilled: true`.
2. `do_classifier_free_guidance` is `self._guidance_scale > 1 and not
   self.config.is_distilled` — always `False` for klein, and the CFG branch
   (`noise_pred + scale * (noise_pred - neg_noise_pred)`) is the **only**
   consumer of `guidance_scale` in the pipeline.
3. Unlike FLUX.1-dev there is no guidance-embedding fallback: the transformer
   is called with `guidance=None` unconditionally.

**Consequences:**

- A `--guidance-scale 2.0`/`4.0` run is bit-identical to the same-prompt
  default-guidance run (fixed seed) — the probes were duplicates of the
  `prompt-enhance-slight` run and measured nothing (~360 passes each).
- `--negative-prompt` is equally inert (negative embeddings are only computed
  under CFG).
- The sweep was killed mid-probe; the meaningful legs (prompt ladder,
  weathering attractor) were already on disk. `output/flux2-klein-9b/guidance2.0/`
  is a partial duplicate of `prompt-enhance-slight/` — delete it (and
  `guidance4.0/` if it ever started).

**Follow-ups applied 2026-09-13:** the guidance leg of `sweep-klein.sh` is
replaced by a `--num-inference-steps` probe (`STEPS="2 8"`, bracketing the
default 4 that the ladder legs already run) — steps are the one remaining
direct per-pass edit-intensity knob that actually reaches klein. `dltb-klein`
now prints a one-time warning when `--guidance-scale > 1` is passed (warn,
not refuse, so a future diffusers that implements a real guidance path does
not break the tool). The 2-frame hash A/B remains optional and only worth
doing after a diffusers upgrade. Next-experiment sketch for klein's control
problem: dual-reference conditioning, see the last section.

## Klein dual-reference conditioning (`--conditioning dual-ref`)

**Status: IMPLEMENTED 2026-09-13.** The candidate fix for klein's control
problem, replacing the pixel-blend proxy. The design below is what landed
(one deviation: `continuous.run` takes a `make_conditioning` factory that
returns `(combine, tail_source)` function pairs, rather than a single
`condition(...)` callable, because tails need their own source construction).
Runs validating it (prompt ladder + order A/B) are still pending.

**Why:** klein's calibration trouble (too weak at blend 0.6–0.8, "melting" at
0.1 — see the calibration section) is plausibly an artifact of pixel-blending
two frames into ONE reference image. Klein is trained as a (multi-)reference
editor, and the pipeline natively accepts a LIST of reference images:
verified in diffusers 0.40.0 `Flux2KleinPipeline.__call__` (step 4 — each
image is preprocessed, downscaled to ≤ 1 MP if needed, packed, and the packed
latents are `torch.cat([latents, image_latents], dim=1)`-ed on the SEQUENCE
axis; batch size comes from the prompt, not the image count). So the carried
state and the fresh frame can both be conditioning inputs:

    blend  (today) : P_n = f(image = (1-a)*R(P_{n-1}) + a*N_n)
    dual-ref (new) : P_n = f(image = [R(P_{n-1}), N_n])    # two references

**Design:**

- CLI in `dltb-klein`: `--conditioning {blend,dual-ref}` (default `blend`
  until validated). `--anchor-blend` applies to `blend` only; dual-ref run
  tag: `<stem>_dualref[-norepro]_tails…` (no blend component).
- `imaging.run_pass` needs NO change — it forwards `image=source`, and a
  list of two PIL images flows straight through. `--width/--height` still
  set the output canvas; references are resized/packed per-image by the
  pipeline (our 768² frames are under the 1 MP auto-resize cap).
- `continuous.run` was generalized (no fork): it accepts
  `make_conditioning(args, estimate_flow, warp) -> (combine, tail_source)`,
  with the former blend/reproject logic as the default
  (`_blend_conditioning`); `klein.py` supplies `_dual_ref_conditioning`.
- Reprojection: probably UNNECESSARY in dual-ref (the fresh frame is an
  explicit reference; the model aligns content, not pixel coordinates) — but
  keep it probeable: warping the carried reference may still help temporal
  stability. A/B `--reproject` / `--no-reproject`.
- Reference order is a real variable: `[P, N]` vs `[N, P]` — likely encodes
  "primary vs target"; cheap 2-frame A/Bs answer it empirically.
- Tails: freeze = `[P, last_source]`; free = `[P]` alone (single-reference
  regeneration from state — the pure buffer-echo case); black = `[P, black]`.

**Open questions / risks:**

- Token budget: each 768² reference packs to ~2.3k sequence tokens
  (2×2-packed VAE latents), so two references + text is a modest sequence —
  but measure the real VRAM/speed delta with the README_RUNPOD VRAM-probe
  pattern before scheduling runs.
- Does klein weight multiple references equally, or is there an implicit
  "first = primary" convention? (The order A/B above answers this.)
- Interaction with the steps probe: re-run the steps axis under dual-ref —
  intensity may interact with conditioning strength.

**Suggested first runs:**

    uv run dltb-klein --model flux2-klein-4b --input input/video_cropped.mp4 \
        --conditioning dual-ref --prompt "slightly enhance the fine details" \
        --max-frames 30 --tail-frames 10 --save-every 1

    # order A/B (2 frames each, compare): EXTRA_ARGS='--reproject' etc.
    uv run dltb-klein --model flux2-klein-4b --input input/video_cropped.mp4 \
        --conditioning dual-ref --ref-order state-first --max-frames 2

Remaining follow-ups: add a `SMOKE_KLEIN=1` leg to `scripts/smoke.sh` (2 frames
+ 1 tail frame, flux2-klein-4b; off by default because of the 15 GB download);
the `CONDITIONING=dual-ref REF_ORDER={state-first,frame-first}` toggle in
`scripts/sweep-klein.sh` is done (it also swaps in role-naming prompts).

