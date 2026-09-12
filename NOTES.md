# Notes

Loose ends and follow-ups for imgiter.

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

Preview settings for reference: stateful, `--reproject`,
`--max-frames 30 --tail-frames 10 --tail-modes freeze`.

