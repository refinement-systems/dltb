# imgiter on Runpod — runbook

Practical notes from commissioning and testing the `imgiter` pod template
(`refinementsystems/imgiter`, tag `0.1.0`). Everything here was measured on a
real pod; prices and availability drift, so re-check the console.

## TL;DR

| Decision | Value |
| --- | --- |
| GPU | **48 GB VRAM** (RTX 6000 Ada / L40S / A40). 32 GB cards work only with `--offload` and run the two big models ~10× slower |
| Container disk | **150 GB**, ephemeral — holds the repo, the HF cache and the outputs |
| Volume disk | **0** (none) |
| Network volume | **none** — it would pin the pod to one datacenter and kill consumer-GPU availability |
| Cache policy | **keep all swept models** (~87.5 GB fits); `EVICT_CACHE=1` restores keep-one for small disks |
| Data copy-out | before **stop / restart / terminate** — the container disk is wiped on all three |

```bash
# local
git add scripts/hf-cache.sh scripts/sweep.sh Justfile README_RUNPOD.md  # untracked files do NOT ship
just bundle

# pod
cd /workspace && tar xzf imgiter-<stamp>.tar.gz && cd imgiter-<stamp>
uv sync --frozen
scripts/smoke.sh                 # post-deploy check (all three tools)
scripts/sweep.sh
```

## 1. GPU requirements

Measured on an RTX 5090 (32 GB), one pass per model with `--num-inference-steps 1 --strength 1.0`:

| Model | Default res | Native fit on 32 GB | Notes |
| --- | --- | --- | --- |
| `sd-turbo` | 512 | ✅ ~2.4 GB of weights | fits anywhere |
| `sdxl-turbo` | 768 | ✅ ~6.5 GB | fits 24 GB cards |
| `flux2-klein-4b` | 768 | ✅ ~14.9 GB | fits 24 GB cards |
| `flux-schnell` | 768 | ❌ OOM (~31.4 GB) | ~34 GB claimed; needs ≥ 48 GB natively |
| `flux2-klein-9b` | 768 | ❌ OOM (~32.3 GB) | needs ≥ 48 GB natively |

**Recommendation: 48 GB.** All five then run natively with no `--offload`.

Approximate rates seen (secure / community, 2026-09):

| GPU | VRAM | Secure $/hr | Community $/hr | Note |
| --- | --- | --- | --- | --- |
| A40 | 48 GB | 0.49 | — | cheapest 48 GB, Ampere, slower; stock was "High" |
| RTX 6000 Ada | 48 GB | 0.84 | 0.74 | best balance for this workload |
| L40S | 48 GB | 1.09 | 0.79 | |
| RTX 5090 | 32 GB | 0.99 | 0.69 | fast, but offload needed for the big two |
| A100 SXM | 80 GB | 1.59 | 1.39 | overkill |

The locked stack is **torch 2.14 with CUDA 13 wheels**, so Ada / Ampere / Blackwell
(RTX 5090, sm_120) are all supported. The image needs host drivers ≥ 580 for
CUDA 13; driver 595.91 was tested.

### Using a 32 GB card anyway

`--offload` (`enable_model_cpu_offload`) works, but the cost is real: on a 5090,
`flux2-klein-9b` measured **~20–25 s per frame** with offload; without it the
same pass should be ~1–2 s on a 48 GB card (≈10×). `flux-schnell` is worse in
relative terms (1 actual denoise step, so per-pass weight streaming dominates).
Run the two big models separately:

```bash
OFFLOAD=1 MODELS="flux-schnell flux2-klein-9b" scripts/sweep.sh
MODELS="sd-turbo sdxl-turbo flux2-klein-4b"   scripts/sweep.sh
```

`OFFLOAD` applies to every run of that invocation, so don't mix the two groups.

Note: `enable_model_cpu_offload` moves a whole component to the GPU when its
`forward` runs and keeps it there until the next component runs. The transfer
happens **once per pipeline call**, not per denoise step — so the tax scales with
the number of frames, not steps.

## 2. Storage

### Layout

```
/workspace                          <- container disk (ephemeral), 150 GB
  imgiter-<stamp>/                  <- extracted bundle (code only; no .venv)
    output/                         <- results, sweep logs
  .cache/huggingface/               <- HF_HOME (base image default)
    hub/models--<org>--<name>/      <- model weights
    xet/                            <- xet chunk cache, hard-capped at 10 GB
```

Dependencies live in the image at `/opt/imgiter/.venv`
(`UV_PROJECT_ENVIRONMENT`), not in the extracted tree.

The image itself does **not** count against the container disk: `df /workspace`
showed ~85 MB used with the 12 GB image present.

### Measured cache sizes (real downloads, not repo totals)

| Model | On-disk |
| --- | --- |
| `sd-turbo` | 2.4 GB |
| `sdxl-turbo` | 6.5 GB |
| `flux2-klein-4b` | 14.9 GB |
| `flux-schnell` | 31.4 GB |
| `flux2-klein-9b` | 32.3 GB |
| **all five** | **~87.5 GB** |

The default cache policy keeps every swept model (all five ≈ 87.5 GB) + xet
(≤10 GB) + outputs (a full video sweep is a few GB) — comfortable on the
150 GB disk, and revisiting an earlier model costs no re-download. With
`EVICT_CACHE=1` (keep-one, `input/inputs.env`-configurable) the steady-state
requirement drops to one model (~33 GB max) + xet + outputs ≈ **45 GB**, for
smaller container disks.

### Why no volume disk / network volume

- **Network volume** = the only way to read files while no pod is running
  (S3-compatible API), but it is pinned to one datacenter. Consumer GPUs are
  spotty enough that this is a losing trade. Also, `runpodctl` has no tier flag
  and S3 is available in select DCs only.
- **Pod volume disk** survives stop/restart, but a **stopped pod's volume cannot
  be read from outside** — no SSH, no file browser, no S3; only "your Pod can
  access the volume". You must resume the pod, and resume may be allocated
  **zero GPUs** if capacity changed.
- **Pod volume pricing** while stopped ($0.20/GB/month) is the expensive kind.

Storage billing, for reference:

| Type | Running | Stopped | Persistence |
| --- | --- | --- | --- |
| Container disk | $0.10/GB/mo | not charged | wiped on stop/restart/reset/terminate |
| Volume disk | $0.10/GB/mo | $0.20/GB/mo | survives stop/restart, dies on terminate |
| Network volume | $0.07/GB/mo | $0.07/GB/mo | independent, shareable, DC-pinned |

**Container disk is wiped on stop and restart too**, not just terminate. Copy
results out before any of those.

### S3-compatible API (if you ever use a network volume)

Bucket = network volume ID; endpoint and region = datacenter. Credentials:
`AWS_ACCESS_KEY_ID` = Runpod user id (`user_…`), `AWS_SECRET_ACCESS_KEY` = an
`rps_…` S3 API key from the console.

```bash
aws s3 cp --recursive \
  --region US-KS-2 --endpoint-url https://s3api-us-ks-2.runpod.io/ \
  s3://<network-volume-id>/output ./output
```

## 3. Pod setup

### The template is private — bring your own

The `imgiter` template is **private and stays that way on purpose**: it wires
account-internal secrets by name (`{{ RUNPOD_SECRET_HF_TOKEN }}`), and other
users have no reason to use the same secret names (no leak risk either way —
secrets resolve per account — but a public template would simply not work for
others). The **image** is public (`refinementsystems/imgiter` on Docker Hub),
so anyone can run the stack without the template:

```bash
runpodctl pod create \
  --image refinementsystems/imgiter:0.1.0 \
  --gpu-id "NVIDIA RTX 6000 Ada" \
  --container-disk-in-gb 150 \
  --ports "22/tcp" \
  --env '{"HF_TOKEN":"<your-token>"}' \
  --name imgiter --wait
```

…or recreate that as a private template of your own in the console. The rest
of this section documents the author's template as the reference
configuration.

Template `imgiter` (`04u1mmp8nf`): container disk 150 GB, no volume, env
`HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}`, port `22/tcp`, image pinned by digest
`sha256:68d934…` (tag `0.1.0`).

```bash
runpodctl pod create \
  --template-id 04u1mmp8nf \
  --gpu-id "NVIDIA RTX 6000 Ada" \
  --container-disk-in-gb 150 \
  --name imgiter --wait
```

Pass `--container-disk-in-gb 150` explicitly: the CLI flag defaults to 20, and
it is not confirmed that omitting it preserves the template's 150. Do not pass
`--volume-in-gb` (the template already has volume 0).

Verify:

```bash
runpodctl pod get <pod-id> | jq '{containerDiskInGb, volumeInGb}'   # 150, 0
# on the pod:
df -h /workspace        # overlay 150G
```

Updating the template (volume cannot be edited by `template update` — recreate
the template if you need to change it):

```bash
runpodctl template update 04u1mmp8nf \
  --container-disk-in-gb 150 \
  --env '{"HF_TOKEN":"{{ RUNPOD_SECRET_HF_TOKEN }}"}'
```

### HF_TOKEN

`flux2-klein-9b` is gated: accept the license on Hugging Face and make sure the
secret resolves. On the pod:

```bash
echo "HF_TOKEN prefix=${HF_TOKEN:0:3} len=${#HF_TOKEN}"   # expect hf_ and ~37
```

If it prints `{{...}}`, the secret did not resolve — `export HF_TOKEN=hf_...`
for the session.

### SSH access (direct ssh / rsync)

The image needs **no manual sshd setup**: the base image's `/start.sh`
generates host keys, writes `$PUBLIC_KEY` into `/root/.ssh/authorized_keys`
and starts sshd — but only when `PUBLIC_KEY` is non-empty, and Runpod injects
that variable from the **account's registered SSH keys at pod start**. A pod
that boots with none registered comes up without sshd (the console web
terminal and the `ssh <pod-id>-<token>@ssh.runpod.io` gateway still work; the
gateway requires a PTY). That, not an image defect, was the cause of the
2026-09-12 "connection refused" session in NOTES.md.

One-time setup — keys added after boot are ignored until a restart:

```bash
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub
runpodctl ssh list-keys                     # confirm it's on the account
```

Then `pod stop` / `pod start` (or create a new pod). Every subsequent start
brings sshd up automatically — nothing to redo manually:

```bash
runpodctl ssh info <pod-id>                 # ip:port + paste-ready ssh_command
rsync -P -e "ssh -i <key> -p <port>" ./dir/ root@<ip>:/workspace/dir/   # ~14 MB/s measured
```

Caveat: after a stop→start the external port is reassigned, and the first
`ssh info` can report a stale port for ~90 s — retry until a connection
succeeds.

### Jupyter Lab (optional, untested)

`/start.sh` also auto-starts Jupyter Lab on port 8888 (preferred dir
`/workspace`) whenever `JUPYTER_PASSWORD` is set in the pod env — zero image
change. Potentially handy for browsing `output/` frames and timelapses on a
running pod. **Not yet exercised on this pod**; to try it, set

```text
JUPYTER_PASSWORD=<token>
```

in the pod env and expose `8888/http` (template edit, or
`--ports "22/tcp,8888/http"` on create). Runpod then serves it at
`https://<pod-id>-8888.proxy.runpod.net`; mark the port **secure** in the
console so it requires your Runpod login instead of being open to anyone who
has the URL.

## 4. On-pod workflow

```bash
cd /workspace
tar xzf imgiter-<stamp>.tar.gz
cd imgiter-<stamp>
uv sync --frozen          # re-points the editable install from /opt/imgiter to this tree

scripts/sweep.sh          # full sweep; add OFFLOAD=1 on <48 GB GPUs for the big models
scripts/sweep-klein.sh    # klein prompt ladder + steps probes + reproject A/B (single model)
scripts/sweep-prompt.sh   # free-running prompt (x strength) sweep on one image
```

- Dependencies are baked into the image at `/opt/imgiter/.venv`
  (`UV_PROJECT_ENVIRONMENT`), so `uv sync` only rebuilds the project itself.
- **`just` is not installed in the image** — call `scripts/*.sh` directly, or use
  `uv run dltb-oneshot|dltb-iterate|dltb-continuous --help` for individual runs.
- `runpodctl` is not in the image either; install it on the pod
  (`curl -sSL https://cli.runpod.net | bash`) or use `scp` for transfers.
- Each sweep run is a fresh `uv run dltb-continuous` process, so the model is
  re-loaded from disk per run. For a 32 GB model that load is ~15–30 s per run,
  times ~6 runs per model. This is inherent to `sweep.sh`'s per-run invocation.

Inputs: the run scripts read their input files from `input/inputs.env`
(`IMG=`, `CLIP=`); without one they fall back to the tracked examples in
`input_example/` (provenance in `SOURCES.txt`). Your own files under `input/`
ride in the bundle, so the usual flow is: drop files into `input/`, copy
`input_example/inputs.env` to `input/inputs.env`, point it at them, `just
bundle`, send. A one-off run with a different clip does not need the file:
`CLIP=input/other.mp4 scripts/sweep.sh` (the environment beats the file).

Smoke test after deploying a new bundle — one tiny run of every tool
(`dltb-oneshot`, `dltb-iterate`, `dltb-continuous` both modes + tails, and the
hf-cache helper), with artifact checks. `SMOKE_KLEIN=1` adds the `dltb-klein`
leg (both blend and dual-ref conditioning, `flux2-klein-4b` by default — a
~15 GB download, which is why it is off by default):

```bash
scripts/smoke.sh
```

Set `MODEL=<key>` to whatever the cache already holds (step 0 prints it) to
skip the model re-download. A sweep-only preview is still available with
`DRY_RUN=1` (see below).

Useful `sweep.sh` env overrides: `MODELS`, `BLENDS`, `BASELINE`, `MAX_FRAMES`,
`TAIL_FRAMES`, `SAVE_EVERY`, `STRENGTH`, `STRENGTH_MODELS`, `DESC`, `OFFLOAD`,
`REPROJECT`, `CLIP`, `EXTRA_ARGS`, `DRY_RUN`, `SKIP_GPU_CHECK`.

Klein regime (`scripts/sweep-klein.sh`, drives `dltb-klein`): `MODEL`
(default `flux2-klein-9b`; `4b` is ungated), `CONDITIONING` (`blend`, the
default, or `dual-ref`), `REF_ORDER` (`state-first`, default; dual-ref only),
`REPROJECT` (`1`/`0`/`ab`; defaults to `ab` under dual-ref — every leg runs
with and without reprojection, the A/B), `BLEND` (default `0.1`), `STEPS`
(default "2 8", bracketing the default 4), plus the shared
`CLIP`/`MAX_FRAMES`/`TAIL_FRAMES`/`TAIL_MODES`/`SAVE_EVERY`/`EXTRA_ARGS`/
`DRY_RUN`/`SKIP_GPU_CHECK`. Single-model, so no hf-cache eviction between
runs.

Prompt sweep on one image (`scripts/sweep-prompt.sh`, drives `dltb-iterate`):
`MODEL` (default `sd-turbo`), `IMG`, `ITERATIONS` (default 20), `SAVE_EVERY`
(default 1 — every frame), `VIDEO_FPS`, `STRENGTHS` (default none; full
prompts × strengths cross product, classic img2img models only),
`STRENGTH_MODELS`, plus `EXTRA_ARGS`/`DRY_RUN`/`SKIP_GPU_CHECK`.

## 5. Model cache management

`scripts/hf-cache.sh`:

```bash
scripts/hf-cache.sh status              # per-repo sizes + filesystem free
scripts/hf-cache.sh keep flux-schnell   # evict every other model (+ xet)
scripts/hf-cache.sh clean               # drop all caches
DRY_RUN=1 DEBUG=1 scripts/hf-cache.sh keep flux-schnell   # preview
```

- With `EVICT_CACHE=1` (off by default — see `input_example/inputs.env`),
  `sweep.sh` calls `keep <model>` at the top of every model iteration, so the
  disk only ever holds the model being swept. All lines are prefixed `hf-cache:`
  and go into `output/sweep_*.log`. With the default `EVICT_CACHE=0` the cache
  keeps every swept model (~87.5 GB for all five, fits the 150 GB disk).
- Eviction happens **only at model boundaries**, never between the separate runs
  of one model — otherwise the same multi-GB weights would be re-downloaded once
  per run.
- Consequence: sweeping `A B A` re-downloads `A`.
- The xet chunk cache is capped at 10 GB by default
  (`HF_XET_CHUNK_CACHE_SIZE_BYTES`) and is cleared whenever a model is evicted.
- Safety: `keep`/`clean` refuse to run when `HF_HOME` is unset, so running them
  locally cannot wipe `~/.cache/huggingface`. Use `HF_HOME=... FORCE=1` to
  override deliberately.
- **Untracked files are not bundled.** Stage `scripts/hf-cache.sh` before
  `just bundle` or the pod gets a bundle without the helper (`just bundle` warns
  about this).

Disk expectations during a sweep, per model: one cache (~33 GB max) + xet +
outputs, with free space rising to ~118 GB after each model switch on a 150 GB
disk.

## 6. Getting data in and out

- Input files do not need a separate transfer: `just bundle` packs the
  gitignored `input/` directory into every bundle (tracked sample inputs live
  in `input_example/`). Alternatively `runpodctl send`/`scp` files into
  `input/` on the pod after extracting.
- `scp`/`rsync` work over the direct-SSH mapping from `runpodctl ssh info`
  (see [SSH access](#3-pod-setup) — register an account key once; rsync
  measured ~14 MB/s).
- Or `runpodctl send <path>` locally and `runpodctl receive <code>` on the pod
  (install runpodctl there first).
- Outputs live under `output/<model>/<stem>_<mode-tag>/`.
- There is **no S3 path in or out** without a network volume; with volume 0 the
  only way to retrieve results is from the running pod.

## 7. Gotchas learned the hard way

1. **Container disk is ephemeral on stop *and* restart** (not just terminate).
   With no volume attached, results and the cache are lost.
2. **Pass `--container-disk-in-gb 150` explicitly** on `runpodctl pod create`;
   the CLI flag defaults to 20 and may override the template.
3. **`HF_HOME` defaults to `/workspace/.cache/huggingface/`** in `runpod/base`
   and `UV_CACHE_DIR` to `/workspace/.cache/uv/`. With no volume these are on the
   container disk — which is what we want; do not add an `HF_HOME` override to
   the template.
4. **Volume-size changes are impossible via `runpodctl template update`**; only
   the console or a template recreate.
5. **`just` and `runpodctl` are absent from the image.** Use the shell scripts.
6. `--num-inference-steps 4 --strength 0.4` (the sweep defaults) results in
   **1 actual denoise step** for `sd-turbo` / `sdxl-turbo` / `flux-schnell`
   (`int(4 × 0.4)`); the `flux2-klein-*` models run all 4 steps
   (`uses_strength=False`). Timing budgets differ accordingly.
7. One observed anomaly: `flux2-klein-9b` measured 32.3 GB when kept, then
   21.6 GB after three interrupted runs — ~10.7 GB of unreferenced/incomplete
   blobs were reclaimed. The pipeline loaded and generated correctly afterwards,
   so it is not corruption; treat per-model peak as ≤ ~35 GB either way.
8. Cost example: RTX 5090 at $0.99/hr, whole disk test (image pull + ~95 GB of
   downloads + a few runs) was well under an hour.
9. Downloads from Hugging Face have no egress cost; the only cost is GPU time
   while downloading.

## Appendix: quick VRAM probe

Runs one pass per model in a fresh process, reporting native fit, peak VRAM and
whether `--offload` is needed. Run it after `uv sync`, from the extracted repo
root (`input_example/test_768.png` ships in every bundle):

```bash
for m in sd-turbo sdxl-turbo flux2-klein-4b flux-schnell flux2-klein-9b; do
  for off in 0 1; do
    echo "──────── $m  offload=$off"
    MODEL="$m" OFFLOAD="$off" python - <<'PY'
import os, time, torch
from PIL import Image
from dltb.imaging import PassSettings, make_generator, prepare_frame, run_pass
from dltb.models import MODELS, load_pipeline

key, offload = os.environ["MODEL"], os.environ["OFFLOAD"] == "1"
spec = MODELS[key]
settings = PassSettings(num_inference_steps=1, strength=1.0)
w, h = spec.default_width, spec.default_height
src = prepare_frame(Image.open("input_example/test_768.png"), w, h)

t0 = time.time()
try:
    pipe = load_pipeline(spec, offload)
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() / 2**30
except Exception as e:
    print(f"RESULT {key}: LOAD FAILED ({type(e).__name__}: {str(e).splitlines()[0][:90]})")
    raise SystemExit(42)

t1 = time.time()
torch.cuda.reset_peak_memory_stats()
try:
    run_pass(pipe, spec, settings, src, make_generator(1234, True), w, h)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    free, total = (v / 2**30 for v in torch.cuda.mem_get_info())
    print(f"RESULT {key} {w}x{h}: OK  load={t1-t0:.0f}s pass={time.time()-t1:.1f}s "
          f"weights={weights:.1f}GiB peak={peak:.1f}GiB free={free:.1f}/{total:.0f}GiB offload={offload}")
except Exception as e:
    print(f"RESULT {key}: PASS FAILED ({type(e).__name__}: {str(e).splitlines()[0][:90]}) offload={offload}")
    raise SystemExit(42)
PY
    rc=$?
    [ "$rc" -eq 0 ] && break
    [ "$off" -eq 1 ] && { echo "  -> failed with offload too"; break; }
    echo "  -> retrying with --offload"
  done
done
```
