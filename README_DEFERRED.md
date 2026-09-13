# Deferred / rejected pod decisions

Parked or rejected ideas around the pod setup. Nothing here is scheduled;
items under *Rejected* are decided unless new information arrives.

## Parked

### Add `torchvision` (flips the preprocessing backend)

NOTES.md, "Runtime log noise" item 3: without torchvision, transformers falls
back to `CLIPImageProcessorPil` — slightly slower preprocessing, same results.
Adding it to `pyproject.toml`/`uv.lock` switches the backend back, so runs
from before/after must not be mixed (reproducibility); ship it with the next
`uv.lock` change that happens anyway. Do it only if preprocessing ever becomes
a bottleneck or a correctness question.

### Re-pin the pod image off the rc tag

The pod image is stock `runpod/base:1.3.0-rc.164-ubuntu2404`, pinned by digest
(immutable), so this is not urgent; when a stable `1.3.0`+ tag exists, re-pin
the template / `pod create` image reference and re-check the uv version pinned
in `scripts/setup-pod.sh` (constraint `>=0.12.7,<0.13.0`) at the same time.

### Exercise the Jupyter option

Documented in README_RUNPOD.md §3 as untested (`JUPYTER_PASSWORD` + port
`8888/http`). Try it once on a live pod, then either bless it in the runbook
or drop the section.

## Rejected

### Build a custom pod image (retired 2026-09-13)

The first commissioning baked the locked venv into a ~12 GB custom image
(published as `refinementsystems/imgiter` on Docker Hub; the old tags remain
there, unchanged) to skip the multi-GB `uv sync` on pod boot. Retired because
Runpod starts billing when the image pull starts: the ~12 GB pull (from Docker
Hub, often throttled) cost more billed GPU time than the ~6 GB PyPI sync it
skipped — and every `uv.lock` change forced an emulated linux/amd64 rebuild, a
push, and a template digest re-pin. The stock base image + `scripts/setup-pod.sh`
(README_RUNPOD.md §4) does the same job with no build train. For scale: the
~87.5 GB of HF model downloads every fresh pod pays dwarfs both sides of the
trade anyway.
