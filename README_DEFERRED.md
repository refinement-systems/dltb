# Deferred / rejected image decisions

Parked changes to the pod image (`image/Dockerfile`, published as
`refinementsystems/imgiter` on Docker Hub). Nothing here is scheduled; items
under *Rejected* are decided unless new information arrives.

Ground rules for any future rebuild:

- The pod template pins the image **by digest** and published tags are never
  mutated, so existing pods and other users cannot break. A changed image
  ships as a **new tag** (`0.2.0` for dependency/behavior changes, `0.1.1` for
  metadata-only), plus a `runpodctl template update` of the image reference.
- Prefer bundling cosmetic changes into the next rebuild that is forced anyway
  (i.e. whenever `uv.lock` changes).

## Parked

### Add `torchvision` (flips the preprocessing backend)

NOTES.md, "Runtime log noise" item 3: without torchvision, transformers falls
back to `CLIPImageProcessorPil` — slightly slower preprocessing, same results.
Adding it to `pyproject.toml`/`uv.lock` switches the backend back, so runs
from before/after must not be mixed (reproducibility); that makes it an image
`0.2.0`. Do it only if preprocessing ever becomes a bottleneck or a
correctness question.

### OCI labels + `EXPOSE 22` (cosmetic)

`org.opencontainers.image.{source,description,licenses}` (ISC) for the public
Docker Hub page; `EXPOSE 22` documents the SSH port the template maps.
Trivial — ride along with the next rebuild.

### Re-pin the base image off the rc tag

`runpod/base:1.3.0-rc.164-ubuntu2404` is pinned by digest (immutable), so
this is not urgent; when a stable `1.3.0`+ tag exists, re-pin during the next
rebuild and re-check the uv version constraint (`>=0.12.7,<0.13.0`) at the
same time.

### Exercise the Jupyter option

Documented in README_RUNPOD.md §3 as untested (`JUPYTER_PASSWORD` + port
`8888/http`). Try it once on a live pod, then either bless it in the runbook
or drop the section.

## Rejected

### Bake SSH host keys / authorized_keys into the image

A public image would give every user's pods identical host keys
(impersonation risk), and baked `authorized_keys` pin one person's key into a
public artifact. Unnecessary anyway: the base image's `/start.sh` generates
host keys at container start and builds `authorized_keys` from `$PUBLIC_KEY`
(account keys injected by Runpod at pod start). The 2026-09-12 "no sshd"
incident (NOTES.md) was a missing registered key at first boot, not an image
defect — see README_RUNPOD.md §3, "SSH access".

### Bake `runpodctl` into the image

rsync over direct SSH (automatic once account keys are registered, ~14 MB/s
measured) is the preferred transfer path; a baked CLI would duplicate it and
drift out of date.

### Bake model weights or the HF token into the image

~87.5 GB of weights, one model is gated (needs a per-pod `HF_TOKEN` anyway),
and the keep-all cache policy on the 150 GB disk already works
(README_RUNPOD.md §5).
