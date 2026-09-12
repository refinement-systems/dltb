# imgiter pod image: Runpod base + the uv-locked python environment baked in.
#
# Why bake it: a fresh pod skips the multi-GB `uv sync` (torch/diffusers stack)
# that used to run on pod boot. Code is NOT frozen into the image — the normal
# workflow (`just bundle` + `runpodctl send`) still delivers it. Extracting a
# bundle on the pod and re-running `uv sync --frozen` upgrades the code in
# seconds: UV_PROJECT_ENVIRONMENT below pins every `uv sync`/`uv run` to the
# baked venv, so only the project itself (editable install) gets re-pointed,
# never the dependencies. Rebuild the image only when uv.lock changes.
#
# Base: runpod/base (no CUDA toolkit, no torch) rather than runpod/pytorch.
#  - The toolkit is unnecessary: nothing compiles CUDA here, and the locked
#    torch 2.14 linux wheels bundle their whole CUDA 13 userspace via the
#    nvidia-*/cuda-* PyPI wheels inside the venv (that is also how the earlier
#    bundle+`uv sync` runs worked — they never used the pod's system CUDA).
#    Only the host driver must match, and that is injected per-pod by the
#    NVIDIA container toolkit regardless of base image.
#  - runpod/pytorch would additionally ship a system torch that the venv's
#    torch 2.14 shadows — ~4 GB of dead, confusing weight.
#  - runpod/base keeps the pod plumbing we DO need: /start.sh (SSH etc., kept
#    by not overriding CMD) and the /workspace cache conventions (HF_HOME,
#    UV_CACHE_DIR, HF_XET_HIGH_PERFORMANCE — already set in the base env).
# Pinned by digest: the tag is an rc and Runpod may mutate or drop it.
FROM docker.io/runpod/base:1.3.0-rc.164-ubuntu2404@sha256:95357957d7660542b37226fb31863c4510502006f2018efe23e2869137daf000

# uv, pinned to the version the lockfile was produced with (must also satisfy
# the uv_build backend constraint in pyproject.toml: >=0.12.7,<0.13.0).
# Static binaries — no python needed in the base at this point.
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /usr/local/bin/

# All `uv sync` / `uv run` invocations — build-time and pod-side — resolve to
# this one venv, instead of a project-local .venv (which, in a bundle extracted
# under /workspace, would start empty and re-download the stack).
ENV UV_PROJECT_ENVIRONMENT=/opt/imgiter/.venv

WORKDIR /opt/imgiter

# Lock-first layering: copy only the dependency inputs so this layer caches
# across code-only changes. README.md is required by uv_build (readme field),
# .python-version pins the interpreter (uv provisions CPython 3.13 itself —
# the base image ships no python 3.13).
COPY pyproject.toml uv.lock README.md .python-version ./
COPY src ./src

# --frozen       : uv.lock is the source of truth; never re-resolve in-image.
# --no-dev       : no dev deps exist today; keeps the flag honest if they appear.
# --compile-bytecode : faster interpreter startup in the image.
# UV_CACHE_DIR is redirected off the base's /workspace default for this RUN:
# a BuildKit cache mount keeps the ~6 GB of downloaded wheels out of the final
# image while still warming rebuilds on the build host.
RUN --mount=type=cache,target=/tmp/uv-cache \
    UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen --compile-bytecode --no-dev

# The imgiter CLI (and venv tools) on PATH. After a bundle overlay + `uv sync`,
# this same entry point runs the freshly sent code.
ENV PATH="/opt/imgiter/.venv/bin:${PATH}"

# no CMD override: inherit /start.sh (SSH + pod bootstrap) from the base image.
