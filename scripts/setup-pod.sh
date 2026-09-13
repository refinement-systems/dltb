#!/usr/bin/env bash

# Permission to use, copy, modify, and/or distribute this software for
# any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#
# setup-pod.sh -- pod bootstrap: pinned uv + the locked dependency venv.
#
# The pod runs the stock runpod/base image, pinned by digest (README_RUNPOD.md
# §3) -- no custom image is built: Runpod starts billing when the image pull
# starts, so a ~12 GB baked-deps image cost more billed pull time than the
# ~6 GB `uv sync` it was meant to skip (NOTES.md, 2026-09-13). runpod/base
# ships neither uv nor Python 3.13, so this script installs the one and
# provisions the other:
#
#   1. uv 0.12.13 -- the version uv.lock was produced with, and inside the
#      uv_build backend constraint (>=0.12.7,<0.13.0) -- from the GitHub
#      release tarball into /usr/local/bin (static binaries, no curl|sh).
#      Skipped when that exact version is already installed.
#   2. `uv sync --frozen` -- creates .venv in this tree, downloading CPython
#      3.13 (per .python-version) and the locked torch stack (~6 GB of
#      wheels, a couple of minutes from PyPI; the wheels bundle the whole
#      CUDA userspace, so only the host driver version matters).
#
# Run from the extracted bundle root right after extraction, and re-run after
# every new bundle extract over the same tree (idempotent; with an unchanged
# uv.lock the re-run only re-points the editable install, in seconds):
#
#     cd /workspace && mkdir -p imgiter
#     tar xzf imgiter-<stamp>.tar.gz --strip-components=1 -C imgiter
#     cd imgiter && scripts/setup-pod.sh
#
# Linux only. The wheel cache lands under UV_CACHE_DIR (runpod/base default:
# /workspace/.cache/uv, i.e. the ephemeral container disk) -- expect a fresh
# sync on every new pod, same as for the HF model cache.

set -euo pipefail

UV_VERSION="0.12.13"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "setup-pod: for the Runpod pod (Linux), not $(uname -s)" >&2
    exit 1
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. uv, pinned to the lockfile's version.
if command -v uv >/dev/null 2>&1 && [[ "$(uv --version)" == "uv ${UV_VERSION} "* ]]; then
    echo "setup-pod: uv ${UV_VERSION} already at $(command -v uv)"
else
    [[ ${EUID} -eq 0 ]] || { echo "setup-pod: need root to install uv into /usr/local/bin" >&2; exit 1; }
    command -v curl >/dev/null || { echo "setup-pod: curl not found" >&2; exit 1; }

    case "$(uname -m)" in
        x86_64)  uv_arch="x86_64-unknown-linux-gnu" ;;
        aarch64) uv_arch="aarch64-unknown-linux-gnu" ;;
        *) echo "setup-pod: unsupported architecture $(uname -m)" >&2; exit 1 ;;
    esac

    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT
    echo "setup-pod: installing uv ${UV_VERSION} (${uv_arch}) -> /usr/local/bin"
    curl -fL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${uv_arch}.tar.gz" \
        | tar xz -C "${tmp}"
    install -m 0755 "${tmp}/uv-${uv_arch}/uv" /usr/local/bin/uv
    if [[ -f "${tmp}/uv-${uv_arch}/uvx" ]]; then
        install -m 0755 "${tmp}/uv-${uv_arch}/uvx" /usr/local/bin/uvx
    fi
fi

# 2. The locked environment into ./.venv (default project venv location, so
# plain `uv run` from any later shell finds it without extra env vars).
echo "setup-pod: uv sync --frozen (first run on a fresh pod: ~6 GB of wheels)"
uv sync --frozen --no-dev

echo "setup-pod: ready -- venv at ${PWD}/.venv; next: scripts/smoke.sh"
