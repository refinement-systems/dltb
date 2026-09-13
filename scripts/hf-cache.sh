#!/usr/bin/env bash

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

# hf-cache.sh -- inspect and evict the HuggingFace cache used by dltb.
#
# The sweeps run one model at a time, so the cache only ever needs to hold the
# model currently being swept. `keep <model-key>` evicts every other cached
# model; `clean` drops everything. `status` is read-only.
#
# Model keys are the same values as the tools' --model flag; the key -> repo
# id mapping is read from dltb.models.MODELS so it cannot drift from the
# tools. The command is run through `uv run --frozen`, i.e. the project
# `.venv` on a pod (created by `scripts/setup-pod.sh`).
#
# Usage:
#   scripts/hf-cache.sh status              # sizes per repo, filesystem free
#   scripts/hf-cache.sh keep <model-key>    # evict every other cached model
#   scripts/hf-cache.sh clean               # drop every cached model + xet
#
# Environment:
#   HF_HOME   cache root (default: $HOME/.cache/huggingface, HF's own default)
#   DRY_RUN=1 print what would be evicted, change nothing
#   DEBUG=1   extra detail (kept/skip decisions, xet left in place)
#   FORCE=1   allow destructive commands while HF_HOME is unset (see guard)
#
# Safety: destructive commands refuse to run while HF_HOME is unset, so a
# mistaken local `just hf-clean` cannot wipe the machine's default HF cache
# (on a Runpod pod the base image always sets HF_HOME=/workspace/.cache/...).

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN="${DRY_RUN:-0}"
DEBUG="${DEBUG:-0}"
FORCE="${FORCE:-0}"

if [[ -n "${HF_HOME:-}" ]]; then
    ROOT="${HF_HOME%/}"
else
    ROOT="${HOME}/.cache/huggingface"
fi
HUB="${ROOT}/hub"

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

debug() {
    [[ "$DEBUG" == "1" ]] && echo "hf-cache: debug: $*"
    return 0
}

kb_human() {
    awk -v kb="${1:-0}" 'BEGIN {
        split("B KB MB GB TB PB", u, " ");
        i = 2;  # input is in KB
        while (kb >= 1024 && i < 6) { kb /= 1024; i++ }
        if (i == 2) printf "%d%s", kb, u[i];
        else printf "%.1f%s", kb, u[i];
    }'
}

dir_kb() {
    local kb
    kb="$(du -sk "$1" 2>/dev/null | awk 'NR==1 {print $1}')" || true
    echo "${kb:-0}"
}

free_kb() {
    local probe="$ROOT"
    while [[ ! -d "$probe" && "$probe" != "/" ]]; do
        probe="$(dirname "$probe")"
    done
    df -Pk "$probe" 2>/dev/null | awk 'NR==2 {print $4}'
}

resolve_repo_id() {
    local key="$1"
    MODEL_KEY="$key" uv run --quiet --frozen python -c '
import os
import sys

from dltb.models import MODELS

key = os.environ["MODEL_KEY"].strip().lower().replace("_", "-")
spec = MODELS.get(key)
if spec is None:
    sys.exit(
        "hf-cache: unknown model key %r (choose from: %s)"
        % (key, ", ".join(sorted(MODELS)))
    )
print(spec.model_id)
'
}

status() {
    echo "hf-cache: status root=${ROOT}"
    if [[ ! -d "$ROOT" ]]; then
        echo "hf-cache: no cache yet"
        return 0
    fi
    echo "hf-cache: filesystem free: $(kb_human "$(free_kb)")"

    local lines="" total=0 d name kb
    shopt -s nullglob
    for d in "$HUB"/models--* "$HUB"/datasets--* "$ROOT"/xet; do
        [[ -d "$d" ]] || continue
        name="$(basename "$d")"
        kb="$(dir_kb "$d")"
        total=$((total + kb))
        lines="${lines}${kb}\t${name}\n"
    done
    if [[ -z "$lines" ]]; then
        echo "hf-cache: cache is empty"
        return 0
    fi
    printf '%b' "$lines" | sort -rn | while IFS=$'\t' read -r kb name; do
        printf '  %8s  %s\n' "$(kb_human "$kb")" "$name"
    done
    echo "hf-cache: total $(kb_human "$total")"
}

keep() {
    local key="${1:?usage: hf-cache.sh keep <model-key>}"
    local start
    start="$(date +%s)"

    local repo
    repo="$(resolve_repo_id "$key")"
    local target="${HUB}/models--${repo//\//--}"
    echo "hf-cache: keep key='${key}' repo='${repo}'"
    echo "hf-cache: root=${ROOT} free=$(kb_human "$(free_kb)")"
    debug "target dir: ${target}"

    local evicted=0 freed=0 d name kb
    shopt -s nullglob
    for d in "$HUB"/models--*; do
        [[ -d "$d" ]] || continue
        name="$(basename "$d")"
        if [[ "$d" == "$target" ]]; then
            debug "keeping ${name} ($(kb_human "$(dir_kb "$d")"))"
            continue
        fi
        kb="$(dir_kb "$d")"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "hf-cache: [dry-run] would evict ${name} ($(kb_human "$kb"))"
        else
            rm -rf -- "$d"
            echo "hf-cache: evicted ${name} ($(kb_human "$kb"))"
        fi
        evicted=$((evicted + 1))
        freed=$((freed + kb))
    done

    if [[ -d "$target" ]]; then
        echo "hf-cache: target cached ($(kb_human "$(dir_kb "$target")"))"
    else
        echo "hf-cache: target not cached -- will download on first run"
    fi
    if [[ "$evicted" -eq 0 ]]; then
        debug "no other cached models to evict"
    fi

    local xet="${ROOT}/xet"
    if [[ "$evicted" -gt 0 && -d "$xet" ]]; then
        kb="$(dir_kb "$xet")"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "hf-cache: [dry-run] would clear xet chunk cache ($(kb_human "$kb"))"
        else
            rm -rf -- "$xet"
            echo "hf-cache: cleared xet chunk cache ($(kb_human "$kb"))"
            freed=$((freed + kb))
        fi
    elif [[ -d "$xet" ]]; then
        debug "xet cache left in place ($(kb_human "$(dir_kb "$xet")"))"
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "hf-cache: [dry-run] ${evicted} repo(s) would be evicted, $(kb_human "$freed") freed"
    else
        echo "hf-cache: evicted ${evicted} repo(s), freed $(kb_human "$freed") in $(( $(date +%s) - start ))s"
        echo "hf-cache: free now $(kb_human "$(free_kb)")"
    fi
}

clean() {
    echo "hf-cache: clean root=${ROOT} free=$(kb_human "$(free_kb)")"
    local freed=0 n=0 d name kb
    shopt -s nullglob
    for d in "$HUB"/models--* "$HUB"/datasets--* "$ROOT"/xet; do
        [[ -d "$d" ]] || continue
        name="$(basename "$d")"
        kb="$(dir_kb "$d")"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "hf-cache: [dry-run] would remove ${name} ($(kb_human "$kb"))"
        else
            rm -rf -- "$d"
            echo "hf-cache: removed ${name} ($(kb_human "$kb"))"
        fi
        freed=$((freed + kb))
        n=$((n + 1))
    done

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "hf-cache: [dry-run] ${n} item(s) would be removed, $(kb_human "$freed") freed"
    else
        echo "hf-cache: removed ${n} item(s), freed $(kb_human "$freed")"
        echo "hf-cache: free now $(kb_human "$(free_kb)")"
    fi
}

cmd="${1:-status}"
case "$cmd" in
    status)
        status
        ;;
    keep)
        if [[ -z "${HF_HOME:-}" && "$FORCE" != "1" && "$DRY_RUN" != "1" ]]; then
            echo "hf-cache: refusing to evict without HF_HOME set (would touch ${ROOT})" >&2
            echo "hf-cache: set HF_HOME=/path or FORCE=1 to override" >&2
            exit 2
        fi
        keep "${2:-}"
        ;;
    clean)
        if [[ -z "${HF_HOME:-}" && "$FORCE" != "1" && "$DRY_RUN" != "1" ]]; then
            echo "hf-cache: refusing to clean without HF_HOME set (would touch ${ROOT})" >&2
            echo "hf-cache: set HF_HOME=/path or FORCE=1 to override" >&2
            exit 2
        fi
        clean
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "hf-cache: unknown command: ${cmd} (status | keep <model-key> | clean)" >&2
        exit 2
        ;;
esac
