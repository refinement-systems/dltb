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

#
# sweep.sh -- imgiter video sweep: boil test, anchor-blend sweep with tails,
#            failure tails, semantic anchor, and strength probes.
#
# Structure (per model):
#   1. anchored boil test                      (independent frames, no tail)
#   2. stateful blend sweep over BLENDS        (freeze tail; the BASELINE blend
#                                               also gets free+black tails)
#   3. semantic anchor at the BASELINE blend   (--prompt "$DESC", freeze tail)
#   4. strength probe, classic img2img models  (--strength "$STRENGTH", freeze tail)
#
# Cache policy: by default swept models are KEPT in the HuggingFace cache (all
# five fit the 150 GB pod disk at once, ~87.5 GB, so re-runs on an earlier
# model cost nothing). EVICT_CACHE=1 restores the keep-one policy: before step
# 1 of each model, scripts/hf-cache.sh evicts every other cached model, so the
# disk only ever needs room for the model being swept (for small container
# disks). Eviction happens at model boundaries only -- never between the runs
# of one model.
#
# Stateful video runs reproject the carried state by optical flow before
# blending (--reproject, on by default in dltb-continuous):
# (1-a)*warp(P, flow) + a*N. REPROJECT=0 selects the naive history blend;
# dltb-continuous tags those directories with a -norepro suffix so the two
# variants never collide.
#
# All runs write under the gitignored output/ tree, using --output-dir:
#   output/<model>/<stem>_<mode tag>/     e.g. video_cropped_stateful-a0.3_tailsfreeze60
#   output/<model>/prompt/...             semantic-anchor variant
#   output/<model>/strength<value>/...    strength variant
# dltb-continuous's directory tag encodes only mode/blend/tails, so the prompt
# and strength runs get their own --output-dir subtrees; without that they
# would silently overwrite the baseline run's frames and videos.
#
# The failure tails (free, black) share the baseline blend run with the freeze
# tail. dltb-continuous branches every tail from the same in-memory end state,
# which as separate runs is not guaranteed bit-identical (CUDA kernels), and it
# also avoids reprocessing the source video three times.
#
# Usage (from any directory inside the repo):
#   scripts/sweep.sh
#   DRY_RUN=1 scripts/sweep.sh                 # print every command, run nothing
#   MODELS="sd-turbo" MAX_FRAMES=20 TAIL_FRAMES=5 scripts/sweep.sh   # smoke test
#
# Environment overrides:
#   CLIP              source video             (default: input/inputs.env if
#                                               present, else the tracked
#                                               input_example/video_cropped.mp4;
#                                               see scripts/inputs.sh)
#   MODELS            models to sweep          (default sd-turbo sdxl-turbo flux-schnell flux2-klein-9b)
#   BLENDS            anchor-blend values      (default 0.1 0.3 0.5)
#   BASELINE          baseline blend           (default 0.3)
#   MAX_FRAMES        source frames per run    (default 300; empty = whole clip)
#   TAIL_FRAMES       frames per tail          (default 60; 0 = no tails)
#   SAVE_EVERY        save every Nth frame     (default 10)
#   STRENGTH          strength-probe value     (default 0.55)
#   STRENGTH_MODELS   models for the probe     (default sd-turbo sdxl-turbo flux-schnell)
#   DESC              semantic-anchor prompt
#                     (default: describes the tracked example clip -- an
#                      octopus in a coral habitat; see input_example/SOURCES.txt)
#   EVICT_CACHE       1 = keep-one cache policy: evict every other cached model
#                     at each model boundary (default 0 = keep everything; all
#                     five models ≈ 87.5 GB fit the 150 GB pod disk)
#   OFFLOAD=1         add --offload to every run (small GPUs; flux-schnell on 24 GB)
#   REPROJECT        1 = optical-flow reprojection (default), 0 = naive blend
#   EXTRA_ARGS        extra flags, word-split, appended to every run
#   DRY_RUN=1         print commands without executing anything
#   SKIP_GPU_CHECK=1  bypass the CUDA preflight
#
# Log: output/sweep_<UTC timestamp>.log (config header + all run output).
# Stops at the first failing run (set -euo pipefail).

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Input files: env > input/inputs.env (user) > input_example/inputs.env.
source scripts/inputs.sh

CLIP="${CLIP:-input_example/video_cropped.mp4}"
MODELS="${MODELS:-sd-turbo sdxl-turbo flux-schnell flux2-klein-9b}"
BLENDS="${BLENDS:-0.1 0.3 0.5}"
BASELINE="${BASELINE:-0.3}"
MAX_FRAMES="${MAX_FRAMES-300}"
TAIL_FRAMES="${TAIL_FRAMES:-60}"
SAVE_EVERY="${SAVE_EVERY:-10}"
STRENGTH="${STRENGTH:-0.55}"
STRENGTH_MODELS="${STRENGTH_MODELS:-sd-turbo sdxl-turbo flux-schnell}"
DESC="${DESC:-underwater footage of an octopus in a coral habitat}"
OFFLOAD="${OFFLOAD:-0}"
REPROJECT="${REPROJECT:-1}"
EVICT_CACHE="${EVICT_CACHE:-0}"
DRY_RUN="${DRY_RUN:-0}"

case "$REPROJECT" in
    0|1) ;;
    *) echo "sweep: REPROJECT must be 0 or 1 (got '$REPROJECT')" >&2; exit 1 ;;
esac

if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$CLIP" ]] || { echo "sweep: clip not found: $CLIP" >&2; exit 1; }
    command -v uv >/dev/null || { echo "sweep: 'uv' not on PATH" >&2; exit 1; }

    # Gated models need a token before any model download starts.
    case " $MODELS " in
        *" flux2-klein-9b "*)
            if [[ -z "${HF_TOKEN:-}" ]]; then
                echo "sweep: HF_TOKEN is not set (required for flux2-klein-9b)." >&2
                echo "       Accept the license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B" >&2
                echo "       then: export HF_TOKEN=hf_... and re-run." >&2
                exit 1
            fi
            ;;
    esac

    # Preflight: the tools build a CUDA generator, so fail now rather than
    # after the first multi-GB model download.
    if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
        if ! uv run python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
            echo "sweep: torch reports no CUDA device -- run this on a GPU pod" >&2
            echo "       (SKIP_GPU_CHECK=1 to bypass, DRY_RUN=1 to preview)." >&2
            exit 1
        fi
    fi
fi

mkdir -p output
LOG="output/sweep_$(date -u +%Y%m%d-%H%M%S).log"

# Flags shared by every run. Arrays are expanded with the ${arr[@]+...} form so
# an empty array is safe under `set -u` on bash 3.2 (macOS).
common=(--input "$CLIP" --save-every "$SAVE_EVERY")
if [[ -n "$MAX_FRAMES" ]]; then common+=(--max-frames "$MAX_FRAMES"); fi
if [[ "$OFFLOAD" == "1" ]]; then common+=(--offload); fi
# Pin the reprojection mode explicitly: it is the tool default, but making it
# visible keeps the -norepro tag and the behavior deliberate.
if [[ "$REPROJECT" == "0" ]]; then
    common+=(--no-reproject)
else
    common+=(--reproject)
fi
# Freeze-tail flags shared by the prompt/strength runs (empty when TAIL_FRAMES=0).
tail_freeze=()
if [[ "$TAIL_FRAMES" -gt 0 ]]; then
    tail_freeze=(--tail-frames "$TAIL_FRAMES" --tail-modes freeze)
fi

log() { echo "$@" | tee -a "$LOG"; }

run() {
    log ""
    log "== $(date -u +%Y-%m-%dT%H:%M:%SZ) dltb-continuous $*"
    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY RUN"
        return 0
    fi
    uv run dltb-continuous "$@" 2>&1 | tee -a "$LOG"
}

log "sweep: clip=$CLIP"
log "sweep: models='$MODELS' blends='$BLENDS' baseline=$BASELINE"
log "sweep: max_frames=${MAX_FRAMES:-<all>} tail_frames=$TAIL_FRAMES save_every=$SAVE_EVERY reproject=$REPROJECT evict_cache=$EVICT_CACHE"
log "sweep: log=$LOG"

for MODEL in $MODELS; do
    log ""
    log "################ $MODEL ################"

    # Optional keep-one cache eviction (EVICT_CACHE=1). Runs at the model
    # boundary only -- never between the runs of one model: sweep.sh starts a
    # fresh `uv run dltb-continuous` per run, so evicting there would
    # re-download the same multi-GB weights once per run.
    if [[ "$EVICT_CACHE" == "1" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            DRY_RUN=1 scripts/hf-cache.sh keep "$MODEL" 2>&1 | tee -a "$LOG"
        elif ! scripts/hf-cache.sh keep "$MODEL" 2>&1 | tee -a "$LOG"; then
            log "sweep: WARNING -- cache eviction failed for $MODEL; continuing (disk may fill)"
        fi
    else
        log "sweep: cache eviction off (EVICT_CACHE=0) -- swept models stay cached"
    fi

    out_base="output/$MODEL"

    # 1. Boil test: every frame processed independently (no carried state).
    #    No tail -- tail phases use the stateful blend semantics.
    run --model "$MODEL" ${common[@]+"${common[@]}"} --mode anchored \
        --output-dir "$out_base" ${EXTRA_ARGS:+$EXTRA_ARGS}

    # 2. Blend sweep. The baseline blend carries all three tails; other blends
    #    only the freeze tail (static-menu scenario).
    for BLEND in $BLENDS; do
        tails=()
        if [[ "$TAIL_FRAMES" -gt 0 ]]; then
            if [[ "$BLEND" == "$BASELINE" ]]; then
                # Failure tails must share the baseline end state -> one run.
                # shellcheck disable=SC2054  # comma-separated single argument
                tails=(--tail-frames "$TAIL_FRAMES" --tail-modes freeze,free,black)
            else
                tails=(--tail-frames "$TAIL_FRAMES" --tail-modes freeze)
            fi
        fi
        run --model "$MODEL" ${common[@]+"${common[@]}"} \
            --mode stateful --anchor-blend "$BLEND" \
            ${tails[@]+"${tails[@]}"} \
            --output-dir "$out_base" ${EXTRA_ARGS:+$EXTRA_ARGS}
    done

    # 3. Semantic anchor: does a faithful prompt stabilise stateful mode?
    run --model "$MODEL" ${common[@]+"${common[@]}"} \
        --mode stateful --anchor-blend "$BASELINE" \
        ${tail_freeze[@]+"${tail_freeze[@]}"} --prompt "$DESC" \
        --output-dir "$out_base/prompt" ${EXTRA_ARGS:+$EXTRA_ARGS}

    # 4. Strength probe (classic partial-noise img2img models only).
    case " $STRENGTH_MODELS " in
        *" $MODEL "*)
            run --model "$MODEL" ${common[@]+"${common[@]}"} \
                --mode stateful --anchor-blend "$BASELINE" \
                ${tail_freeze[@]+"${tail_freeze[@]}"} --strength "$STRENGTH" \
                --output-dir "$out_base/strength$STRENGTH" ${EXTRA_ARGS:+$EXTRA_ARGS}
            ;;
    esac
done

log ""
log "sweep: done -- log: $LOG"
