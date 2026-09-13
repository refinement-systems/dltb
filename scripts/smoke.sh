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
# smoke.sh -- post-deploy smoke test: one tiny run of every tool.
#
# Verifies the console scripts and the cache helper against a real GPU using
# the cheapest model, inputs from input/inputs.env (falling back to the
# tracked input_example/ files), and minimal budgets:
#
#   0. scripts/hf-cache.sh status   (dltb.models import + cache probe)
#   1. dltb-oneshot                 1 pass on the configured image
#   2. dltb-iterate                 3 free-running passes + timelapse.mp4
#   3. dltb-continuous stateful     3 reprojected source frames + 2 frames
#                                   per tail (freeze, free, black)
#   4. dltb-continuous anchored     2 frames boil test
#   5. dltb-klein (optional)        SMOKE_KLEIN=1: both conditionings
#                                   (blend + dual-ref), 2 source frames +
#                                   1 freeze-tail frame -- the cheapest
#                                   end-to-end check of the klein loop,
#                                   including the dual-ref reference-list
#                                   code path
#
# Every step writes under output/smoke/ (wiped at start); the expected
# artifacts are checked for existence and non-emptiness afterwards. Any
# failing step aborts via set -e with its output above it.
#
# Usage:
#   scripts/smoke.sh
#   MODEL=sdxl-turbo scripts/smoke.sh     # smoke another model
#   SKIP_GPU_CHECK=1 scripts/smoke.sh     # bypass the CUDA preflight
#
# MODEL defaults to sd-turbo (2.4 GB). With the default EVICT_CACHE=0 every
# swept model stays cached, so check step 0's output and set MODEL=<a cached
# key> to skip any re-download (under EVICT_CACHE=1 the disk holds only
# whichever model the last sweep ended on).
#
# Environment:
#   MODEL            model key to smoke          (default sd-turbo)
#   IMG / CLIP       input files                 (default: input/inputs.env if
#                                                 present, else the tracked
#                                                 input_example/ files; see
#                                                 scripts/inputs.sh)
#   SMOKE_KLEIN=1    also smoke dltb-klein, both conditionings (off by default:
#                    flux2-klein-4b is a ~15 GB download)
#   KLEIN_MODEL      klein model for that leg    (default flux2-klein-4b)
#   SKIP_GPU_CHECK=1 bypass the CUDA preflight
#
# Log: output/smoke_<UTC timestamp>.log

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Input files: env > input/inputs.env (user) > input_example/inputs.env.
source scripts/inputs.sh

MODEL="${MODEL:-sd-turbo}"
IMG="${IMG:-input_example/test_512.png}"
CLIP="${CLIP:-input_example/video_cropped.mp4}"

# Keep these in sync with the expected-dir strings below (they mirror the
# tools' run-tag encoding: stateful-a<blend>_tails<modes><frames>).
ITERATIONS=3
SRC_FRAMES=3
TAIL_FRAMES=2
BLEND=0.3
STEPS=1          # 1 step x strength 1.0 = exactly 1 denoise step per pass
STRENGTH=1.0

img_stem="$(basename "$IMG")"; img_stem="${img_stem%.*}"
clip_stem="$(basename "$CLIP")"; clip_stem="${clip_stem%.*}"

for f in "$IMG" "$CLIP"; do
    [[ -f "$f" ]] || { echo "smoke: input not found: $f" >&2; exit 1; }
done

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
    if ! uv run python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
        echo "smoke: torch reports no CUDA device -- run this on a GPU pod" >&2
        echo "       (SKIP_GPU_CHECK=1 to bypass)" >&2
        exit 1
    fi
fi

mkdir -p output
LOG="output/smoke_$(date -u +%Y%m%d-%H%M%S).log"
OUT="output/smoke"
rm -rf "$OUT"
mkdir -p "$OUT"

log() { echo "$@" | tee -a "$LOG"; }

run() {
    log ""
    log "== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
    "$@" 2>&1 | tee -a "$LOG"
}

check() {
    local f="$1"
    if [[ -s "$f" ]]; then
        log "ok: $f ($(du -h "$f" | cut -f1))"
    else
        log "SMOKE FAIL: missing or empty $f"
        exit 1
    fi
}

log "smoke: model=$MODEL img=$IMG clip=$CLIP"
log "smoke: log=$LOG"

# 0. Cache helper: exercises the dltb.models import and shows what is
#    cached (pick MODEL=<cached key> to avoid a re-download).
run scripts/hf-cache.sh status

# 1. dltb-oneshot
run uv run dltb-oneshot --model "$MODEL" --input "$IMG" \
    --num-inference-steps "$STEPS" --strength "$STRENGTH" \
    --output-dir "$OUT"
d="$OUT/${img_stem}_oneshot"
check "$d/frame_0000_original.png"
check "$d/frames/frame_0001.png"

# 2. dltb-iterate
run uv run dltb-iterate --model "$MODEL" --input "$IMG" \
    --iterations "$ITERATIONS" --save-every 1 --video-fps 12 \
    --num-inference-steps "$STEPS" --strength "$STRENGTH" \
    --output-dir "$OUT"
d="$OUT/${img_stem}_free-running"
check "$d/frames/frame_0001.png"
check "$d/frames/frame_${ITERATIONS}.png"
check "$d/timelapse.mp4"

# 3. dltb-continuous, stateful + all three tails (reprojection on)
run uv run dltb-continuous --model "$MODEL" --input "$CLIP" \
    --mode stateful --anchor-blend "$BLEND" --reproject \
    --max-frames "$SRC_FRAMES" \
    --tail-frames "$TAIL_FRAMES" --tail-modes freeze,free,black \
    --save-every 1 \
    --num-inference-steps "$STEPS" --strength "$STRENGTH" \
    --output-dir "$OUT"
d="$OUT/${clip_stem}_stateful-a${BLEND}_tailsfreeze-free-black${TAIL_FRAMES}"
check "$d/frame_0000_source.png"
check "$d/end_state.png"
check "$d/processed_stateful.mp4"
check "$d/tail_freeze.mp4"
check "$d/tail_free.mp4"
check "$d/tail_black.mp4"

# 4. dltb-continuous, anchored boil test
run uv run dltb-continuous --model "$MODEL" --input "$CLIP" \
    --mode anchored --max-frames 2 --save-every 1 \
    --num-inference-steps "$STEPS" --strength "$STRENGTH" \
    --output-dir "$OUT"
check "$OUT/${clip_stem}_anchored/processed_anchored.mp4"

# 5. dltb-klein, optional (SMOKE_KLEIN=1): both conditionings, klein-default
#    settings (no strength; 4 steps), 2 source frames + 1 freeze-tail frame.
#    klein run tags: stateful-a<blend> vs dualref, + _tailsfreeze1.
if [[ "${SMOKE_KLEIN:-0}" == "1" ]]; then
    KLEIN_MODEL="${KLEIN_MODEL:-flux2-klein-4b}"
    log ""
    log "smoke: klein leg (model=$KLEIN_MODEL, both conditionings)"
    if [[ "$KLEIN_MODEL" == "flux2-klein-9b" && -z "${HF_TOKEN:-}" ]]; then
        log "SMOKE FAIL: HF_TOKEN is not set (required for flux2-klein-9b)"
        exit 1
    fi

    run uv run dltb-klein --model "$KLEIN_MODEL" --input "$CLIP" \
        --mode stateful --conditioning blend --anchor-blend 0.1 --reproject \
        --max-frames 2 --tail-frames 1 --tail-modes freeze \
        --save-every 1 --output-dir "$OUT/klein-blend"
    d="$OUT/klein-blend/${clip_stem}_stateful-a0.1_tailsfreeze1"
    check "$d/processed_stateful.mp4"
    check "$d/tail_freeze.mp4"

    run uv run dltb-klein --model "$KLEIN_MODEL" --input "$CLIP" \
        --mode stateful --conditioning dual-ref --ref-order state-first \
        --reproject \
        --max-frames 2 --tail-frames 1 --tail-modes freeze \
        --save-every 1 --output-dir "$OUT/klein-dualref"
    d="$OUT/klein-dualref/${clip_stem}_dualref_tailsfreeze1"
    check "$d/processed_stateful.mp4"
    check "$d/tail_freeze.mp4"
fi

log ""
log "smoke: PASS -- all requested tools produced their artifacts under $OUT"
