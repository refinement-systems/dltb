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
# Verifies the three console scripts and the cache helper against a real GPU
# using the cheapest model, bundle-shipped inputs, and minimal budgets:
#
#   0. scripts/hf-cache.sh status   (dltb.models import + cache probe)
#   1. dltb-oneshot                 1 pass on input/test_512.png
#   2. dltb-iterate                 3 free-running passes + timelapse.mp4
#   3. dltb-continuous stateful     3 reprojected source frames + 2 frames
#                                   per tail (freeze, free, black)
#   4. dltb-continuous anchored     2 frames boil test
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
# MODEL defaults to sd-turbo (2.4 GB). hf-cache keep-one means the disk
# holds whichever model the last sweep ended on -- check step 0's output
# and set MODEL=<that key> to skip the re-download.
#
# Environment:
#   MODEL            model key to smoke          (default sd-turbo)
#   IMG / CLIP       input files                 (default input/test_512.png,
#                                                 input/video_cropped.mp4)
#   SKIP_GPU_CHECK=1 bypass the CUDA preflight
#
# Log: output/smoke_<UTC timestamp>.log

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-sd-turbo}"
IMG="${IMG:-input/test_512.png}"
CLIP="${CLIP:-input/video_cropped.mp4}"

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

log ""
log "smoke: PASS -- all three tools produced their artifacts under $OUT"
