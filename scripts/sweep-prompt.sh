#!/usr/bin/env bash

# Permission to use, copy, modify, and/or distribute this software for
# any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#
# sweep-prompt.sh -- free-running prompt sweep on ONE image (dltb-iterate).
#
# For each prompt (optionally crossed with strength values) run a SHORT
# free-running self-iteration of one model on a single input image:
#
#     P_n = f(P_{n-1}, prompt [, strength])
#
# Every pass is saved (frames/frame_NNNN.png, --save-every 1) and assembled
# into a per-run timelapse.mp4, so a whole leg is a few dozen images -- cheap
# enough to walk a ladder of prompts in one sitting. The prompt table defaults
# match the tracked example input (a bronze lion sculpture; see
# input_example/SOURCES.txt): a neutral preservation baseline, a rising
# enhancement ladder, and one thematic compounding instruction (patina).
#
# STRENGTH axis: full prompts x strengths cross product, but only for the
# classic img2img models (STRENGTH_MODELS) -- klein is a reference editor
# with no --strength, and there the prompt is already the per-pass knob
# (see scripts/sweep-klein.sh for the klein regime). Setting STRENGTHS with
# a klein model logs a note and runs the prompt ladder alone.
#
# Each run writes its own --output-dir subtree (dltb-iterate's directory tag
# is a constant <stem>_free-running, so without subtrees the legs would
# silently overwrite each other):
#
#     output/<model>/prompt-<slug>/                        no strength axis
#     output/<model>/prompt-<slug>/strength<value>/...     with the axis
#
# Every subtree is complete the moment its run finishes, so partial results
# can be copied off the pod while the sweep keeps going.
#
# Usage (from any directory inside the repo):
#   scripts/sweep-prompt.sh
#   DRY_RUN=1 scripts/sweep-prompt.sh
#   MODEL=sdxl-turbo ITERATIONS=40 scripts/sweep-prompt.sh
#   MODEL=sd-turbo STRENGTHS="0.4 0.7" scripts/sweep-prompt.sh
#
# Environment overrides:
#   MODEL           model key                (default sd-turbo, cheapest)
#   IMG             input image              (default: input/inputs.env if
#                                             present, else the tracked
#                                             input_example/test_512.png;
#                                             see scripts/inputs.sh)
#   ITERATIONS      passes per run            (default 20)
#   SAVE_EVERY      save every Nth frame      (default 1 -- every frame)
#   VIDEO_FPS       timelapse fps             (default 12)
#   STRENGTHS       strength values, space-separated (default: none -> single
#                   run per prompt at the tool's default strength)
#   STRENGTH_MODELS models the axis applies to
#                   (default sd-turbo sdxl-turbo flux-schnell)
#   EXTRA_ARGS      extra flags, word-split, appended to every run
#   DRY_RUN=1       print commands without executing anything
#   SKIP_GPU_CHECK=1  bypass the CUDA preflight
#
# Log: output/sweep_prompt_<UTC timestamp>.log
# Stops at the first failing run (set -euo pipefail).

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Input files: env > input/inputs.env (user) > input_example/inputs.env.
source scripts/inputs.sh

MODEL="${MODEL:-sd-turbo}"
IMG="${IMG:-input_example/test_512.png}"
ITERATIONS="${ITERATIONS:-20}"
SAVE_EVERY="${SAVE_EVERY:-1}"
VIDEO_FPS="${VIDEO_FPS:-12}"
STRENGTHS="${STRENGTHS:-}"
STRENGTH_MODELS="${STRENGTH_MODELS:-sd-turbo sdxl-turbo flux-schnell}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

# -------------------------------------------------------------- prompts ----
# slug|prompt pairs; the slug becomes the output subdirectory. Empty prompt =
# preservation baseline (no --prompt flag passed). Defaults describe the
# tracked example image (bronze lion sculpture) -- point IMG at your own and
# adjust PATINA/PROBE lines accordingly.
PROMPT_TABLE=(
    "neutral|"
    "enhance-slight|slightly enhance the fine details"
    "enhance-photo|enhance details and lighting, make it look like a professional photograph"
    "enhance-dramatic|dramatically enhance every texture and surface detail"
    "patina|add more blue-green patina and weathering to the bronze surface"
)

# -------------------------------------------------------------- preflight ----
if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$IMG" ]] || { echo "sweep-prompt: image not found: $IMG" >&2; exit 1; }
    command -v uv >/dev/null || { echo "sweep-prompt: 'uv' not on PATH" >&2; exit 1; }

    if [[ "$MODEL" == "flux2-klein-9b" && -z "${HF_TOKEN:-}" ]]; then
        echo "sweep-prompt: HF_TOKEN is not set (required for flux2-klein-9b)." >&2
        echo "              Accept the license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B" >&2
        echo "              then: export HF_TOKEN=hf_... and re-run." >&2
        exit 1
    fi

    if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
        if ! uv run python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
            echo "sweep-prompt: torch reports no CUDA device -- run this on a GPU pod" >&2
            echo "              (SKIP_GPU_CHECK=1 to bypass, DRY_RUN=1 to preview)." >&2
            exit 1
        fi
    fi
fi

# Strength axis applies to classic img2img models only.
STRENGTH_AXIS=0
case " $STRENGTH_MODELS " in
    *" $MODEL "*) STRENGTH_AXIS=1 ;;
esac
if [[ "$STRENGTH_AXIS" != "1" ]]; then
    STRENGTHS=""   # klein et al.: no --strength semantics; prompts only
fi

mkdir -p output
LOG="output/sweep_prompt_$(date -u +%Y%m%d-%H%M%S).log"

log() { echo "$@" | tee -a "$LOG"; }

run() {
    log ""
    log "== $(date -u +%Y-%m-%dT%H:%M:%SZ) dltb-iterate $*"
    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY RUN"
        return 0
    fi
    uv run dltb-iterate "$@" 2>&1 | tee -a "$LOG"
}

log "sweep-prompt: model=$MODEL img=$IMG iterations=$ITERATIONS save_every=$SAVE_EVERY"
log "sweep-prompt: strength_axis=$STRENGTH_AXIS strengths='${STRENGTHS:-<none>}'"
log "sweep-prompt: log=$LOG"
if [[ "$STRENGTH_AXIS" != "1" ]]; then
    log "sweep-prompt: NOTE: $MODEL has no --strength semantics -- prompt ladder only"
fi

for entry in "${PROMPT_TABLE[@]}"; do
    slug="${entry%%|*}"
    prompt="${entry#*|}"

    if [[ "$STRENGTH_AXIS" == "1" && -n "$STRENGTHS" ]]; then
        strengths="$STRENGTHS"
    else
        strengths=""          # single run at the tool's default strength
    fi

    for S in ${strengths:-default}; do
        [[ "$S" == "default" ]] && S=""
        out="output/$MODEL/prompt-$slug"
        args=(--model "$MODEL" --input "$IMG" --iterations "$ITERATIONS"
              --save-every "$SAVE_EVERY" --video-fps "$VIDEO_FPS")
        if [[ -n "$S" ]]; then
            out="$out/strength$S"
            args+=(--strength "$S")
        fi
        args+=(--output-dir "$out")
        if [[ -n "$prompt" ]]; then args+=(--prompt "$prompt"); fi

        log ""
        log "########## prompt '$slug'${S:+ @ strength $S}: ${prompt:-<empty>} ##########"
        run "${args[@]}" ${EXTRA_ARGS:+$EXTRA_ARGS}
        log "sweep-prompt: leg done: $out (safe to copy off mid-sweep)"
    done
done

log ""
log "sweep-prompt: done -- log: $LOG"
