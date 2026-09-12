#!/usr/bin/env bash
#
# sweep-klein.sh -- prompt-ladder sweep for FLUX.2 klein (editing-model regime).
# Drives dltb-klein (the klein-restricted dltb-continuous).
#
# Klein is a reference-image EDITOR: it regenerates from pure noise attending to
# pristine reference tokens, ignores --strength, and is trained to preserve
# everything the prompt does not target. Per-pass change therefore scales with
# the blend alpha (~linear, no constant rewrite bias like the img2img models),
# and the PROMPT is the de-facto per-pass edit-strength knob.
#
# What this sweep runs (all --mode stateful, one source pass per prompt):
#
#   1. Prompt ladder       : neutral -> slight -> photo -> dramatic enhancement,
#                            i.e. rising per-pass edit intensity. The freeze tail
#                            shows whether the edit keeps compounding on static
#                            input (the generative-ratchet / static-menu case).
#   2. Semantic attractor  : a THEMATIC instruction (weathering). If klein works
#                            as trained, the loop converges toward "maximally
#                            weathered" instead of melting - directed attractor
#                            vs. undirected collapse.
#   3. Steps probes        : the mild prompt at --num-inference-steps 2 / 8
#                            (bracketing the klein card default of 4, which the
#                            ladder legs already run). Steps are the one direct
#                            per-pass edit-intensity knob that actually reaches
#                            klein -- real scheduler steps, no distillation
#                            short-circuit. (Supersedes the guidance probes,
#                            removed 2026-09-13: --guidance-scale > 1 is provably
#                            inert for step-wise distilled klein; see NOTES.md.)
#
# Each prompt gets its own --output-dir subtree (output/<model>/prompt-<slug>/),
# because dltb-klein's directory tag encodes only mode/blend/tails - without
# the subtree, prompt runs would silently overwrite each other.
#
# Usage (from any directory inside the repo):
#   scripts/sweep-klein.sh
#   DRY_RUN=1 scripts/sweep-klein.sh
#   MODEL=flux2-klein-4b BLEND=0.2 scripts/sweep-klein.sh
#
# Environment overrides:
#   MODEL         klein model key          (default flux2-klein-9b; 4b is ungated)
#   CLIP          source video             (default input/video_cropped.mp4)
#   BLEND         anchor-blend for all runs (default 0.1 - klein's active range
#                 is far below the img2img models; try 0.03 0.05 0.2 manually)
#   MAX_FRAMES    source frames per run    (default 300; empty = whole clip)
#   TAIL_FRAMES   frames per tail          (default 60; 0 = no tails)
#   TAIL_MODES    tail scenario list       (default freeze; "freeze,free" etc.)
#   SAVE_EVERY    save every Nth frame     (default 10)
#   STEPS         steps-probe values       (default "2 8", bracketing the default 4;
#                 empty = skip the probe)
#   EXTRA_ARGS    extra flags, word-split, appended to every run
#   DRY_RUN=1     print commands without executing anything
#   SKIP_GPU_CHECK=1  bypass the CUDA preflight
#
# Log: output/sweep_klein_<UTC timestamp>.log
# Stops at the first failing run (set -euo pipefail).
#
# NOTE: unlike scripts/sweep.sh this is single-model, so it does not run the
# hf-cache keep-one eviction -- if it follows a multi-model sweep, the disk
# briefly holds two models (still well within the 150 GB pod disk).

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-flux2-klein-9b}"
CLIP="${CLIP:-input/video_cropped.mp4}"
BLEND="${BLEND:-0.1}"
MAX_FRAMES="${MAX_FRAMES-300}"
TAIL_FRAMES="${TAIL_FRAMES:-60}"
TAIL_MODES="${TAIL_MODES:-freeze}"
SAVE_EVERY="${SAVE_EVERY:-10}"
STEPS="${STEPS-2 8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------- prompts ----
# slug|prompt pairs. The slug becomes the output subdirectory; keep slugs short,
# lowercase, hyphenated. Empty prompt = preservation baseline (no flag passed).
PROMPT_TABLE=(
    "neutral|"
    "enhance-slight|slightly enhance the fine details"
    "enhance-photo|enhance details and lighting, make it photorealistic"
    "enhance-dramatic|dramatically enhance every texture and surface detail"
    "weathering|add more weathering, moss and water stains to the stone"
)

# Steps probes reuse the mild-enhancement prompt.
PROBE_SLUG="enhance-slight"
PROBE_PROMPT="slightly enhance the fine details"

# -------------------------------------------------------------- preflight ----
if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$CLIP" ]] || { echo "sweep-klein: clip not found: $CLIP" >&2; exit 1; }
    command -v uv >/dev/null || { echo "sweep-klein: 'uv' not on PATH" >&2; exit 1; }

    if [[ "$MODEL" == "flux2-klein-9b" && -z "${HF_TOKEN:-}" ]]; then
        echo "sweep-klein: HF_TOKEN is not set (required for flux2-klein-9b)." >&2
        echo "             Accept the license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B" >&2
        echo "             then: export HF_TOKEN=hf_... and re-run." >&2
        exit 1
    fi

    if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
        if ! uv run python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
            echo "sweep-klein: torch reports no CUDA device -- run this on a GPU pod" >&2
            echo "             (SKIP_GPU_CHECK=1 to bypass, DRY_RUN=1 to preview)." >&2
            exit 1
        fi
    fi
fi

mkdir -p output
LOG="output/sweep_klein_$(date -u +%Y%m%d-%H%M%S).log"

# Flags shared by every run.
common=(--model "$MODEL" --input "$CLIP" --save-every "$SAVE_EVERY"
        --mode stateful --anchor-blend "$BLEND")
if [[ -n "$MAX_FRAMES" ]]; then common+=(--max-frames "$MAX_FRAMES"); fi
if [[ "$TAIL_FRAMES" -gt 0 ]]; then
    common+=(--tail-frames "$TAIL_FRAMES" --tail-modes "$TAIL_MODES")
fi

log() { echo "$@" | tee -a "$LOG"; }

run() {
    log ""
    log "== $(date -u +%Y-%m-%dT%H:%M:%SZ) dltb-klein $*"
    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY RUN"
        return 0
    fi
    uv run dltb-klein "$@" 2>&1 | tee -a "$LOG"
}

log "sweep-klein: model=$MODEL clip=$CLIP blend=$BLEND"
log "sweep-klein: max_frames=${MAX_FRAMES:-<all>} tail=${TAIL_FRAMES}x${TAIL_MODES} steps='${STEPS:-<none>}'"
log "sweep-klein: log=$LOG"

# ------------------------------------------------------- 1+2. prompt ladder ----
for entry in "${PROMPT_TABLE[@]}"; do
    slug="${entry%%|*}"
    prompt="${entry#*|}"

    args=(${common[@]+"${common[@]}"} --output-dir "output/$MODEL/prompt-$slug")
    if [[ -n "$prompt" ]]; then args+=(--prompt "$prompt"); fi

    log ""
    log "########## prompt '$slug': ${prompt:-<empty>} ##########"
    run "${args[@]}" ${EXTRA_ARGS:+$EXTRA_ARGS}
done

# --------------------------------------------------------- 3. steps probe ----
if [[ -n "${STEPS:-}" ]]; then
    for N in $STEPS; do
        log ""
        log "########## steps probe: '$PROBE_SLUG' @ num-inference-steps=$N ##########"
        run ${common[@]+"${common[@]}"} \
            --prompt "$PROBE_PROMPT" --num-inference-steps "$N" \
            --output-dir "output/$MODEL/steps$N" ${EXTRA_ARGS:+$EXTRA_ARGS}
    done
fi

log ""
log "sweep-klein: done -- log: $LOG"
