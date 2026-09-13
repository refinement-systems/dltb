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
# CONDITIONING (stateful mode):
#   blend    (default) carried state and fresh frame pixel-blended into ONE
#            reference; --anchor-blend applies (BLEND env).
#   dual-ref carried state and fresh frame as TWO separate clean references --
#            no ghosted blend for the editor to parse; state/anchor weighting
#            is done by the model. BLEND is ignored; REF_ORDER picks [P,N]
#            (state-first, default) or [N,P]. The prompt table switches to
#            role-naming prompts whose image indices are DERIVED from
#            REF_ORDER ("image N is the current frame; keep the appearance of
#            image M, ..."), so the instruction keeps its semantic roles under
#            either order -- no manual role swapping.
#
# REPROJECT (stateful mode): 1 = warp the carried reference by source-frame
#   optical flow before pairing (emulates engine motion vectors); 0 = pass
#   both references clean. Under dual-ref the default is AB: every leg runs
#   TWICE, once per setting -- the sweep itself is the reproject A/B (does
#   warp alignment help, or do warp artifacts get amplified by an editor
#   trained on clean references?). The run tags distinguish the variants
#   (..._dualref vs ..._dualref-norepro), so nothing collides; both variants
#   of a leg complete before the next leg starts, so partial results can be
#   copied off the pod mid-sweep. Pin one variant with REPROJECT=1 or 0.
#
# What this sweep runs (all --mode stateful, one source pass per prompt):
#
#   1. Prompt ladder       : neutral -> slight -> photo -> dramatic enhancement,
#                            i.e. rising per-pass edit intensity. The freeze tail
#                            shows whether the edit keeps compounding on static
#                            input (the generative-ratchet / static-menu case).
#   2. Semantic attractor  : a THEMATIC instruction (coral overgrowth -- the
#                            tracked example clip is an octopus in a coral
#                            habitat). If klein works as trained, the loop
#                            converges toward "maximally overgrown" instead
#                            of melting - directed attractor vs. undirected
#                            collapse.
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
# the subtree, prompt runs would silently overwrite each other. (Dual-ref runs
# are tagged ..._dualref[-norepro] instead of ..._stateful-a<blend>.)
#
# Usage (from any directory inside the repo):
#   scripts/sweep-klein.sh
#   DRY_RUN=1 scripts/sweep-klein.sh
#   MODEL=flux2-klein-4b BLEND=0.2 scripts/sweep-klein.sh
#   CONDITIONING=dual-ref scripts/sweep-klein.sh
#   CONDITIONING=dual-ref REF_ORDER=frame-first scripts/sweep-klein.sh
#   CONDITIONING=dual-ref REPROJECT=1 scripts/sweep-klein.sh   # pin one A/B variant
#
# Environment overrides:
#   MODEL         klein model key          (default flux2-klein-9b; 4b is ungated)
#   CLIP          source video             (default: input/inputs.env if
#                                             present, else the tracked
#                                             input_example/video_cropped.mp4;
#                                             see scripts/inputs.sh)
#   CONDITIONING  blend | dual-ref         (default blend)
#   REF_ORDER     state-first | frame-first (default state-first; dual-ref only)
#   BLEND         anchor-blend for all runs (default 0.1 - klein's active range
#                 is far below the img2img models; try 0.03 0.05 0.2 manually;
#                 IGNORED under CONDITIONING=dual-ref)
#   MAX_FRAMES    source frames per run    (default 300; empty = whole clip)
#   TAIL_FRAMES   frames per tail          (default 60; 0 = no tails)
#   TAIL_MODES    tail scenario list       (default freeze; "freeze,free" etc.
#                 NOTE: under dual-ref, free = single-reference regeneration
#                 from state alone)
#   REPROJECT     1 | 0 | ab               (default: ab under dual-ref = both
#                 variants per leg, the A/B; 1 under blend. See the REPROJECT
#                 paragraph above)
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

# Input files: env > input/inputs.env (user) > input_example/inputs.env.
source scripts/inputs.sh

MODEL="${MODEL:-flux2-klein-9b}"
CLIP="${CLIP:-input_example/video_cropped.mp4}"
CONDITIONING="${CONDITIONING:-blend}"
REF_ORDER="${REF_ORDER:-state-first}"
BLEND="${BLEND:-0.1}"
MAX_FRAMES="${MAX_FRAMES-300}"
TAIL_FRAMES="${TAIL_FRAMES:-60}"
TAIL_MODES="${TAIL_MODES:-freeze}"
SAVE_EVERY="${SAVE_EVERY:-10}"
STEPS="${STEPS-2 8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

case "$CONDITIONING" in
    blend|dual-ref) ;;
    *) echo "sweep-klein: CONDITIONING must be blend or dual-ref (got '$CONDITIONING')" >&2; exit 1 ;;
esac
case "$REF_ORDER" in
    state-first|frame-first) ;;
    *) echo "sweep-klein: REF_ORDER must be state-first or frame-first (got '$REF_ORDER')" >&2; exit 1 ;;
esac

# Reprojection: default AB under dual-ref (the sweep doubles as the A/B),
# pinned on under blend (parity with scripts/sweep.sh).
REPROJECT="${REPROJECT:-}"
if [[ -z "$REPROJECT" ]]; then
    if [[ "$CONDITIONING" == "dual-ref" ]]; then REPROJECT=ab; else REPROJECT=1; fi
fi
case "$REPROJECT" in
    1|0|ab) ;;
    *) echo "sweep-klein: REPROJECT must be 1, 0, or ab (got '$REPROJECT')" >&2; exit 1 ;;
esac
if [[ "$REPROJECT" == "ab" ]]; then REPROJECT_LIST="1 0"; else REPROJECT_LIST="$REPROJECT"; fi

# ---------------------------------------------------------------- prompts ----
# slug|prompt pairs. The slug becomes the output subdirectory; keep slugs short,
# lowercase, hyphenated. Empty prompt = preservation baseline (no flag passed).
# Prompts match the tracked example clip (octopus in a coral habitat; see
# input_example/SOURCES.txt) -- point CLIP at your own and adjust OVERGROWTH.
#
# Role-naming for dual-ref: image indices are resolved from REF_ORDER, so
# "image $FRAME_IMG is the current frame; keep the appearance of image
# $STATE_IMG" keeps its semantic roles whether the list is [P, N] or [N, P].
if [[ "$REF_ORDER" == "state-first" ]]; then
    STATE_IMG=1
    FRAME_IMG=2
else
    STATE_IMG=2
    FRAME_IMG=1
fi

if [[ "$CONDITIONING" == "dual-ref" ]]; then
    PROMPT_TABLE=(
        "neutral|"
        "enhance-slight|image ${FRAME_IMG} is the current frame; keep the appearance of image ${STATE_IMG}, slightly enhancing fine details"
        "enhance-photo|image ${FRAME_IMG} is the current frame; keep the appearance of image ${STATE_IMG}, enhancing details and lighting to look photorealistic"
        "enhance-dramatic|image ${FRAME_IMG} is the current frame; keep the appearance of image ${STATE_IMG}, dramatically enhancing every texture and surface detail"
        "overgrowth|image ${FRAME_IMG} is the current frame; keep the appearance of image ${STATE_IMG}, adding more coral and marine growth over every surface"
    )
else
    PROMPT_TABLE=(
        "neutral|"
        "enhance-slight|slightly enhance the fine details"
        "enhance-photo|enhance details and lighting, make it photorealistic"
        "enhance-dramatic|dramatically enhance every texture and surface detail"
        "overgrowth|add more coral and marine growth over every surface"
    )
fi

# Steps probes reuse the mild-enhancement prompt.
PROBE_SLUG="enhance-slight"
if [[ "$CONDITIONING" == "dual-ref" ]]; then
    PROBE_PROMPT="image ${FRAME_IMG} is the current frame; keep the appearance of image ${STATE_IMG}, slightly enhancing fine details"
else
    PROBE_PROMPT="slightly enhance the fine details"
fi

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
        --mode stateful --conditioning "$CONDITIONING")
if [[ "$CONDITIONING" == "dual-ref" ]]; then
    common+=(--ref-order "$REF_ORDER")   # --anchor-blend is ignored under dual-ref
else
    common+=(--anchor-blend "$BLEND")
fi
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

log "sweep-klein: model=$MODEL clip=$CLIP conditioning=$CONDITIONING ref_order=$REF_ORDER blend=$BLEND reproject=$REPROJECT"
if [[ "$CONDITIONING" == "dual-ref" ]]; then
    log "sweep-klein: dual-ref roles: image $STATE_IMG = carried state, image $FRAME_IMG = current frame"
fi
log "sweep-klein: max_frames=${MAX_FRAMES:-<all>} tail=${TAIL_FRAMES}x${TAIL_MODES} steps='${STEPS:-<none>}'"
log "sweep-klein: log=$LOG"

# ------------------------------------------------------- 1+2. prompt ladder ----
# Inner loop over the reproject settings: both A/B variants of a leg finish
# before the next leg starts (comparable pairs land on disk early).
for entry in "${PROMPT_TABLE[@]}"; do
    slug="${entry%%|*}"
    prompt="${entry#*|}"

    for R in $REPROJECT_LIST; do
        args=(${common[@]+"${common[@]}"} $([[ "$R" == "1" ]] && echo --reproject || echo --no-reproject)
              --output-dir "output/$MODEL/prompt-$slug")
        if [[ -n "$prompt" ]]; then args+=(--prompt "$prompt"); fi

        log ""
        log "########## prompt '$slug' [reproject=$R]: ${prompt:-<empty>} ##########"
        run "${args[@]}" ${EXTRA_ARGS:+$EXTRA_ARGS}
        log "sweep-klein: leg done: output/$MODEL/prompt-$slug (safe to copy off mid-sweep)"
    done
done

# --------------------------------------------------------- 3. steps probe ----
if [[ -n "${STEPS:-}" ]]; then
    for N in $STEPS; do
        for R in $REPROJECT_LIST; do
            log ""
            log "########## steps probe: '$PROBE_SLUG' @ num-inference-steps=$N [reproject=$R] ##########"
            run ${common[@]+"${common[@]}"} \
                $([[ "$R" == "1" ]] && echo --reproject || echo --no-reproject) \
                --prompt "$PROBE_PROMPT" --num-inference-steps "$N" \
                --output-dir "output/$MODEL/steps$N" ${EXTRA_ARGS:+$EXTRA_ARGS}
            log "sweep-klein: leg done: output/$MODEL/steps$N (safe to copy off mid-sweep)"
        done
    done
fi

log ""
log "sweep-klein: done -- log: $LOG"
