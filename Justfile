default:
    @just --list

# Stage the pod-ready tree for `runpodctl send`
bundle:
    scripts/bundle.sh

# Remove the staged bundle
clean:
    rm -rf bundle/*

# Inspect the HuggingFace model cache (per-repo sizes, filesystem free)
hf-status:
    scripts/hf-cache.sh status

# Evict every cached model except <model> (same key as --model)
hf-keep model:
    scripts/hf-cache.sh keep {{model}}

# Drop every cached model (and the xet chunk cache)
hf-clean:
    scripts/hf-cache.sh clean
