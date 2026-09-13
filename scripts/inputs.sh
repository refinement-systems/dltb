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
# inputs.sh -- shared IMG/CLIP selection for the run scripts.
#
# Source this from the repo root (after the cd in sweep.sh / sweep-klein.sh /
# smoke.sh):
#     source scripts/inputs.sh
#
# Sources input/inputs.env (user config: KEY=VALUE lines, IMG= and CLIP=) when
# present, otherwise the tracked input_example/inputs.env. input/ is
# gitignored, but scripts/bundle.sh packs it into pod bundles, so user inputs
# travel with the code.
#
# Precedence for a variable: environment > input/inputs.env > example
# defaults -- provided the conf file assigns with ${VAR:-value} guards, as
# input_example/inputs.env does. Plain assignments in a user file work too;
# they just make the file win over the environment for that variable.
#
# Callers still apply their own ${VAR:-...} fallbacks after sourcing, so a
# missing conf file only costs the built-in defaults.

# Guard: every driver invokes the tools via `uv run`, which would silently
# create and sync a fresh project venv (~6 GB on a pod) if none exists yet.
# Fail fast instead; DRY_RUN=1 previews bypass this.
if [[ "${DRY_RUN:-0}" != "1" && ! -d .venv ]]; then
    echo "inputs: no .venv in ${PWD} -- run scripts/setup-pod.sh first" >&2
    echo "        (locally: uv sync)" >&2
    exit 1
fi

if [[ -f input/inputs.env ]]; then
    source input/inputs.env
elif [[ -f input_example/inputs.env ]]; then
    source input_example/inputs.env
fi
