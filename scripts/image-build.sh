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

set -euo pipefail

# Build the imgiter pod image (see image/Dockerfile). Usage:
#   scripts/image-build.sh <registry>/<namespace>/<name>:<tag>
# The tag is chosen by the caller; push it yourself afterwards:
#   docker push <tag>

tag="${1:?usage: image-build.sh <registry>/<namespace>/<name>:<tag>}"

temp_dir="$(mktemp -d)"
trap 'rm -rf "${temp_dir}"' EXIT
stage="${temp_dir}/ctx"
mkdir -p "${stage}"

# Tracked files only, copied from the working tree so images match the repo
# (same policy as scripts/bundle.sh). --no-xattrs keeps macOS xattrs out of
# the build context; the Dockerfile itself is copied explicitly so an
# uncommitted edit still builds.
git ls-files -z | tar --no-xattrs --null -T - -cf - | tar -xf - -C "${stage}"
cp image/Dockerfile "${stage}/Dockerfile"

# Guard against building something broken.
for f in pyproject.toml uv.lock \
         src/dltb/models.py src/dltb/imaging.py src/dltb/output.py src/dltb/args.py \
         src/dltb/oneshot.py src/dltb/iterate.py src/dltb/continuous.py src/dltb/klein.py \
         Dockerfile; do
    [[ -f "${stage}/$f" ]] || { echo "image-build: missing $f" >&2; exit 1; }
done

# Untracked files are NOT baked in — surface them so nothing silently gets
# left behind.
untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "${untracked}" ]]; then
    echo "image-build: warning — untracked files are NOT included:"
    printf '%s\n' "${untracked}" | sed 's/^/  /'
fi

if ! git diff --quiet HEAD 2>/dev/null; then
    echo "image-build: note — building with uncommitted changes to tracked files."
fi

# linux/amd64 explicitly: builds correctly (under emulation) on Apple Silicon.
docker buildx build --platform linux/amd64 -t "${tag}" "${stage}"

echo
echo "Next: docker push ${tag}"
