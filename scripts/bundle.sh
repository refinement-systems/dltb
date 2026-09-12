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

# macOS tar converts the com.apple.provenance xattr (which the staging pipeline
# leaves on every extracted file) into AppleDouble ._* members in the final
# archive, even with --no-xattrs. COPYFILE_DISABLE=1 suppresses that; the
# variable is a no-op on Linux (see NOTES.md for the full investigation).
export COPYFILE_DISABLE=1

temp_dir="$(mktemp -d)"
trap 'rm -rf "${temp_dir}"' EXIT

stamp="$(date +%Y%m%d%H%M)"
top="imgiter-${stamp}"
stage="${temp_dir}/${top}"

out_dir="bundle"
bundle_file="${out_dir}/${top}.tar.gz"

# Tracked files only, copied from the working tree so pod runs match local.
# --no-xattrs keeps macOS xattrs (com.apple.provenance) out of the archive;
# GNU tar on the pod otherwise warns about LIBARCHIVE.xattr.* pax headers.
mkdir -p "${stage}"
git ls-files -z | tar --no-xattrs --null -T - -cf - | tar -xf - -C "${stage}"

# Guard against staging something broken.
for f in pyproject.toml uv.lock \
         src/dltb/models.py src/dltb/imaging.py src/dltb/output.py src/dltb/args.py \
         src/dltb/oneshot.py src/dltb/iterate.py src/dltb/continuous.py; do
    [[ -f "${stage}/$f" ]] || { echo "bundle: missing $f" >&2; exit 1; }
done

# Untracked files are NOT bundled — surface them so nothing silently gets left behind.
untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$untracked" ]]; then
    echo "bundle: warning — untracked files are NOT included:"
    printf '%s\n' "$untracked" | sed 's/^/  /'
fi

if ! git diff --quiet HEAD 2>/dev/null; then
    echo "bundle: note — bundling uncommitted changes to tracked files."
fi

# -C + the top-level dir name keep the archive rooted at ${top}/; without it
# the members carry the absolute temp path (var/folders/.../tmp.XXXXXXXX/...).
mkdir -p "${out_dir}"
tar --no-xattrs -czf "${bundle_file}" -C "${temp_dir}" "${top}"

# Verify the archive is free of AppleDouble members. `tar -tzf` cannot be used
# for this on macOS: it hides them. python3's tarfile sees every member.
if command -v python3 >/dev/null 2>&1; then
    if ! python3 - "${bundle_file}" <<'PY'
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:gz") as tf:
    bad = [n for n in tf.getnames() if n.startswith("._") or "/._" in n]
if bad:
    print("bundle: archive contains macOS AppleDouble files:", file=sys.stderr)
    for name in bad[:10]:
        print(f"  {name}", file=sys.stderr)
    raise SystemExit(1)
PY
    then
        rm -f "${bundle_file}"
        echo "bundle: removed the bad archive; check COPYFILE_DISABLE in this script" >&2
        exit 1
    fi
else
    echo "bundle: warning - python3 not found, skipped AppleDouble verification" >&2
fi

echo "Next: runpodctl send ${bundle_file}"
