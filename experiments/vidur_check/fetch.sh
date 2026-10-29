#!/usr/bin/env bash
# Sparse checkout of Vidur's open operation profiles (MIT) into data/vidur/.
# Only data/profiling is checked out, at a pinned commit.
set -euo pipefail
COMMIT="${1:-abae7f63aa857300f5cdc6f5e0d27860cd24721b}"
DEST="$(cd "$(dirname "$0")/../.." && pwd)/data/vidur"
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout --sparse https://github.com/microsoft/vidur "$DEST"
fi
git -C "$DEST" sparse-checkout set data/profiling
git -C "$DEST" checkout --quiet "$COMMIT"
git -C "$DEST" rev-parse HEAD
