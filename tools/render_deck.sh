#!/usr/bin/env bash
# render_deck.sh <deck_dir> — turn a deck.html into deck.pdf + preview PNGs.
#
# Works from WSL using a Windows Chrome/Edge. Falls back to chromium/google-chrome
# if a Linux browser is present. Also writes preview/*.png so an agent can LOOK at
# the result and iterate.
#
# Usage:
#   tools/render_deck.sh /path/to/lecture_folder/deck
set -euo pipefail

DECK_DIR="${1:?usage: render_deck.sh <deck_dir>}"
DECK_DIR="$(cd "$DECK_DIR" && pwd)"
HTML="$DECK_DIR/deck.html"
PDF="$DECK_DIR/deck.pdf"
PREVIEW="$DECK_DIR/preview"

[ -f "$HTML" ] || { echo "missing $HTML" >&2; exit 1; }
mkdir -p "$PREVIEW"

find_browser() {
  local cands=(
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe"
  )
  for c in "${cands[@]}"; do [ -x "$c" ] && { echo "$c"; return; }; done
  for c in google-chrome google-chrome-stable chromium chromium-browser; do
    command -v "$c" >/dev/null 2>&1 && { echo "$(command -v "$c")"; return; }
  done
  return 1
}

BROWSER="$(find_browser)" || { echo "no Chrome/Edge found" >&2; exit 1; }
echo "browser: $BROWSER"

# Build a file:// URL that the chosen browser understands.
if [[ "$BROWSER" == *.exe ]]; then
  HTML_URL="file:///$(wslpath -m "$HTML")"
  PDF_OUT="$(wslpath -m "$PDF")"
else
  HTML_URL="file://$HTML"
  PDF_OUT="$PDF"
fi

"$BROWSER" --headless=new --disable-gpu --no-sandbox \
  --virtual-time-budget=10000 --no-pdf-header-footer \
  --print-to-pdf="$PDF_OUT" "$HTML_URL" 2>/dev/null || true

if [ ! -f "$PDF" ]; then
  echo "PDF not produced; trying legacy --headless" >&2
  "$BROWSER" --headless --disable-gpu --no-sandbox \
    --virtual-time-budget=10000 \
    --print-to-pdf="$PDF_OUT" "$HTML_URL" 2>/dev/null || true
fi

[ -f "$PDF" ] || { echo "failed to produce $PDF" >&2; exit 1; }

# PDF -> PNG previews (let the caller LOOK at each slide).
python3 - "$PDF" "$PREVIEW" <<'PY'
import pathlib, sys
try:
    import pymupdf
except ModuleNotFoundError:
    from pip._vendor import pymupdf  # type: ignore
    raise
path, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
doc = pymupdf.open(path)
for i, page in enumerate(doc):
    page.get_pixmap(dpi=90).save(out / f"p{i + 1:02d}.png")
print(f"{len(doc)} pages -> {out}")
PY

echo "pdf: $PDF"
echo "previews: $PREVIEW"
