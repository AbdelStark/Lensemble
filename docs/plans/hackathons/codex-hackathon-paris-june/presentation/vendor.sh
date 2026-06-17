#!/usr/bin/env bash
# Vendor reveal.js + the variable fonts locally so the deck runs with zero
# network at the venue. Run once from this directory, then set
# window.DECK_USE_VENDOR = true in index.html.
set -euo pipefail

REVEAL_VERSION="5.1.0"
RBASE="https://cdn.jsdelivr.net/npm/reveal.js@${REVEAL_VERSION}"
ROUT="vendor/reveal.js@${REVEAL_VERSION}"

REVEAL_FILES=(
  "dist/reveal.css"
  "dist/reveal.js"
  "plugin/notes/notes.js"
  "plugin/highlight/highlight.js"
  "plugin/highlight/monokai.css"
)

echo "Vendoring reveal.js@${REVEAL_VERSION} into ${ROUT}/ ..."
for f in "${REVEAL_FILES[@]}"; do
  mkdir -p "${ROUT}/$(dirname "$f")"
  echo "  - $f"
  curl -fsSL "${RBASE}/${f}" -o "${ROUT}/${f}"
done

# Variable fonts (Fontsource). Flat into vendor/fonts/ — index.html's vendored
# path is ./vendor/fonts/<file>, matching the @font-face filenames it injects.
FBASE="https://cdn.jsdelivr.net/npm/@fontsource-variable"
FOUT="vendor/fonts"
FONT_FILES=(
  "fraunces/files/fraunces-latin-full-normal.woff2"
  "fraunces/files/fraunces-latin-wght-italic.woff2"
  "newsreader/files/newsreader-latin-wght-normal.woff2"
  "newsreader/files/newsreader-latin-wght-italic.woff2"
  "spline-sans-mono/files/spline-sans-mono-latin-wght-normal.woff2"
)

echo "Vendoring variable fonts into ${FOUT}/ ..."
mkdir -p "${FOUT}"
for f in "${FONT_FILES[@]}"; do
  base="$(basename "$f")"
  echo "  - ${base}"
  curl -fsSL "${FBASE}/${f}" -o "${FOUT}/${base}"
done

echo
echo "Done. Now set  window.DECK_USE_VENDOR = true  near the top of index.html."
echo "The deck then loads reveal.js AND the fonts from ./vendor/ — zero network."
