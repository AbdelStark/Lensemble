#!/usr/bin/env python3
"""Build a standalone, single-file offline export of the Demo-Night deck.

The live deck (index.html) pulls reveal.js and the fonts from a CDN (or from
./vendor/ when DECK_USE_VENDOR is true). This script bakes EVERYTHING into one
self-contained HTML file with no external references at all: reveal.js, the two
plugins, the theme, the seismograph script, and all five variable fonts inlined
as base64. The result opens by double-click (file://) with zero network, so it
is a safe fallback if the live presentation machine, Wi-Fi, or CDN misbehaves.

With --pdf it also renders the deck to a static, no-JS PDF (one slide per page,
in talk order) as the ultimate fallback. The PDF is captured from the standalone
file so it matches the bespoke fixed-1280x720 layout exactly (reveal's generic
print-pdf does not handle this deck's persistent fixed chrome).

Run:
    python3 scripts/build_presentation_export.py            # standalone HTML
    python3 scripts/build_presentation_export.py --pdf      # HTML + PDF (needs Chrome)
Output:
    docs/.../presentation/dist/lensemble-deck.html
    docs/.../presentation/dist/lensemble-deck.pdf   (with --pdf)
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# slide hashes in talk order: each depth slide right after its parent.
SLIDE_ORDER = [
    "0", "1", "2", "3", "4", "5", "6", "6/1", "7", "8",
    "9", "10", "10/1", "11", "12", "13", "14", "14/1", "14/2", "14/3",
]

PRES = Path(__file__).resolve().parent.parent / (
    "docs/plans/hackathons/codex-hackathon-paris-june/presentation"
)
REVEAL = PRES / "vendor" / "reveal.js@5.1.0"
FONTS = PRES / "vendor" / "fonts"
OUT = PRES / "dist" / "lensemble-deck.html"

# (family, file, style, weight-range) — mirrors the @font-face the live loader injects.
FACES = [
    ("Fraunces Variable", "fraunces-latin-full-normal.woff2", "normal", "300 900"),
    ("Fraunces Variable", "fraunces-latin-wght-italic.woff2", "italic", "300 900"),
    ("Newsreader Variable", "newsreader-latin-wght-normal.woff2", "normal", "200 800"),
    ("Newsreader Variable", "newsreader-latin-wght-italic.woff2", "italic", "200 800"),
    ("Spline Sans Mono Variable", "spline-sans-mono-latin-wght-normal.woff2", "normal", "300 700"),
]

REQUIRED = [
    REVEAL / "dist" / "reveal.css",
    REVEAL / "dist" / "reveal.js",
    REVEAL / "plugin" / "notes" / "notes.js",
    REVEAL / "plugin" / "highlight" / "highlight.js",
    *[FONTS / f for _, f, _, _ in FACES],
]


def ensure_vendor() -> None:
    missing = [p for p in REQUIRED if not p.exists()]
    if not missing:
        return
    print("[export] vendored assets missing, running vendor.sh (needs network) ...")
    subprocess.run(["./vendor.sh"], cwd=PRES, check=True)
    still = [p for p in REQUIRED if not p.exists()]
    if still:
        sys.exit(f"[export] ERROR: still missing after vendoring: {still}")


def js_safe(text: str) -> str:
    """Neutralise any literal </script> inside inlined JS so the tag does not close early."""
    return text.replace("</script", "<\\/script")


def font_faces_css() -> str:
    blocks = []
    for fam, fname, style, wght in FACES:
        b64 = base64.b64encode((FONTS / fname).read_bytes()).decode("ascii")
        blocks.append(
            f"@font-face{{font-family:'{fam}';font-style:{style};font-display:swap;"
            f"font-weight:{wght};src:url('data:font/woff2;base64,{b64}') format('woff2');}}"
        )
    return "\n".join(blocks)


def find_chrome() -> str | None:
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if Path(mac).exists():
        return mac
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def build_pdf(html_path: Path) -> None:
    """Capture each slide from the standalone file at 2x and assemble a PDF."""
    chrome = find_chrome()
    if not chrome:
        print("[export] --pdf skipped: Chrome/Chromium not found on this machine.")
        return
    file_url = html_path.resolve().as_uri()
    with tempfile.TemporaryDirectory(prefix="deck-pdf-") as tmp:
        frames = []
        for i, h in enumerate(SLIDE_ORDER):
            png = Path(tmp) / f"{i:02d}.png"
            subprocess.run(
                [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                 "--force-device-scale-factor=2", "--window-size=1280,720",
                 "--virtual-time-budget=2600", f"--screenshot={png}", f"{file_url}#/{h}"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if png.exists():
                frames.append(png)
        if not frames:
            print("[export] --pdf skipped: no frames rendered.")
            return
        pages = "\n".join(f'<div class="pg"><img src="{p.as_uri()}"></div>' for p in frames)
        print_html = Path(tmp) / "print.html"
        print_html.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><style>"
            "@page{size:1280px 720px;margin:0}html,body{margin:0;padding:0;background:#fff}"
            "img{width:1280px;height:720px;display:block}"
            ".pg{page-break-after:always}.pg:last-child{page-break-after:auto}"
            "</style></head><body>" + pages + "</body></html>",
            encoding="utf-8",
        )
        pdf_out = html_path.with_suffix(".pdf")
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             "--virtual-time-budget=4000", f"--print-to-pdf={pdf_out}", print_html.as_uri()],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if pdf_out.exists():
            print(f"[export] wrote {pdf_out} ({len(frames)} pages, "
                  f"{pdf_out.stat().st_size / 1_000_000:.2f} MB)")
        else:
            print("[export] --pdf failed: Chrome did not produce a PDF.")


def main() -> None:
    want_pdf = "--pdf" in sys.argv[1:]
    ensure_vendor()

    html = (PRES / "index.html").read_text(encoding="utf-8")
    reveal_css = (REVEAL / "dist" / "reveal.css").read_text(encoding="utf-8")
    theme_css = (PRES / "assets" / "theme.css").read_text(encoding="utf-8")
    reveal_js = js_safe((REVEAL / "dist" / "reveal.js").read_text(encoding="utf-8"))
    notes_js = js_safe((REVEAL / "plugin" / "notes" / "notes.js").read_text(encoding="utf-8"))
    highlight_js = js_safe(
        (REVEAL / "plugin" / "highlight" / "highlight.js").read_text(encoding="utf-8")
    )
    deck_js = js_safe((PRES / "assets" / "deck.js").read_text(encoding="utf-8"))

    head_inline = (
        "<style>/* reveal.js core (inlined) */\n" + reveal_css + "</style>\n"
        "  <style>/* variable fonts (inlined woff2) */\n" + font_faces_css() + "</style>\n"
        "  <style>/* The Offprint theme (inlined) */\n" + theme_css + "</style>"
    )
    # Replace the head loader (CDN/vendor switch + document.write) AND the theme link.
    html, n_head = re.subn(
        r'<script>\s*// Flip to true.*?</script>\s*'
        r'<link rel="stylesheet" href="\./assets/theme\.css" />',
        lambda _m: head_inline,
        html,
        count=1,
        flags=re.DOTALL,
    )

    body_inline = (
        "<script>/* reveal.js */\n" + reveal_js + "</script>\n"
        "  <script>/* notes plugin */\n" + notes_js + "</script>\n"
        "  <script>/* highlight plugin */\n" + highlight_js + "</script>\n"
        "  <script>/* the deck: seismograph + reveal init */\n" + deck_js + "</script>\n"
        "  <script>window.__initDeck__();</script>"
    )
    # Replace the body loader (dynamic script chain) AND the external deck.js tag.
    html, n_body = re.subn(
        r'<script>\s*\(function \(\) \{\s*var base = window\.__REVEAL_BASE__;.*?</script>\s*'
        r'<script src="\./assets/deck\.js"></script>',
        lambda _m: body_inline,
        html,
        count=1,
        flags=re.DOTALL,
    )

    if n_head != 1 or n_body != 1:
        sys.exit(
            f"[export] ERROR: anchor match failed (head={n_head}, body={n_body}). "
            "index.html structure changed; update the regex anchors."
        )

    # A standalone file has no external references left. Verify before writing.
    # Check for real load points (attributes, url(), the loader base, CDN host),
    # not bare substrings (which appear harmlessly in comments).
    for needle in ("cdn.jsdelivr.net", "__REVEAL_BASE__"):
        if needle in html:
            sys.exit(f"[export] ERROR: residual external reference '{needle}' remains.")
    if re.search(r'(?:href|src)\s*=\s*["\']\.?/?(?:assets|vendor)/', html):
        sys.exit("[export] ERROR: residual local file reference (href/src) remains.")
    if re.search(r'url\(\s*["\']?\.?/?(?:assets|vendor)/', html):
        sys.exit("[export] ERROR: residual local url() reference remains.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    size_mb = OUT.stat().st_size / 1_000_000
    print(f"[export] wrote {OUT}")
    print(f"[export] size: {size_mb:.2f} MB, fully self-contained (no network, no sibling files)")
    print("[export] open it by double-click, or: open " + str(OUT))

    if want_pdf:
        build_pdf(OUT)


if __name__ == "__main__":
    main()
