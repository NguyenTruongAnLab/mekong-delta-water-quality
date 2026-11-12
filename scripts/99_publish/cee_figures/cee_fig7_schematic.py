#!/usr/bin/env python3
"""
CEE Figure 7 — Conceptual Schematic of the Mekong Delta Regime Shift
====================================================================
Exports `blender/cee-hero-overlay.svg` (over `blender/cee_bg_render.png`) via
Inkscape to vector PDF, compresses the embedded 3D raster stream with pure DCT
(JPEG quality 92, `compress=False` under `/Filter /DCTDecode`) to keep the PDF
<500 KB while preserving 100% crisp vector text and paths, and renders the
matching 300 DPI PNG to `manuscript/figures/cee_composites/`.
"""
from __future__ import annotations
import io
import shutil
import subprocess
import zlib
from pathlib import Path
from PIL import Image
import pymupdf

ROOT = Path(__file__).resolve().parents[3]
BLENDER_DIR = Path(__file__).resolve().parent / "blender"
SVG_PATH = BLENDER_DIR / "cee-hero-overlay.svg"
BG_PNG = BLENDER_DIR / "cee_bg_render.png"
RAW_PDF = BLENDER_DIR / "CEE_Figure_7_Conceptual_Schematic_raw.pdf"
OPT_PDF = BLENDER_DIR / "CEE_Figure_7_Conceptual_Schematic.pdf"
OUT_DIR = ROOT / "manuscript" / "figures" / "cee_composites"

INKSCAPE_CANDIDATES = [
    Path(r"C:\Program Files\Inkscape\bin\inkscape.COM"),
    Path(r"C:\Program Files\Inkscape\bin\inkscape.exe"),
]


def _find_inkscape() -> Path | None:
    for p in INKSCAPE_CANDIDATES:
        if p.exists():
            return p
    which_p = shutil.which("inkscape")
    return Path(which_p) if which_p else None


def _optimize_pdf_jpeg(src_pdf: Path, dst_pdf: Path, quality: int = 92) -> None:
    doc = pymupdf.open(str(src_pdf))
    page = doc[0]
    target_xref = None
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        base_img = doc.extract_image(xref)
        if base_img["width"] >= 2000 and base_img["height"] >= 1000:
            target_xref = xref
            break

    if target_xref is not None:
        raw_stream = doc.xref_stream_raw(target_xref)
        is_deflated = raw_stream.startswith((b"\x78\xda", b"\x78\x9c", b"\x78\x01"))
        if is_deflated:
            try:
                raw_img = zlib.decompress(raw_stream)
                im = Image.open(io.BytesIO(raw_img))
            except Exception:
                base_img = doc.extract_image(target_xref)
                im = Image.open(io.BytesIO(base_img["image"]))
        else:
            base_img = doc.extract_image(target_xref)
            im = Image.open(io.BytesIO(base_img["image"]))

        if im.mode != "RGB":
            im = im.convert("RGB")
        jpg_buf = io.BytesIO()
        im.save(jpg_buf, format="JPEG", quality=quality, optimize=True)
        jpg_bytes = jpg_buf.getvalue()

        # CRITICAL: compress=False prevents wrapping JPEG bytes in FlateDecode
        doc.update_stream(target_xref, jpg_bytes, compress=False)
        doc.xref_set_key(target_xref, "Filter", "/DCTDecode")

    doc.save(str(dst_pdf), garbage=4, deflate=True)
    doc.close()


def generate_figure_7() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inkscape_bin = _find_inkscape()

    if inkscape_bin is not None and SVG_PATH.exists() and BG_PNG.exists():
        # Update relative/local URI inside SVG if run from a cloned repo path
        svg_txt = SVG_PATH.read_text(encoding="utf-8")
        bg_uri = BG_PNG.resolve().as_uri()
        import re
        svg_updated = re.sub(r'href="file:///[^"]*cee_bg_render\.png"', f'href="{bg_uri}"', svg_txt)
        if svg_updated != svg_txt:
            SVG_PATH.write_text(svg_updated, encoding="utf-8")

        subprocess.run(
            [str(inkscape_bin), str(SVG_PATH), "--export-type=pdf", f"--export-filename={RAW_PDF}"],
            check=True,
            capture_output=True,
            text=True,
        )
        _optimize_pdf_jpeg(RAW_PDF, OPT_PDF, quality=92)
        if RAW_PDF.exists():
            RAW_PDF.unlink()

    dest_pdf = OUT_DIR / "CEE_Figure_7_Conceptual_Schematic.pdf"
    dest_png = OUT_DIR / "CEE_Figure_7_Conceptual_Schematic.png"

    if OPT_PDF.exists():
        shutil.copy2(str(OPT_PDF), str(dest_pdf))
    elif not dest_pdf.exists():
        raise FileNotFoundError(f"Schematic PDF missing: {OPT_PDF}")

    # Render crisp 300 DPI PNG directly from the vector-overlaid PDF
    doc = pymupdf.open(str(dest_pdf))
    pix = doc[0].get_pixmap(dpi=300)
    doc.close()

    center_px = pix.pixel(pix.width // 2, pix.height // 2)
    landscape_px = pix.pixel(pix.width // 4, pix.height * 3 // 4)
    assert center_px[0] < 245 or landscape_px[0] < 245, (
        f"Background rendered blank white! (center={center_px}, landscape={landscape_px})"
    )

    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    im.save(dest_png, format="PNG", optimize=True, dpi=(300, 300))

    print(
        f"  ✓ CEE_Figure_7_Conceptual_Schematic  →  "
        f"{dest_pdf.stat().st_size / 1024:.1f} KB PDF + {dest_png.stat().st_size / 1024:.1f} KB PNG"
    )


if __name__ == "__main__":
    generate_figure_7()
