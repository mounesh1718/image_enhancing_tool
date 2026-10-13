"""Builds test PDFs covering the layouts the PDF path must handle. Generated in code: no binary fixtures."""

from __future__ import annotations

import cv2
import numpy as np
import pymupdf

import samples


def _encode(img: np.ndarray, ext: str, params=None) -> bytes:
    ok, buf = cv2.imencode(ext, img, params or [])
    assert ok
    return buf.tobytes()


def build_feature_pdf(path: str, password: str | None = None) -> dict:
    """Five pages exercising tables, reuse, rotation, masks, scans, forms, inline and CMYK images.

    Returns the xrefs of interest so tests can assert on them.
    """
    doc = pymupdf.open()
    noisy_photo = _encode(samples.add_noise(samples.photo(240, 320), 12), ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 92])
    out: dict = {}

    # Page 1: text, a 3x3 table with an image in one cell, a JPEG photo outside the table, vector diagram.
    p = doc.new_page()
    p.insert_text((72, 60), "Quarterly report - Figure 1", fontsize=14)
    for r in range(4):
        p.draw_line((72, 100 + r * 60), (432, 100 + r * 60))
    for c in range(4):
        p.draw_line((72 + c * 120, 100), (72 + c * 120, 280))
    p.insert_text((80, 130), "Compound A")
    out["table_image"] = p.insert_image(pymupdf.Rect(196, 164, 308, 216),
                                        stream=_encode(samples.faded_scan(samples.line_art()), ".png"))
    out["photo"] = p.insert_image(pymupdf.Rect(72, 320, 312, 500), stream=noisy_photo)
    p.draw_circle((450, 400), 40, color=(0, 0, 1))
    p.draw_line((400, 450), (520, 520), color=(1, 0, 0), width=2)

    # Page 2: the same photo object reused at another size, a rotated copy of a different image,
    # a transparent PNG (soft mask) and a grayscale PNG.
    p = doc.new_page()
    p.insert_image(pymupdf.Rect(72, 72, 192, 162), xref=out["photo"])
    out["rotated"] = p.insert_image(pymupdf.Rect(250, 72, 340, 192),
                                    stream=_encode(samples.add_noise(samples.photo(120, 160, seed=4), 10), ".jpg"), rotate=90)
    rgba = np.dstack([samples.add_noise(samples.photo(150, 200, seed=5), 12), np.full((150, 200), 180, np.uint8)])
    out["alpha"] = p.insert_image(pymupdf.Rect(72, 220, 272, 370), stream=_encode(rgba, ".png"))
    gray = cv2.cvtColor(samples.add_noise(samples.photo(100, 140, seed=6), 12), cv2.COLOR_BGR2GRAY)
    out["gray"] = p.insert_image(pymupdf.Rect(300, 220, 440, 320), stream=_encode(gray, ".png"))

    # Page 3: a scanned page (full-page grayscale JPEG).
    p = doc.new_page()
    scan = cv2.cvtColor(samples.faded_scan(samples.line_art(842, 595)), cv2.COLOR_BGR2GRAY)
    out["scan"] = p.insert_image(p.rect, stream=_encode(scan, ".jpg"))

    # Page 4: an image inside a Form XObject, plus an identical copy of the photo stored as its own object.
    src = pymupdf.open()
    sp = src.new_page(width=200, height=150)
    sp.insert_image(sp.rect, stream=_encode(samples.add_noise(samples.photo(150, 200, seed=8), 12), ".jpg"))
    p = doc.new_page()
    p.show_pdf_page(pymupdf.Rect(72, 72, 272, 222), src, 0)
    src.close()
    out["duplicate"] = doc.get_new_xref()
    doc.update_object(out["duplicate"], "<<>>")
    doc.xref_copy(out["photo"], out["duplicate"])
    p.insert_image(pymupdf.Rect(300, 72, 420, 162), xref=out["duplicate"])

    # Page 5: an inline image and a CMYK image (not modifiable), plus a tiny icon.
    p = doc.new_page()
    out["cmyk"] = doc.get_new_xref()
    doc.update_object(out["cmyk"], "<< /Type /XObject /Subtype /Image /Width 40 /Height 40 "
                                   "/ColorSpace /DeviceCMYK /BitsPerComponent 8 >>")
    doc.update_stream(out["cmyk"], bytes(range(256)) * 25)
    p.insert_image(pymupdf.Rect(72, 72, 152, 152), xref=out["cmyk"])
    out["icon"] = p.insert_image(pymupdf.Rect(200, 72, 210, 82), stream=_encode(samples.photo(8, 8), ".png"))
    inline = (b"\nq 60 0 0 60 300 600 cm BI /W 4 /H 4 /CS /G /BPC 8 ID\n"
              + bytes([0, 69, 73, 255] * 4)  # data contains "EI" bytes on purpose
              + b"\nEI Q\n")
    contents = p.get_contents()[0]
    doc.update_stream(contents, doc.xref_stream(contents) + inline)

    save_args = {"garbage": 1}
    if password:
        save_args |= {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "user_pw": password, "owner_pw": password + "-owner"}
    doc.save(path, **save_args)
    doc.close()
    return out


def build_many_images_pdf(path: str, pages: int, images_per_page: int = 2, side: int = 400) -> None:
    """A large document for scale tests: distinct noisy photos on every page."""
    doc = pymupdf.open()
    base = samples.photo(side * 3 // 4, side)
    for i in range(pages):
        p = doc.new_page()
        p.insert_text((72, 60), f"Page {i + 1}")
        for k in range(images_per_page):
            img = samples.add_noise(np.roll(base, (i * 7 + k * 13) % side, axis=1), 10, seed=i * 31 + k)
            y = 90 + k * 330
            p.insert_image(pymupdf.Rect(72, y, 472, y + 300), stream=_encode(img, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]))
    doc.save(path, garbage=1)
    doc.close()
