"""Selective, conservative image enhancement and the worker commands that expose it.

Per image: probe header -> check limits -> decode -> analyse -> plan -> apply -> compare -> encode -> write.
Images that are already good, vector, bilevel or over budget are skipped with a reason; the original is kept.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

import analyse
import figure_classifier
import image_io
import panels as panel_finder
import photo_ai
import restore_ai
import sysinfo
import textcheck
from protocol import WorkerError, command

DEFAULT_LIMITS = {
    "maxPixels": 50_000_000,
    "maxMemoryBytes": 2 * 1024**3,
}

DEFAULT_SETTINGS = {
    "mode": "auto",  # auto | manual | off
    "denoise": True,
    "contrast": True,
    "sharpen": True,
    "resizeFactor": 1.0,
    # The result always has exactly the original's pixel size. Where more pixels would help (low-resolution text,
    # deblurring), the detail steps run on a finer grid internally and the result is reduced back to the original
    # size. False allows real enlargement (resizeFactor / automatic), which is off in the app.
    "keepPixelSize": True,
    # Read the text before and after with OCR (images with text only) and flag changed characters or numbers.
    "textCheck": True,
    # Redraw chemical structures when the reading is proven (see redraw/pipeline.py); otherwise enhance as usual.
    "redraw": False,
    # With "redraw": also redraw structures that cannot be fully proven, marked "not proven" for a careful check.
    "redrawAll": False,
    # AI clarity for photos (Real-ESRGAN, see photo_ai.py): photos only, never text or structures; flagged for review.
    "aiPhoto": False,
    # Composite images (a sheet of framed panels): each panel is processed as an image of its own.
    "panels": True,
    # AI repair (restore_ai.py): grain (SCUNet), JPEG blocks (FBCNN) and blurred text (DocRes) are tried where
    # needed, next to the filters, and the measured best version is kept (see _choose). The app turns it on.
    "aiRepair": False,
    # Auto mode, levels 1-4: gentle | standard | strong | maximum.
    "strength": "standard",
    # Bolder text and lines on low-quality text/diagram images: None = by level (Strong, Maximum), True, False.
    "boldText": None,
    # Manual mode only: 0 disables the operation.
    "denoiseStrength": 0.0,  # NLM filter strength h, 0-15
    "contrastStrength": 0.0,  # CLAHE clip limit, 0-4
    "sharpenAmount": 0.0,  # unsharp-mask amount, 0-2
}

# Peak working memory = decoded image x factor + fixed overhead (libraries, one denoise tile).
# Measured: input + output + LAB/LUT copies peak at ~4-5x the decoded size; tiled denoise adds ~150 MB.
MEMORY_WORK_FACTOR = 6
MEMORY_FIXED_OVERHEAD = 256 * 1024**2

# Denoise only when the measured "noise" behaves like noise (see analyse.noiseRatio); above this it is texture.
MAX_NOISE_RATIO = 0.85
# Diagrams more colourful than this keep their colours exactly (no contrast stretch).
DIAGRAM_MAX_COLORFULNESS = 15.0
# ... unless clearly coloured pixels are fewer than this share (a table header, chart lines on light paper).
LIGHTLY_COLOURED = 0.25
# Deblurring (Richardson-Lucy deconvolution with the measured blur). Below MIN_DEBLUR the image is sharp enough.
# Measured on a blurred table with known sharp original: similarity to the original rose from 0.84 (sharpening)
# to 0.88 (deconvolution + crisp edges), with letters intact.
MIN_DEBLUR = 0.9
# Up to this blur (pixels, at the original size) text is recovered reliably and made crisp. Beyond it the fine
# parts of letters are lost: deblurring and crisp edges then produce confident but wrong shapes (measured: "B" read
# as "D", "kg" as "hg"), so only part of the blur is removed and edges are left soft, and the image is flagged.
RELIABLE_BLUR = 1.6
DEBLUR_ITERATIONS = {"gentle": 0, "standard": 15, "strong": 25, "maximum": 35}

# Pages where at least this share looks like a document (but not all of it) are processed region by region.
MIXED_MIN_DOCUMENT = 0.30  # photos of objects with a paper label measured up to 0.23; composite pages 0.5-0.85

# Photos flatter than this (std of grey levels) get a moderate local-contrast boost; contrasty photos stay as they are.
LOCAL_CONTRAST_MAX_STD = 60.0
# Photos noisier than this (estimated sigma) get no local contrast: it would amplify the remaining grain.
HEAVY_NOISE = 8.0
# A contrast stretch never amplifies tonal differences by more than this.
MAX_STRETCH_GAIN = 1.8

# How strongly auto mode works. "upscaleDpi": images shown in a document below this resolution get more pixels
# (their displayed size never changes), so text and lines print and display crisply.
# "upscaleSide": standalone image files smaller than (2x, 1.5x) these widths/heights get more pixels, so they look
# clear when viewed large or zoomed; documents use the displayed resolution instead.
STRENGTH = {
    "gentle": {"sharpen": 0.6, "local": 0.0, "upscaleDpi": 100, "upscaleSide": (0, 0), "clean": False},
    # Local contrast changes how a photo looks (measured against the true picture it always costs accuracy), so it is
    # left to the stronger levels; Standard only repairs what is measured (noise, blur, JPEG blocks, fading).
    "standard": {"sharpen": 1.0, "local": 0.0, "upscaleDpi": 150, "upscaleSide": (1000, 1600), "clean": True},
    "strong": {"sharpen": 1.4, "local": 0.5, "upscaleDpi": 220, "upscaleSide": (1400, 2400), "clean": True},
    "maximum": {"sharpen": 1.7, "local": 0.75, "upscaleDpi": 300, "upscaleSide": (2000, 3200), "clean": True},
}
# Levels that recover heavy blur fully (crisper, but a character may come out wrong; flagged for review).
FULL_DEBLUR = ("strong", "maximum")
# Bolder text: how far (share of one pixel at the final size) thin strokes grow, per level. Levels without an entry
# only get it when "boldText" is switched on, at the Strong amount.
BOLD_AMOUNT = {"strong": 0.6, "maximum": 1.0}
# Strokes at least this wide (pixels at the original size) are already bold and are never thickened.
BOLD_THICK_STROKE = 2.5
# A gap this narrow or narrower (pixels at the final size) is never narrowed further, so the holes of "e", "a", "8",
# double bonds and close table lines stay open.
BOLD_MIN_GAP = 3
# Small letters are bolded by at most this share of a pixel (see _bold_cap).
BOLD_MIN_AMOUNT = 0.25

# Review thresholds: beyond these the result changed enough that a person should look at it.
# AI repair is tried only where it is needed (measured triggers), and kept only when it measures better (_choose).
GRAIN_MIN_NOISE = 4.0          # estimated grain level (sigma) from which SCUNet is tried
JPEG_MAX_QUALITY = 60          # JPEGs saved at this quality or lower are tried with FBCNN
DOC_DEBLUR_MIN_BLUR = 1.6      # text blurred at least this much (sigma, pixels) is tried with DocRes
AI_REPAIR_MAX_PIXELS = 4_000_000
# A de-blurred text version may show at most this share of letters, and no digit, that the original does not show
# readably (read the same way twice).
UNVERIFIED_LETTERS = 0.05
# Versions this close in legibility count as equally readable; then the less grainy, then the simpler one wins.
LEGIBILITY_TIE = 0.98
LESS_READABLE = 0.90
# Pictures without readable text: an AI version may lose or add at most this much more detail (share of edges) than
# the filters' version.
AI_DETAIL_MARGIN = 0.10

REVIEW_MIN_SSIM = 0.85
REVIEW_MAX_NEW_EDGES = 0.30
REVIEW_MAX_LOST_EDGES = 0.15
REVIEW_MAX_MEAN_SHIFT = 15.0


def estimate_memory(width: int, height: int, channels: int = 4, bits: int = 8, resize: float = 1.0) -> int:
    bytes_per_pixel = max(channels, 3) * (2 if bits > 8 else 1)
    return int(width * height * bytes_per_pixel * MEMORY_WORK_FACTOR * max(1.0, resize) ** 2 + MEMORY_FIXED_OVERHEAD)


def _merged(defaults: dict, given: dict | None) -> dict:
    merged = dict(defaults)
    for key, value in (given or {}).items():
        if key in defaults and value is not None:
            merged[key] = value
    return merged


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------- planning

def plan(analysis: dict, settings: dict) -> list[dict]:
    """Chooses operations from the analysis and the requested strength.

    Photos: noise reduction (only real noise, never texture), tone and local contrast, sharpening.
    Text and diagrams: shapes and colours are never altered. Faded or shaded scans on light paper get a clean,
    even background; low-resolution images get more pixels and crisp edges.
    """
    ops: list[dict] = []
    mode = settings["mode"]
    strength = STRENGTH.get(settings.get("strength") or "standard", STRENGTH["standard"])
    factor = float(settings["resizeFactor"] or 1.0)
    if mode != "manual" and abs(factor - 1.0) < 1e-6:
        factor = float(analysis.get("autoUpscale", 1.0))

    if mode == "manual":
        if settings["denoiseStrength"] > 0:
            ops.append({"op": "denoise", "h": _clamp(float(settings["denoiseStrength"]), 1, 15)})
        if settings["contrastStrength"] > 0:
            ops.append({"op": "contrast", "method": "clahe", "clip": _clamp(float(settings["contrastStrength"]), 0.5, 4)})
        if settings["sharpenAmount"] > 0:
            ops.append({"op": "sharpen", "amount": _clamp(float(settings["sharpenAmount"]), 0.05, 2), "sigma": 1.0})
    elif analysis.get("clean"):
        # A clean digital picture (chart, table, diagram, drawing): nothing unwanted to remove, so colours, shading,
        # lines and text stay exactly as they are. Only measured blur is reversed, for visibility.
        iterations = DEBLUR_ITERATIONS.get(settings.get("strength") or "standard", 15)
        blur = analysis.get("blur", 0.0)
        factor = 1.0 if abs(float(settings["resizeFactor"] or 1.0) - 1.0) < 1e-6 else factor
        if settings["sharpen"] and iterations > 0 and blur >= MIN_DEBLUR:
            if factor < 2.0 and analysis.get("upscaleAllowed", True):
                factor = 2.0
            share = 0.8 if blur <= RELIABLE_BLUR or settings.get("strength") in FULL_DEBLUR else 0.5
            ops.append({"op": "deblur", "sigma": round(blur * share * factor, 2), "iterations": iterations})
    elif analysis["content"] == "photo" and MIXED_MIN_DOCUMENT <= analysis.get("documentFraction", 0) and strength["clean"]:
        # A page mixing photos with text, tables, charts or drawings: each region gets its own treatment.
        photo_ops = [op for op in plan(analysis | {"content": "photo", "documentFraction": 0, "autoUpscale": 1.0,
                                                   "aiPhoto": False},
                                       settings | {"resizeFactor": 1.0, "sharpen": False})
                     # A tone stretch measured on the whole page would recolour labels and backgrounds.
                     if op["op"] != "resize" and op.get("method") != "stretch"]
        document_ops = [{"op": "smooth_speckle"}, {"op": "clean_background"}, {"op": "remove_specks"}]
        amount = _bold_amount(analysis, settings, True, factor)
        if amount:
            document_ops.append({"op": "bolden", "amount": amount, "thick": round(BOLD_THICK_STROKE * factor, 2), "gap": BOLD_MIN_GAP})
        ops.append({"op": "regional", "document": document_ops, "photo": photo_ops})
        if settings["sharpen"]:
            ops.append({"op": "sharpen", "amount": round(0.6 * strength["sharpen"], 2), "sigma": _sharpen_sigma(factor)})
    elif analysis["content"] == "photo":
        noise, sharp, dyn = analysis["noise"], analysis["sharpness"], analysis["dynamicRange"]
        # Same need as the classic steps below (they would denoise, deblur or sharpen this photo).
        if analysis.get("aiPhoto") and (analysis.get("blur", 0.0) >= MIN_DEBLUR or sharp < 600
                                        or (noise > 2.0 and analysis["noiseRatio"] < MAX_NOISE_RATIO)):
            # AI restoration (noise, compression, blur and softness in one step), then tone only.
            ops.append({"op": "ai_photo"})
            if settings["contrast"] and dyn < 170:
                ops.append(_tone(analysis))
            if settings["contrast"] and strength["local"] > 0 and analysis["contrastStd"] < LOCAL_CONTRAST_MAX_STD:
                ops.append({"op": "local_contrast", "clip": 2.0, "blend": strength["local"]})
        else:
            if settings["denoise"] and noise > 2.0 and analysis["noiseRatio"] < MAX_NOISE_RATIO:
                ops.append({"op": "denoise", "h": round(_clamp(noise, 3, 12), 1)})
            if settings["contrast"] and dyn < 170:
                ops.append(_tone(analysis))
            # Local contrast would bring back the grain that is left after denoising a very grainy photo.
            if settings["contrast"] and strength["local"] > 0 and analysis["contrastStd"] < LOCAL_CONTRAST_MAX_STD                     and noise <= HEAVY_NOISE:
                ops.append({"op": "local_contrast", "clip": 2.0, "blend": strength["local"]})
            photo_iterations = DEBLUR_ITERATIONS.get(settings.get("strength") or "standard", 15) // 2
            if settings["sharpen"] and photo_iterations > 0 and analysis.get("blur", 0.0) >= MIN_DEBLUR + 0.3 and noise < 4:
                share = 0.8 if analysis["blur"] <= RELIABLE_BLUR else 0.5
                ops.append({"op": "deblur", "sigma": round(analysis["blur"] * share * factor, 2), "iterations": photo_iterations})
            elif settings["sharpen"]:
                base = 0.6 if sharp < 100 else 0.45 if sharp < 600 else 0.0
                if analysis.get("jpegQuality", 100) <= JPEG_MAX_QUALITY:
                    base = 0.0  # sharpening brings out the blocks of a strongly compressed JPEG (AI repair removes them)
                if factor > 1:
                    base = max(base, 0.6)  # enlarged images need crisp edges on the finer pixel grid
                if base > 0:
                    ops.append({"op": "sharpen", "amount": round(base * strength["sharpen"], 2), "sigma": _sharpen_sigma(factor)})
    else:
        noise, sharp, dyn = analysis["noise"], analysis["sharpness"], analysis["dynamicRange"]
        # Light paper with little colour: a table with a coloured header or a chart with coloured lines counts
        # too (their colours are kept: background cleaning works on brightness only).
        light_document = (analysis["backgroundLevel"] >= 140
                          and (analysis["colorfulness"] < DIAGRAM_MAX_COLORFULNESS
                               or analysis.get("colorFraction", 1.0) < LIGHTLY_COLOURED))
        blur = analysis.get("blur", 0.0)
        iterations = DEBLUR_ITERATIONS.get(settings.get("strength") or "standard", 15)
        deblur = settings["sharpen"] and iterations > 0 and blur >= MIN_DEBLUR
        if deblur and factor < 2.0 and abs(float(settings["resizeFactor"] or 1.0) - 1.0) < 1e-6 \
                and analysis.get("upscaleAllowed", True):
            factor = 2.0  # blurred text needs a finer pixel grid to come back crisp
        if settings["denoise"] and noise > (2.0 if light_document else 3.0) and analysis["noiseRatio"] < MAX_NOISE_RATIO:
            ops.append({"op": "denoise", "h": 3.0 if noise < 6 else 5.0, "lumaOnly": True})
        # A narrow tonal range on a dark background or in a colour-coded drawing (e.g. a chemical structure) is
        # the design: only light, low-colour pages (scans, printed text) get their background cleaned.
        # Grey or shaded paper, or faded ink (measured on the darkest ink, so a sharp chart with black text on white
        # is left as it is).
        faded = analysis["backgroundLevel"] - analysis.get("inkLevel", analysis["p1"]) < 200
        if settings["contrast"] and light_document and (analysis["backgroundLevel"] < 245 or faded):
            if strength["clean"]:
                if settings["denoise"]:
                    ops.append({"op": "smooth_speckle"})
                ops.append({"op": "clean_background"})
                if settings["denoise"]:
                    ops.append({"op": "remove_specks"})
            elif dyn < 120:
                ops.append({"op": "contrast", "method": "stretch", "low": analysis["p1"], "high": analysis["p99"]})
        if deblur:
            # Undo the measured blur on the enlarged image, then make the ink/paper transition crisp
            # (documents on light paper only; ink keeps its own darkness).
            # Strong strength recovers heavy blur fully too (crisper, but a character may come out wrong; flagged).
            reliable = blur <= RELIABLE_BLUR or settings.get("strength") in FULL_DEBLUR
            share = 0.8 if reliable else 0.5
            ops.append({"op": "deblur", "sigma": round(blur * share * factor, 2), "iterations": iterations})
            if light_document and reliable:
                ops.append({"op": "crisp_text"})
        elif settings["sharpen"]:
            base = 0.7 if sharp < 200 else 0.45 if sharp < 1500 else 0.0
            if factor > 1:
                base = max(base, 0.6)  # crisp edges on the new, finer pixel grid
            if base > 0:
                ops.append({"op": "sharpen", "amount": round(base * strength["sharpen"], 2), "sigma": _sharpen_sigma(factor)})
            if settings.get("strength") == "maximum" and light_document and base > 0:
                ops.append({"op": "crisp_text"})
        amount = _bold_amount(analysis, settings, light_document, factor)
        if amount:
            ops.append({"op": "bolden", "amount": amount, "thick": round(BOLD_THICK_STROKE * factor, 2), "gap": BOLD_MIN_GAP})

    if abs(factor - 1.0) > 1e-6:
        # Resize before sharpening so sharpening acts on the final pixel grid.
        position = next((i for i, op in enumerate(ops) if op["op"] in ("sharpen", "deblur")), len(ops))
        if settings.get("keepPixelSize", True):
            # Same pixel size out: the detail steps (deblur, sharpen, crisp text, bolder text) run on a finer grid
            # and the result is reduced back to the original size, which gives smoother, cleaner edges.
            if factor > 1 and position < len(ops):
                ops = ops[:position] + [{"op": "fine_grid", "factor": _clamp(factor, 1.0, 4.0), "ops": ops[position:]}]
        else:
            ops.insert(position, {"op": "resize", "factor": _clamp(factor, 0.25, 4.0)})
    return ops


# A faded photo's veil: the blacks of all three colour channels are raised at least this much.
FADE_VEIL = 30
# Per-channel gains (colour cast correction) stay within this share of the common gain.
MAX_CAST_CORRECTION = 0.15


def _tone(analysis: dict) -> dict:
    """The tone step for a flat or faded photo: each colour channel's own black and white point when the photo is
    faded (which also removes the colour cast of the fading), otherwise a lightness stretch."""
    low, high = analysis.get("channelLow"), analysis.get("channelHigh")
    if low and high and min(low) >= FADE_VEIL:
        return {"op": "contrast", "method": "levels", "low": low, "high": high}
    return {"op": "contrast", "method": "stretch", "low": analysis["p1"], "high": analysis["p99"]}


def flatten(ops: list[dict]) -> list[dict]:
    """The operations as a flat list for reporting (the steps inside "fine_grid" follow it)."""
    flat: list[dict] = []
    for op in ops:
        if op["op"] == "fine_grid":
            flat.append({k: v for k, v in op.items() if k != "ops"})
            flat.extend(flatten(op.get("ops", [])))
        else:
            flat.append(op)
    return flat


def _bold_amount(analysis: dict, settings: dict, light_document: bool, factor: float) -> float:
    """How much to thicken thin text and lines (0 = not at all). Only ink on light paper.
    Left to the level (Strong, Maximum), only low-quality images get it: low resolution, blurred, faded or soft."""
    bold = settings.get("boldText")
    if bold is False or not light_document:
        return 0.0
    level = settings.get("strength") or "standard"
    if bold is not True:
        if level not in BOLD_AMOUNT:
            return 0.0
        low_quality = (factor > 1 or analysis.get("blur", 0.0) >= MIN_DEBLUR
                       or analysis["dynamicRange"] < 160 or analysis["sharpness"] < 200)
        if not low_quality:
            return 0.0
    return BOLD_AMOUNT.get(level, BOLD_AMOUNT["strong"])


def _sharpen_sigma(factor: float) -> float:
    """After enlarging, edges span more pixels, so the sharpening radius grows with the enlargement."""
    return round(max(1.0, 0.8 * factor), 2)


def auto_upscale(width: int, height: int, channels: int, display_width_pt: float | None,
                 settings: dict, limits: dict, standalone: bool = False) -> float:
    """More pixels for low-resolution images. Documents: when shown below the strength's dpi (the displayed size
    never changes). Standalone image files: when small. Never beyond the pixel and memory limits."""
    if settings["mode"] != "auto" or abs(float(settings["resizeFactor"] or 1.0) - 1.0) > 1e-6:
        return 1.0
    strength = STRENGTH.get(settings.get("strength") or "standard", STRENGTH["standard"])
    if standalone:
        side = max(width, height)
        double, half = strength["upscaleSide"]
        candidates = (2.0, 1.5) if side < double else (1.5,) if side < half else ()
    elif display_width_pt:
        dpi = width * 72.0 / float(display_width_pt)
        threshold = strength["upscaleDpi"]
        candidates = () if dpi >= threshold else (2.0, 1.5) if dpi < threshold * 0.75 else (1.5,)
    else:
        candidates = ()
    for factor in candidates:
        fits_pixels = width * height * factor * factor <= limits["maxPixels"]
        fits_memory = estimate_memory(width, height, channels, 8, factor) <= limits["maxMemoryBytes"]
        if fits_pixels and fits_memory:
            return factor
    return 1.0


# --------------------------------------------------------------------------- operations

def _luma_apply(img: np.ndarray, fn) -> np.ndarray:
    """Applies fn to the lightness channel only (colour images) to avoid colour fringes."""
    if img.ndim == 2:
        return fn(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = fn(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    del lab
    return out


# Non-local means needs ~25x the image size in working memory, so large images are denoised in overlapping
# tiles. The overlap covers the search + template radius, so tile seams do not change the result.
DENOISE_TILE = 1024


def _denoise(img: np.ndarray, op: dict) -> np.ndarray:
    if op.get("lumaOnly") and img.ndim == 3:
        # Diagrams and text: colours of lines and labels must stay exactly as they are.
        return _luma_apply(img, lambda luma: _denoise(luma, {"h": op["h"]}))
    h = float(op["h"])
    # A smaller search window keeps very large images tractable on ordinary CPUs.
    template, search = 7, (21 if img.shape[0] * img.shape[1] <= 4_000_000 else 11)
    pad = search // 2 + template // 2 + 2

    def nlm(tile: np.ndarray) -> np.ndarray:
        if tile.ndim == 2:
            return cv2.fastNlMeansDenoising(tile, None, h, template, search)
        return cv2.fastNlMeansDenoisingColored(tile, None, h, h, template, search)

    rows, cols = img.shape[:2]
    if rows <= DENOISE_TILE and cols <= DENOISE_TILE:
        return nlm(img)

    out = np.empty_like(img)
    for y in range(0, rows, DENOISE_TILE):
        for x in range(0, cols, DENOISE_TILE):
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(rows, y + DENOISE_TILE + pad), min(cols, x + DENOISE_TILE + pad)
            result = nlm(np.ascontiguousarray(img[y0:y1, x0:x1]))
            ty1, tx1 = min(rows, y + DENOISE_TILE), min(cols, x + DENOISE_TILE)
            out[y:ty1, x:tx1] = result[y - y0:ty1 - y0, x - x0:tx1 - x0]
    return out


def _contrast(img: np.ndarray, op: dict) -> np.ndarray:
    if op["method"] == "levels" and img.ndim == 3:
        # Each channel from its own black and white point to [4, 251]; the gains differ by at most
        # MAX_CAST_CORRECTION, so only a colour cast is removed, never a photo's own colour.
        low = np.array(op["low"], np.float32)
        gains = 247.0 / np.maximum(np.array(op["high"], np.float32) - low, 1.0)
        common = float(np.median(gains))
        gains = np.minimum(np.clip(gains, common * (1 - MAX_CAST_CORRECTION), common * (1 + MAX_CAST_CORRECTION)),
                           MAX_STRETCH_GAIN)
        luts = [np.clip(4 + (np.arange(256, dtype=np.float32) - low[c]) * gains[c], 0, 255).astype(np.uint8) for c in range(3)]
        out = img.copy()
        for c in range(3):
            out[:, :, c] = cv2.LUT(np.ascontiguousarray(img[:, :, c]), luts[c])
        return out
    if op["method"] in ("stretch", "levels"):
        low, high = float(np.mean(op["low"])), float(np.mean(op["high"]))
        if high - low < 8:
            return img
        # Limit amplification: widen the source range around its centre so the gain is at most MAX_STRETCH_GAIN.
        min_span = 247.0 / MAX_STRETCH_GAIN
        if high - low < min_span:
            centre = (low + high) / 2
            low, high = centre - min_span / 2, centre + min_span / 2
        # Map [low, high] to [4, 251] rather than the full range so highlights and shadows are not clipped hard.
        lut = np.clip(4 + (np.arange(256, dtype=np.float32) - low) * 247.0 / (high - low), 0, 255).astype(np.uint8)
        if img.ndim == 2:
            return cv2.LUT(img, lut)
        return _luma_apply(img, lambda luma: cv2.LUT(luma, lut))
    clahe = cv2.createCLAHE(clipLimit=float(op["clip"]), tileGridSize=(8, 8))
    return _luma_apply(img, clahe.apply)


def _local_contrast(img: np.ndarray, op: dict) -> np.ndarray:
    """CLAHE on lightness, blended with the original so the effect stays natural."""
    clahe = cv2.createCLAHE(clipLimit=float(op["clip"]), tileGridSize=(8, 8))
    blend = float(op["blend"])
    return _luma_apply(img, lambda luma: cv2.addWeighted(clahe.apply(luma), blend, luma, 1.0 - blend, 0))


def _clean_background(img: np.ndarray, op: dict) -> np.ndarray:
    """Evens out paper shading and greyness so the background becomes white and ink stands out.

    The paper brightness is estimated with a morphological closing (which removes the darker strokes) and divided
    out; ink keeps its relative darkness and shape. Ink is then darkened with a capped gain.
    """
    def clean(luma: np.ndarray) -> np.ndarray:
        h, w = luma.shape
        scale = min(1.0, 800.0 / max(h, w))
        small = cv2.resize(luma, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else luma
        k = max(15, int(min(small.shape) / 30)) | 1
        paper = _paper_level(small, k)
        if scale < 1:
            paper = cv2.resize(paper, (w, h), interpolation=cv2.INTER_LINEAR)
        norm = np.clip(luma.astype(np.float32) * 255.0 / np.maximum(paper.astype(np.float32), 1.0), 0, 255)
        # White point just below most of the paper (paper is the majority of a page), so paper grain turns white.
        white = float(np.clip(np.percentile(norm, 30), 200, 250))
        ink = float(np.percentile(norm, 1))
        gain = min(MAX_STRETCH_GAIN, 250.0 / max(white - ink, 1.0))
        return np.clip(255.0 - (white - norm) * gain, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _remove_faint_specks(clean(img))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a, b, a_seen, b_seen = _chroma(lab)
    # Only paper-coloured pixels (neutral, grey, yellowed) and ink are cleaned; pastel backgrounds such as coloured
    # labels or table headers, and clearly coloured areas (chart bars, coloured lines) keep their exact colour.
    keep_colour = ~analyse.paper_hue(a_seen, b_seen) & ((lab[:, :, 0] > 120) | (np.hypot(a_seen, b_seen) > STRONG_COLOUR))
    cleaned = _remove_faint_specks(clean(lab[:, :, 0]))
    lab[:, :, 0] = np.where(keep_colour, lab[:, :, 0], cleaned)
    # Paper becomes neutral white (no yellow/grey cast); coloured ink (darker) keeps its colour.
    paper_weight = np.clip((lab[:, :, 0].astype(np.float32) - 200.0) / 55.0, 0, 1) * ~keep_colour
    for c, chroma in ((1, a), (2, b)):
        lab[:, :, c] = np.clip(128.0 + chroma * (1.0 - paper_weight), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    del lab
    return out


# Colour (LAB a/b distance from grey) from which a dark area is coloured design (a chart bar), not ink.
STRONG_COLOUR = 18.0
# A filled area (a chart bar, a dark box or band) is this much darker than the paper ...
FILL_DROP = 30
# ... and its edge is a sharp step (grey levels per pixel); paper shading changes gradually.
FILL_EDGE = 12


def _paper_level(luma: np.ndarray, k: int) -> np.ndarray:
    """The paper's brightness everywhere (shading, greyness), from a closing that removes the darker strokes.

    Areas wider than the strokes (bars of a chart, filled boxes, dark bands) survive the closing and would be taken
    for dark paper and turned white. They are recognised by their sharp edge, left out, and the paper under them is
    filled in from the paper around them."""
    paper = cv2.morphologyEx(luma, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    low = (paper < float(np.percentile(paper, 95)) - FILL_DROP).astype(np.uint8)
    if low.any():
        count, labels = cv2.connectedComponents(low, connectivity=8)
        step = cv2.morphologyEx(paper, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
        rim = (low > 0) & (cv2.erode(low, np.ones((3, 3), np.uint8)) == 0)
        steps = np.bincount(labels[rim], weights=step[rim], minlength=count)
        sizes = np.bincount(labels[rim], minlength=count)
        sharp = steps / np.maximum(sizes, 1) >= FILL_EDGE
        sharp[0] = False
        fills = sharp[labels]
        if fills.any():
            keep = (~fills).astype(np.float32)
            spread = max(3.0 * k, min(luma.shape) / 8.0)
            total = cv2.GaussianBlur(paper.astype(np.float32) * keep, (0, 0), spread)
            weight = cv2.GaussianBlur(keep, (0, 0), spread)
            around = np.where(weight > 1e-3, total / np.maximum(weight, 1e-3), float(np.median(paper[~fills])))
            paper = np.where(fills, around, paper).astype(np.float32)
    return cv2.GaussianBlur(paper.astype(np.float32), (0, 0), k / 2)


# Colour noise (LAB a/b) above this is grain from a scan or camera, not design colour.
CHROMA_NOISE = 2.0


def _chroma(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Colour offsets a and b of every pixel, plus the same averaged over a few pixels for deciding what is coloured.

    In a grainy scan every pixel is randomly a little red, green or blue. Judged pixel by pixel, half the paper
    would count as "coloured" and be skipped, which turns grain into salt-and-pepper dots. Real colour (labels,
    table headers, chart lines) covers many pixels and survives the averaging.
    """
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    noise = max(analyse.estimate_noise(analyse.center_crop(lab[:, :, 1])),
                analyse.estimate_noise(analyse.center_crop(lab[:, :, 2])))
    if noise <= CHROMA_NOISE:
        return a, b, a, b
    sigma = _clamp(noise / 3.0, 1.0, 3.0)
    return a, b, cv2.GaussianBlur(a, (0, 0), sigma), cv2.GaussianBlur(b, (0, 0), sigma)


def _remove_faint_specks(luma: np.ndarray) -> np.ndarray:
    """Removes only tiny, faint, dot-like specks left on cleaned paper. Anything letter-sized, line-shaped or as
    dark as ink is never touched, so text (including grey text, subscripts and punctuation) is preserved."""
    dark = 255 - luma.astype(np.int32)
    ink = float(np.percentile(dark, 99.7))
    if ink < 40:
        return luma
    count, labels, stats, _ = cv2.connectedComponentsWithStats((dark > 20).astype(np.uint8), connectivity=8)
    peak = np.zeros(count, np.int32)
    np.maximum.at(peak, labels.ravel(), dark.ravel())
    area = stats[:, cv2.CC_STAT_AREA]
    max_area = max(4, int(luma.size * 0.000006))  # ~9 px on a 1.5-megapixel image
    compact = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) <= 2 * np.sqrt(max_area) + 1
    remove = (area <= max_area) & compact & (peak < 0.55 * ink)
    remove[0] = False
    if remove.any():
        # A faint grid line breaks into dots at this threshold; its pieces are not specks.
        on_line = np.unique(labels[_long_lines(luma) > 0.5])
        remove[on_line] = False
    if not remove.any():
        return luma
    out = luma.copy()
    out[remove[labels]] = 255
    return out


def _remove_specks(img: np.ndarray, op: dict) -> np.ndarray:
    """Removes isolated dark specks (photocopy and scanner dirt) from cleaned paper. A speck is small and lies far
    from any text or line: dots near letters (full stops, decimal points, the dot of an i) are always kept.
    Removed specks are filled from the surrounding paper (inpainting), so pastel backgrounds keep their colour."""
    gray = analyse.to_gray8(img)
    paper = float(np.percentile(gray, 90))
    if paper < 150:
        return img
    ink = (gray < paper - 60).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count < 3:
        return img
    area = stats[:, cv2.CC_STAT_AREA]
    # Letter size from the largest marks (letters, lines): specks are many but small, so a plain median would
    # mistake them for letters.
    sizes = area[1:]
    marks = (sizes >= 0.1 * sizes.max()) & (stats[1:, cv2.CC_STAT_HEIGHT] <= 0.3 * gray.shape[0])
    if marks.sum() < 3:
        return img
    letter = float(np.median(stats[1:, cv2.CC_STAT_HEIGHT][marks]))
    if letter < 6:
        return img
    big = area >= (0.5 * letter) ** 2
    big[0] = False
    # Specks are compact; anything line-shaped (a short bond, a dash, a minus sign) is kept whatever its size.
    longest = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    small = ~big & (longest < 0.4 * letter)
    small[0] = False
    if not small.any() or not big.any():
        return img
    # Kept within 0.6 letter heights of a letter or line: full stops, decimal points, dots of i, charges.
    reach = int(1.2 * letter) | 1
    near = cv2.dilate(big[labels].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach)))
    ids = np.unique(labels[(near == 0) & small[labels]])
    if ids.size == 0:
        return img
    mask = np.isin(labels, ids).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    if img.ndim == 2:
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return cv2.inpaint(np.ascontiguousarray(img[:, :, :3]), mask, 3, cv2.INPAINT_TELEA)


def _deblur(img: np.ndarray, op: dict) -> np.ndarray:
    """Richardson-Lucy deconvolution with a Gaussian of the measured blur, on lightness only (no colour fringes).
    It redistributes the light that blur spread out; it does not add detail that is not in the image."""
    sigma, iterations = float(op["sigma"]), int(op["iterations"])
    if sigma <= 0 or iterations <= 0:
        return img

    def rl(channel: np.ndarray) -> np.ndarray:
        observed = channel.astype(np.float32) + 1.0
        estimate = observed.copy()
        for _ in range(iterations):
            reblurred = cv2.GaussianBlur(estimate, (0, 0), sigma)
            estimate *= cv2.GaussianBlur(observed / np.maximum(reblurred, 1e-3), (0, 0), sigma)
            np.clip(estimate, 1.0, 256.0, out=estimate)
        return np.clip(estimate - 1.0, 0, 255).astype(np.uint8)

    return _luma_apply(img, rl)


def _crisp_text(img: np.ndarray, op: dict) -> np.ndarray:
    """Steeper ink/paper transition for text and line art on light paper, so letters have clean edges.
    Ink is only ever darkened (never lightened), and coloured areas are untouched."""
    def crisp(luma: np.ndarray, neutral: np.ndarray) -> np.ndarray:
        values = luma.astype(np.float32)
        sample = values[neutral] if neutral.any() else values.ravel()
        ink = float(np.percentile(sample, 0.3))
        if 255.0 - ink < 40:
            return luma
        mid, soft = (ink + 255.0) / 2.0, max(10.0, (255.0 - ink) / 9.0)
        curve = lambda v: 1.0 / (1.0 + np.exp(-(v - mid) / soft))
        lo, hi = curve(ink), curve(255.0)
        target_ink = ink * 0.75  # darker, but grey text stays grey rather than turning black
        out = target_ink + (curve(values) - lo) / (hi - lo) * (255.0 - target_ink)
        out = np.where(values < mid, np.minimum(out, values), out)  # ink never gets lighter
        return np.clip(out, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return crisp(img, np.ones(img.shape, bool))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _a, _b, a, b = _chroma(lab)
    # Pastel backgrounds stay as they are, and so do coloured strokes (red arrows, blue chart lines): only neutral
    # ink is darkened, so diagram colours are never changed.
    keep = (~analyse.paper_hue(a, b) & (lab[:, :, 0] > 120)) | (np.hypot(a, b) > 18)
    lab[:, :, 0] = np.where(keep, lab[:, :, 0], crisp(lab[:, :, 0], ~keep))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    del lab
    return out


def _bolden(img: np.ndarray, op: dict) -> np.ndarray:
    """Makes thin text and lines slightly thicker and darker: each stroke's soft edge is moved outwards by up to one
    pixel (op["amount"] of a pixel), in the stroke's own colour. Strokes that are already thick stay as they are,
    and a narrow gap is never narrowed, so letter holes, double bonds and close table lines stay open.
    Nothing new is drawn, and no pixel gets lighter."""
    gray = analyse.to_gray8(img)
    paper = float(np.percentile(gray, 90))
    if paper - float(np.percentile(gray, 1)) < 40:
        return img  # no clear ink on paper
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (gray <= min(threshold, paper - 25)).astype(np.uint8)
    # Thin ink only: what survives an opening by the "thick" width belongs to strokes that are already bold.
    k = max(3, int(round(float(op.get("thick", BOLD_THICK_STROKE)))) | 1)
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    near_thin = cv2.dilate(ink & (1 - thick), cross, iterations=2)
    allowed = near_thin & (1 - _narrow_gap(ink, int(op.get("gap", BOLD_MIN_GAP))))
    # Each pixel may take the colour of its darkest direct neighbour (the stroke it borders), so the whole soft
    # edge shifts outwards smoothly, coloured strokes keep their colour and grey ink stays grey.
    best_gray = gray.copy()
    best = img.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_gray = np.roll(gray, (dy, dx), axis=(0, 1))
        darker = shifted_gray < best_gray
        best_gray = np.where(darker, shifted_gray, best_gray)
        best = np.where(darker[..., None] if img.ndim == 3 else darker, np.roll(img, (dy, dx), axis=(0, 1)), best)
    amount = min(float(op.get("amount", 1.0)), _bold_cap(ink, float(op.get("thick", BOLD_THICK_STROKE)) / BOLD_THICK_STROKE))
    weight = cv2.GaussianBlur(allowed.astype(np.float32), (0, 0), 0.7) * allowed * amount
    if img.ndim == 3:
        weight = weight[..., None]
    source = img.astype(np.float32)
    out = np.minimum(source + (best.astype(np.float32) - source) * weight, source)
    return np.clip(out, 0, 255).astype(np.uint8)


def _bold_cap(ink: np.ndarray, scale: float) -> float:
    """Small letters can take only a light touch: measured with OCR on small scanned text (letters about 7 px tall),
    bolding by more than a quarter pixel made letters harder to tell apart (e -> o, l -> t). Larger text takes more."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT] / max(scale, 1.0)
    heights = heights[(heights >= 4) & (heights <= 60)]  # letter-sized marks at the original pixel size
    if heights.size < 5:
        return 1.0  # lines and shapes rather than text
    return float(np.clip((np.median(heights) - 6.0) / 12.0, BOLD_MIN_AMOUNT, 1.0))


def _narrow_gap(ink: np.ndarray, gap: int) -> np.ndarray:
    """Background pixels with ink on both sides within `gap` pixels (along any of four directions): filling them
    would narrow a gap of at most 2 x gap - 1 pixels."""
    size = 2 * gap + 1
    blocked = np.zeros(ink.shape, np.uint8)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        before = np.zeros((size, size), np.uint8)
        after = np.zeros((size, size), np.uint8)
        for i in range(1, gap + 1):
            before[gap - i * dy, gap - i * dx] = 1
            after[gap + i * dy, gap + i * dx] = 1
        blocked |= cv2.dilate(ink, before) & cv2.dilate(ink, after)
    return blocked & (1 - ink)


def _ai_jpeg(img: np.ndarray, op: dict) -> np.ndarray:
    return restore_ai.run("jpeg", img)


def _ai_grain(img: np.ndarray, op: dict) -> np.ndarray:
    return restore_ai.run("grain", img)


def _ai_deblur_text(img: np.ndarray, op: dict) -> np.ndarray:
    return restore_ai.run("deblur_text", img)


def _smooth_speckle(img: np.ndarray, op: dict) -> np.ndarray:
    """Edge-preserving smoothing that fades photocopy grain and scanner speckle while keeping text strokes."""
    # Brightness only: speckle is a brightness problem, and smoothing colour would wash thin coloured lines
    # (red arrows, blue curves) into grey. Stronger than h=6 also fades very faint printed lines.
    def smooth(luma: np.ndarray) -> np.ndarray:
        smoothed = _denoise(luma, {"h": 6.0})
        lines = _long_lines(luma)  # faint grid lines would fade into the paper
        return np.clip(luma * lines + smoothed * (1.0 - lines), 0, 255).astype(np.uint8)

    return _luma_apply(img, smooth)


# Grid lines, axes and table rules: at least this long (pixels, or 1/20 of the image) ...
LINE_MIN_LENGTH = 25
# ... and this much darker than the paper beside them (grey levels; full weight from twice this).
LINE_MIN_DEPTH = 4.0


def _long_lines(luma: np.ndarray) -> np.ndarray:
    """Weight (0-1) of thin horizontal and vertical lines such as chart grid lines, axes and table rules.

    They are long, straight and thin, unlike grain, specks and letters, and are kept exactly as they are by the
    steps that clean paper: a faint grey grid line looks like paper grain to a smoothing filter."""
    dark = cv2.GaussianBlur(255 - luma, (0, 0), 1.0)
    length = max(LINE_MIN_LENGTH, min(luma.shape) // 20)
    depth = np.zeros(luma.shape, np.float32)
    for line, band in (((1, length), (9, length)), ((length, 1), (length, 9))):
        thin = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones(line, np.uint8)).astype(np.float32)
        # The same opening with a thick band keeps paper shading and filled areas; what is left is the thin line.
        wide = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones(band, np.uint8)).astype(np.float32)
        depth = np.maximum(depth, thin - wide)
    weight = np.clip((depth - LINE_MIN_DEPTH) / LINE_MIN_DEPTH, 0, 1)
    # Also the line's soft edges (one pixel either side).
    return cv2.dilate(weight, np.ones((3, 3), np.uint8))


def _regional(img: np.ndarray, op: dict) -> np.ndarray:
    """Document regions and photo regions processed separately, then blended with a soft mask (no seams)."""
    mask = analyse.document_mask(img, (img.shape[1], img.shape[0]))
    if img.ndim == 3:
        mask = mask[:, :, None]
    document = img
    for step in op["document"]:
        document = _OPERATIONS[step["op"]](document, step)
    photo = img
    for step in op["photo"]:
        photo = _OPERATIONS[step["op"]](photo, step)
    # Flat, featureless areas (labels, frames, plain backgrounds) keep their exact colour; photo processing
    # applies where there is picture detail.
    gray = analyse.to_gray8(img).astype(np.float32)
    detail = cv2.GaussianBlur(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0)), (0, 0), 4.0)
    weight = np.clip(detail / 4.0, 0, 1)
    if img.ndim == 3:
        weight = weight[:, :, None]
    photo = photo.astype(np.float32) * weight + img.astype(np.float32) * (1.0 - weight)
    out = document.astype(np.float32) * mask + photo * (1.0 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen(img: np.ndarray, op: dict) -> np.ndarray:
    amount, sigma = float(op["amount"]), float(op["sigma"])

    def unsharp(channel: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(channel, (0, 0), sigma)
        sharpened = cv2.addWeighted(channel, 1.0 + amount, blurred, -amount, 0)
        # Leave near-flat areas alone so sharpening does not amplify residual noise.
        detail = cv2.absdiff(channel, blurred)
        return np.where(detail > 2, sharpened, channel)

    return _luma_apply(img, unsharp)


def _resize(img: np.ndarray, op: dict) -> np.ndarray:
    factor = float(op["factor"])
    h, w = img.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    # Lanczos keeps letter edges crisp when adding pixels; area averaging when reducing.
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4 if factor > 1 else cv2.INTER_AREA)


def _fine_grid(img: np.ndarray, op: dict) -> np.ndarray:
    """Runs the inner steps on a finer pixel grid (e.g. 2x), then reduces the result to exactly the original size."""
    h, w = img.shape[:2]
    factor = float(op["factor"])
    work = cv2.resize(img, (max(1, round(w * factor)), max(1, round(h * factor))), interpolation=cv2.INTER_CUBIC)
    for step in op["ops"]:
        work = _OPERATIONS[step["op"]](work, step)
    return cv2.resize(work, (w, h), interpolation=cv2.INTER_AREA)


_OPERATIONS = {
    "ai_jpeg": _ai_jpeg,
    "ai_grain": _ai_grain,
    "ai_deblur_text": _ai_deblur_text,
    "denoise": _denoise,
    "contrast": _contrast,
    "local_contrast": _local_contrast,
    "clean_background": _clean_background,
    "smooth_speckle": _smooth_speckle,
    "deblur": _deblur,
    "crisp_text": _crisp_text,
    "bolden": _bolden,
    "remove_specks": _remove_specks,
    "ai_photo": lambda img, op: _luma_safe_ai(img),
    "regional": _regional,
    "sharpen": _sharpen,
    "resize": _resize,
    "fine_grid": _fine_grid,
}


def apply(img: np.ndarray, ops: list[dict]) -> np.ndarray:
    """Runs the operations on colour/gray data; alpha is carried through untouched (only resized)."""
    alpha = None
    work = img
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        work = np.ascontiguousarray(img[:, :, :3])
    for op in ops:
        work = _OPERATIONS[op["op"]](work, op)
    if alpha is not None:
        if alpha.shape != work.shape[:2]:
            alpha = cv2.resize(alpha, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_LINEAR)
        work = np.dstack([work, alpha])
    return work


def quality_after(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """Quality measurements of the result, taken at the original's pixel size (sharpness and blur depend on scale)."""
    view = enhanced
    if enhanced.shape[:2] != original.shape[:2]:
        view = cv2.resize(enhanced, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    a = analyse.analyse(view)
    return {key: a[key] for key in ("sharpness", "noise", "blur", "dynamicRange", "contrastStd", "meanBrightness")}


# --------------------------------------------------------------------------- commands

def _enhance_one(img: np.ndarray, analysis: dict, settings: dict, limits: dict, display_width_pt, standalone: bool,
                 category: str | None = None):
    """One image (or one panel): redraw when proven, otherwise the planned steps. Returns (enhanced or None when
    nothing is needed, ops, redraw outcome). `category`: the picture's group from the app's classification (redraw
    is only tried for "chemical"); None when the request carries no classification."""
    channels = 1 if img.ndim == 2 else img.shape[2]
    analysis["upscaleAllowed"] = (img.shape[0] * img.shape[1] * 4 <= limits["maxPixels"]
                                  and estimate_memory(img.shape[1], img.shape[0], channels, 8, 2.0) <= limits["maxMemoryBytes"])
    analysis["autoUpscale"] = auto_upscale(img.shape[1], img.shape[0], channels, display_width_pt, settings, limits,
                                           standalone=standalone)
    analysis["aiPhoto"] = bool(settings.get("aiPhoto")) and img.dtype == np.uint8 \
        and img.shape[0] * img.shape[1] <= photo_ai.MAX_PIXELS and photo_ai.available()
    redraw_result = _try_redraw(img, analysis, settings, None if category is None else category == figure_classifier.CHEMICAL)
    if redraw_result.get("redrawn"):
        return _like(redraw_result.pop("image"), img), [{"op": "redraw"}], redraw_result, {}
    ops = plan(analysis, settings)
    # When the filters find nothing to do, AI repair may still help (JPEG blocks do not show in the measures the
    # filters go by): its version is then compared with the unchanged original.
    candidates = [{"name": "filters", "ops": ops, "image": apply(img, ops)} if ops
                  else {"name": "original", "ops": [], "image": img}]
    readers: dict = {}  # the original's text reading, made once and shared by the AI steps and the choice
    candidates += _ai_candidates(img, analysis, settings, ops, readers)
    if not ops and all(c["name"] == "ai_failed" for c in candidates[1:]):
        return None, [], redraw_result, {}
    chosen, choice = _choose(img, analysis, candidates, settings, readers)
    if not chosen["ops"]:
        return None, [], redraw_result, {}
    return chosen["image"], chosen["ops"], redraw_result, choice


def _reader(img: np.ndarray, analysis: dict, settings: dict, readers: dict):
    """The original's text reading for choosing between versions, or None (no text, or text check off)."""
    if "reader" not in readers:
        readers["reader"] = None
        if analysis["content"] == "text_or_diagram" and settings.get("textCheck", True) and textcheck.available():
            try:
                readers["reader"] = textcheck.Reader(img)
            except Exception:
                pass
    return readers["reader"]


def _reanalyse(img: np.ndarray, analysis: dict) -> dict:
    """Analysis of a repaired image for planning the remaining steps; what the image is stays as first decided."""
    fresh = analyse.analyse(img)
    for key in ("content", "documentFraction", "upscaleAllowed", "autoUpscale"):
        if key in analysis:
            fresh[key] = analysis[key]
    fresh["aiPhoto"] = False
    return fresh


def _ai_candidates(img: np.ndarray, analysis: dict, settings: dict, ops: list[dict], readers: dict) -> list[dict]:
    """Versions made with AI repair, to be compared with the filters' version. Only where needed: real grain, a
    strongly compressed JPEG, or blurred text whose letters the text check can verify."""
    if not settings.get("aiRepair") or settings["mode"] != "auto" or settings.get("strength") == "gentle" \
            or img.dtype != np.uint8 or img.shape[0] * img.shape[1] > AI_REPAIR_MAX_PIXELS \
            or any(o["op"] == "ai_photo" for o in flatten(ops)):
        return []
    pre: list[dict] = []
    quality = analysis.get("jpegQuality")
    if quality is not None and quality <= JPEG_MAX_QUALITY and restore_ai.available("jpeg"):
        pre.append({"op": "ai_jpeg"})
    if analysis["noise"] > GRAIN_MIN_NOISE and analysis["noiseRatio"] < MAX_NOISE_RATIO and restore_ai.available("grain"):
        pre.append({"op": "ai_grain"})
    found: list[dict] = []
    cleaned, cleaned_analysis = img, analysis
    try:
        if pre:
            cleaned = apply(img, pre)
            cleaned_analysis = _reanalyse(cleaned, analysis)
            rest = plan(cleaned_analysis, settings)
            found.append({"name": "ai_clean", "ops": pre + rest, "image": apply(cleaned, rest) if rest else cleaned})
        if analysis["content"] == "text_or_diagram" and restore_ai.available("deblur_text") \
                and max(analysis.get("blur", 0.0), cleaned_analysis.get("blur", 0.0)) >= DOC_DEBLUR_MIN_BLUR \
                and (reader := _reader(img, analysis, settings, readers)) is not None and reader.has_text:
            # Only when the original has readable text: the sharpened letters must be checked against it.
            sharp = apply(cleaned, [{"op": "ai_deblur_text"}])
            # No second de-blur or sharpening on top; background cleaning and the rest still apply.
            rest = plan(_reanalyse(sharp, analysis), settings | {"sharpen": False})
            found.append({"name": "ai_deblur", "ops": pre + [{"op": "ai_deblur_text"}] + rest,
                          "image": apply(sharp, rest) if rest else sharp})
    except Exception as exc:  # an AI failure never fails the image: the filters' version is used
        found.append({"name": "ai_failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    return found


def _residual_noise(img: np.ndarray) -> float:
    return round(analyse.estimate_noise(analyse.center_crop(analyse.to_gray8(img))), 2)


# The version the AI versions are measured against: the filters' version, or the original when no filter applies.
BASELINES = ("filters", "original")


def _choose(img: np.ndarray, analysis: dict, candidates: list[dict], settings: dict,
            readers: dict | None = None) -> tuple[dict, dict]:
    """Keeps the best version, measured, among the filters' version and the AI versions.

    Text: a version is ruled out when a number or more than 3 % of the characters read differently from the
    original; a de-blurred version also when it shows any digit (or more than 5 % letters) that the original does not
    show readably, because an AI can sharpen blurred text into the wrong characters. Of the rest, the most legible
    wins; equally legible ones are decided by less grain, then by fewer steps.
    Other images: an AI version may lose or add at most a little more detail than the filters' version, and wins
    when it has clearly less grain. Between equally good versions the AI-cleaned one wins: it was only made because
    grain or JPEG blocks were measured, and blocks do not show in the grain measure."""
    errors = [c["error"] for c in candidates if c["name"] == "ai_failed"]
    candidates = [c for c in candidates if c["name"] != "ai_failed"]
    if len(candidates) == 1:
        return candidates[0], ({"errors": errors} if errors else {})
    reader = _reader(img, analysis, settings, readers if readers is not None else {})
    textual = reader is not None and reader.has_text
    baseline = None  # the filters' version's detail measures (pictures without readable text)
    info: dict = {"candidates": []}
    if textual:
        info["originalLegibility"] = reader.score(img)["legibility"]
    valid = []
    for c in candidates:
        entry: dict = {"name": c["name"], "noise": _residual_noise(c["image"])}
        why = None
        if textual:
            sc = reader.score(c["image"], strict=c["name"] == "ai_deblur")
            entry |= {k: sc[k] for k in ("legibility", "changedCharacters", "unverifiedDigits", "unverifiedLetters")}
            if c["name"] not in BASELINES:
                if sc["numbersChanged"]:
                    why = "a number reads differently from the original"
                elif sc["changeRatio"] > textcheck.MAX_CHANGE_RATIO:
                    why = "too many characters read differently from the original"
                elif c["name"] == "ai_deblur" and sc["unverifiedDigits"]:
                    why = "shows digits the original does not show readably"
                elif c["name"] == "ai_deblur" and sc["unverifiedLetters"] > UNVERIFIED_LETTERS * max(sc["charactersShown"], 1):
                    why = "shows letters the original does not show readably"
        elif c["name"] == "ai_deblur":
            why = "the original has no readable text to check the sharpened letters against"
        if why is None and c["name"] not in BASELINES and not textual:
            if baseline is None:
                baseline = analyse.compare(img, candidates[0]["image"], analysis["noise"])
            m = analyse.compare(img, c["image"], analysis["noise"])
            if m["lostEdgeRatio"] > baseline["lostEdgeRatio"] + AI_DETAIL_MARGIN \
                    or m["newEdgeRatio"] > baseline["newEdgeRatio"] + AI_DETAIL_MARGIN:
                why = "more detail lost or added than with the filters"
        if why:
            entry["rejected"] = why
        else:
            valid.append((c, entry))
        info["candidates"].append(entry)
    pool = valid
    if textual:
        best = max(e["legibility"] for _, e in valid)
        pool = [(c, e) for c, e in valid if e["legibility"] >= LEGIBILITY_TIE * best]
    order = {"ai_clean": 0, "filters": 1, "original": 1, "ai_deblur": 2, "ai_photo": 3}
    chosen, entry = min(pool, key=lambda ce: (round(ce[1]["noise"] * 2) / 2, order[ce[0]["name"]]))
    info["chosen"] = chosen["name"]
    if textual and info["originalLegibility"] and entry["legibility"] < LESS_READABLE * info["originalLegibility"]:
        info["lessReadable"] = True
    if errors:
        info["errors"] = errors
    return chosen, info


def _enhance_panels(img: np.ndarray, boxes, settings: dict, limits: dict, source: dict, classification: dict | None = None):
    """Each framed panel processed as an image of its own (structure, photo, table, text ...); everything outside
    the panels (titles, gutters, frames) is kept exactly as it is."""
    out = img.copy()
    results, used = [], []
    for number, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        analysis = analyse.analyse(crop)
        analysis |= source  # what is known about the whole file (JPEG quality)
        category = figure_classifier.panel_category(classification, (x0, y0, x1, y1)) if classification else None
        enhanced, ops, redraw_result, choice = _enhance_one(crop, analysis, settings, limits, None, True, category)
        for item in redraw_result.get("findings") or []:  # places in the panel -> places in the whole picture
            if item.get("box"):
                item["box"] = [item["box"][0] + x0, item["box"][1] + y0, item["box"][2] + x0, item["box"][3] + y0]
        if enhanced is not None:
            out[y0:y1, x0:x1] = enhanced
        names = [o["op"] for o in flatten(ops)]
        review = []
        if analysis.get("blur", 0.0) > RELIABLE_BLUR and "deblur" in names:
            review.append("heavy_blur")
        if choice.get("lessReadable"):
            review.append("text_less_readable")
        results.append({"panel": number, "box": [x0, y0, x1, y1], "content": analysis["content"],
                        "operations": names, "redraw": redraw_result, "review": review, "choice": choice})
        for op in flatten(ops):
            if op["op"] not in {u["op"] for u in used}:
                used.append(op)
    return out, [{"op": "panels", "count": len(boxes)}] + used, results


def _luma_safe_ai(img: np.ndarray) -> np.ndarray:
    """AI restoration of a photo (colour or grey) at its own pixel size."""
    if img.ndim == 2:
        return cv2.cvtColor(photo_ai.restore(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)), cv2.COLOR_BGR2GRAY)
    return photo_ai.restore(img)


def _try_redraw(img: np.ndarray, analysis: dict, settings: dict, chemical: bool | None = None) -> dict:
    """{} when redraw is off or the image is not a candidate; otherwise the redraw outcome (with "image").
    A noisy or faint drawing is read again after the strongest cleaning (measured best for the reader; no bolder
    text, which made labels harder to read),
    provided the original read directly agrees.
    `chemical`: False when the picture is classified as anything but a chemical diagram (tables, graphs, photos and
    other images never reach MolScribe); None when no classification was sent (older requests)."""
    if not settings.get("redraw") or chemical is False or img.dtype != np.uint8 or analysis["content"] != "text_or_diagram":
        return {}
    from redraw import pipeline
    try:
        labels = pipeline.read_labels(img)
        if not pipeline.has_structure_strokes(img, labels):
            return pipeline.nonchemical()
        if not pipeline.available():
            return {"redrawn": False, "reasons": ["redraw is not installed: " + (pipeline.unavailable_reason() or "")]}
        outcome = pipeline.redraw(img, labels=labels)
        if not outcome["redrawn"]:
            ops = plan(analysis | {"aiPhoto": False}, settings | {"strength": "maximum", "boldText": False})
            cleaned = apply(img, ops) if ops else None
            if cleaned is not None:
                second = pipeline.redraw(cleaned, original=img)
                if second["redrawn"]:
                    return second
                if outcome.get("isChemical") is False and second.get("isChemical") is True:
                    outcome = second  # a noisy structure became recognisable after cleaning: keep its real findings
                outcome["reasons"] = outcome["reasons"] + [r for r in second["reasons"] if r not in outcome["reasons"]][:1]
            if settings.get("redrawAll"):
                # "Redraw all": not proven, but read as a structure; redrawn and marked for a careful check.
                third = pipeline.redraw(cleaned if cleaned is not None else img,
                                        original=img if cleaned is not None else None, mode="all")
                if not third["redrawn"] and cleaned is not None:
                    third = pipeline.redraw(img, mode="all")  # cleaning can also add marks the reader misreads
                if third["redrawn"]:
                    return third
                if outcome.get("isChemical") is False and third.get("isChemical") is True:
                    outcome = third
                outcome["reasons"] = outcome["reasons"] + [r for r in third["reasons"] if r not in outcome["reasons"]][:1]
    except Exception as exc:  # a failed attempt must never fail the image: it is enhanced as usual
        return {"redrawn": False, "reasons": [f"redraw failed ({type(exc).__name__}: {exc})"]}
    return outcome


def _like(picture: np.ndarray, original: np.ndarray) -> np.ndarray:
    """The redrawn picture in the original's channel layout (grey, colour, or colour with an opaque alpha)."""
    if original.ndim == 2:
        return cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    if original.shape[2] == 4:
        return np.dstack([picture, np.full(picture.shape[:2], 255, np.uint8)])
    return picture


def _text_check(original: np.ndarray, enhanced: np.ndarray, analysis: dict, settings: dict) -> dict:
    """OCR before/after for images that hold text (documents and mixed pages); photos are not read."""
    has_text = analysis["content"] == "text_or_diagram" or analysis.get("documentFraction", 0) >= MIXED_MIN_DOCUMENT
    if not settings.get("textCheck", True) or not has_text or original.dtype != np.uint8:
        return {"available": textcheck.available(), "skipped": True}
    try:
        return textcheck.check(original, enhanced)
    except Exception as exc:  # the check must never fail the enhancement itself
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _require_path(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise WorkerError("bad_request", f"'{key}' must be a file path")
    return value


def _skipped(reason: str, message: str, **extra) -> dict:
    return {"status": "skipped", "reason": reason, "message": message, **extra}


def _check_budget(width: int, height: int, channels: int, bits: int, settings: dict, limits: dict) -> dict | None:
    pixels = width * height
    if pixels > limits["maxPixels"]:
        return _skipped(
            "too_large",
            f"image is {pixels / 1e6:.1f} MP; the limit is {limits['maxPixels'] / 1e6:.1f} MP",
        )
    needed = estimate_memory(width, height, channels, bits, float(settings["resizeFactor"] or 1.0))
    if needed > limits["maxMemoryBytes"]:
        return _skipped(
            "too_large",
            f"processing would need about {needed / 1024**2:.0f} MB; the budget is {limits['maxMemoryBytes'] / 1024**2:.0f} MB",
        )
    return None


@command("probe")
def probe_cmd(args, ctx):
    """Header-only inspection: format, size, frames, and the memory enhancement would need."""
    info = image_io.probe(_require_path(args, "path"))
    result = info.to_json()
    if info.width and info.height:
        channels = 1 if info.details.get("components") == 1 else 4
        result["estimatedMemoryBytes"] = estimate_memory(info.width, info.height, channels, info.bits_per_channel)
    return result


def _decode_for_classification(args: dict):
    """The extracted picture as stored (the classifier never extracts), or (None, reason) when it cannot be read."""
    path = _require_path(args, "input")
    info = image_io.probe(path)
    if info.kind == "vector":
        return None, f"{info.format.upper()} is a vector graphic"
    if info.kind in ("unsupported", "unknown"):
        return None, "the picture format cannot be read"
    if info.width and info.height and info.width * info.height > DEFAULT_LIMITS["maxPixels"]:
        return None, "the picture is too large"
    return image_io.decode_buffer(image_io.read_bytes(path), int(args.get("frame") or 0)), None


@command("classify")
def classify_cmd(args, ctx):
    """Which group an extracted picture belongs to (chemical diagram, flowchart or graph, table, other image), with
    its panels classified separately, and MolScribe's second check for what is classified as chemical."""
    started = time.perf_counter()
    if not figure_classifier.available():
        return {"available": False, "reason": figure_classifier.unavailable_reason()}
    img, reason = _decode_for_classification(args)
    if img is None:
        return {"available": True, "model": figure_classifier.MODEL_NAME, "category": figure_classifier.OTHER,
                "confidence": None, "label": None, "uncertain": False, "panels": [], "note": f"not classified: {reason}"}
    boxes = panel_finder.detect(img) if args.get("panels", True) else []
    result = figure_classifier.classify(img, boxes, check_chemistry=bool(args.get("chemicalCheck", True)))
    result["milliseconds"] = round((time.perf_counter() - started) * 1000, 1)
    return result


@command("chemical_check")
def chemical_check_cmd(args, ctx):
    """MolScribe's reading of a picture the user marked as a chemical diagram (a reading, not a verification)."""
    img, reason = _decode_for_classification(args)
    if img is None:
        return {"status": "uncertain", "atoms": 0, "bonds": 0, "confidence": 0.0, "reasons": [reason]}
    return figure_classifier.chemical_check(img)


@command("ai_enhance")
def ai_enhance_cmd(args, ctx):
    """Optional AI enhancement of one picture, asked for by the user after the normal result: the suitable model(s)
    from ai_assist.suggestions, then the usual checks. Writes the AI version (accepted or rejected, so it can be
    compared) to "output"; the normal (OpenCV) result is never touched."""
    import ai_assist
    started = time.perf_counter()
    input_path = _require_path(args, "input")
    output_path = _require_path(args, "output")
    settings = _merged(DEFAULT_SETTINGS, args.get("settings"))
    classification = args.get("classification") if isinstance(args.get("classification"), dict) else None
    thumbs = args.get("thumbnails") or {}
    info = image_io.probe(input_path)
    if info.kind != "raster":
        return {"status": "not_eligible", "reasons": ["Only pictures made of pixels can be repaired by AI."], "suggestions": []}
    if info.width and info.height and _check_budget(info.width, info.height, 4, info.bits_per_channel, settings, DEFAULT_LIMITS):
        return {"status": "not_eligible", "reasons": ["The picture is too large for AI repair on this computer."], "suggestions": []}
    source = image_io.read_bytes(input_path)
    img = image_io.decode_buffer(source, int(args.get("frame") or 0))
    if img.dtype != np.uint8:
        return {"status": "not_eligible", "reasons": ["High-precision (16-bit) pictures are kept as they are."], "suggestions": []}
    analysis = analyse.analyse(img)
    if info.format == "jpeg" and "quality" in info.details:
        analysis["jpegQuality"] = info.details["quality"]
    opencv = None
    if args.get("opencv") and os.path.exists(args["opencv"]):
        opencv = image_io.decode_buffer(image_io.read_bytes(args["opencv"]))
    outcome = ai_assist.try_ai(img, analysis, settings, classification, opencv)
    result = {k: v for k, v in outcome.items() if k != "image"}
    if "image" in outcome:
        picture = outcome["image"]  # same channel layout as the original (enhance.apply keeps it)
        if picture.shape != img.shape:
            raise WorkerError("internal", "the AI version does not have the original pixel size")
        out_format = args.get("format") or image_io.OUTPUT_FORMAT[info.format]
        data = image_io.encode(picture, out_format, info)
        if out_format == info.format == "jpeg":
            data = image_io.copy_jpeg_metadata(source, data, info.details.get("components") == 4)
        elif out_format == info.format == "png":
            data = image_io.copy_png_metadata(source, data)
        image_io.write_atomic(output_path, data)
        result["output"] = {"path": output_path, "format": out_format, "width": int(picture.shape[1]),
                            "height": int(picture.shape[0]), "fileSize": len(data)}
        if thumbs.get("ai"):
            orientation = image_io.jpeg_orientation(source) if info.format == "jpeg" else 1
            result["thumbnail"] = image_io.write_thumbnail(picture, thumbs["ai"], int(thumbs.get("maxSide") or 256), orientation)
        names = {"ai_photo": "Real-ESRGAN", "ai_jpeg": "FBCNN", "ai_grain": "SCUNet", "ai_deblur_text": "DocRes"}
        devices = restore_ai.devices_used() | ({"Real-ESRGAN": photo_ai.device_used()} if photo_ai.device_used() else {})
        result["aiDevices"] = {names[o["op"]]: devices.get(names[o["op"]], "processor")
                               for o in result.get("operations", []) if o["op"] in names}
    result["milliseconds"] = round((time.perf_counter() - started) * 1000, 1)
    return result


@command("enhance")
def enhance_cmd(args, ctx):
    started = time.perf_counter()
    input_path = _require_path(args, "input")
    output_path = _require_path(args, "output")
    frame = int(args.get("frame") or 0)
    settings = _merged(DEFAULT_SETTINGS, args.get("settings"))
    limits = _merged(DEFAULT_LIMITS, args.get("limits"))
    # The app's classification of this picture and its panels ({"category", "panels": [{"box", "category"}]}, the
    # user's own choices included): redraw is only tried where it says "chemical".
    classification = args.get("classification") if isinstance(args.get("classification"), dict) else None
    thumbs = args.get("thumbnails") or {}
    thumb_side = int(thumbs.get("maxSide") or 256)

    # Test hook: lets integration tests simulate a worker crash on a specific image. Inactive unless set.
    crash_marker = os.environ.get("IQE_TEST_CRASH_MARKER")
    if crash_marker and crash_marker in os.path.basename(input_path):
        os._exit(70)

    info = image_io.probe(input_path)
    base = {"input": info.to_json()}

    if info.kind == "vector":
        return _skipped("vector", f"{info.format.upper()} is a vector graphic; it is kept unchanged", **base)
    if info.kind == "unsupported":
        return _skipped("unsupported_format", f"{info.format.upper()} images cannot be read without extra libraries", **base)
    if info.kind == "unknown":
        raise WorkerError("unsupported_format", "the file is not a recognised image format")
    if info.details.get("animated"):
        return _skipped("animated", "animated images are kept unchanged", **base)
    if settings["mode"] == "off":
        return _skipped("disabled", "enhancement is turned off for this image", **base)
    if info.width and info.height:
        over = _check_budget(info.width, info.height, 4, info.bits_per_channel, settings, limits)
        if over:
            return over | base

    t = time.perf_counter()
    source = image_io.read_bytes(input_path)
    if info.format == "gif" and source.count(b"\x00\x21\xf9\x04") > 1:
        return _skipped("animated", "animated GIFs are kept unchanged", **base)
    img = image_io.decode_buffer(source, frame)
    decode_ms = (time.perf_counter() - t) * 1000

    channels = 1 if img.ndim == 2 else img.shape[2]
    over = _check_budget(img.shape[1], img.shape[0], channels, img.dtype.itemsize * 8, settings, limits)
    if over:
        return over | base

    orientation = image_io.jpeg_orientation(source) if info.format == "jpeg" else 1
    result: dict = dict(base)
    if thumbs.get("original"):
        result["originalThumbnail"] = image_io.write_thumbnail(img, thumbs["original"], thumb_side, orientation)

    t = time.perf_counter()
    analysis = analyse.analyse(img)
    source_facts = {"jpegQuality": info.details["quality"]} if info.format == "jpeg" and "quality" in info.details else {}
    analysis |= source_facts
    result["analysis"] = analysis
    # What optional AI enhancement could try later, from the measurements and the picture's group. Nothing is run
    # now: the user decides after looking at the normal result (see ai_assist.py).
    import ai_assist
    result["aiSuggestions"] = ai_assist.suggestions(analysis, classification.get("category") if classification else None,
                                                    bool(classification and classification.get("panels")))

    if analysis["bilevel"]:
        return result | _skipped("bilevel", "pure black-and-white image; filtering cannot add detail")
    if img.dtype != np.uint8:
        return result | _skipped("high_bit_depth", f"{img.dtype.itemsize * 8}-bit images are kept unchanged")

    boxes = panel_finder.detect(img) if settings.get("panels", True) else []
    panel_results: list[dict] = []
    if boxes:
        enhanced, ops, panel_results = _enhance_panels(img, boxes, settings, limits, source_facts, classification)
        redraw_result, choice = {}, {}
        if not any(r["operations"] for r in panel_results):
            return result | _skipped("already_good", "image quality is already good; no enhancement needed")
    else:
        enhanced, ops, redraw_result, choice = _enhance_one(img, analysis, settings, limits,
                                                            args.get("displayWidthPt"), bool(args.get("standalone")),
                                                            classification.get("category") if classification else None)
        if enhanced is None:
            return result | _skipped("already_good", "image quality is already good; no enhancement needed") \
                | ({"redraw": redraw_result} if redraw_result else {})
    if settings.get("keepPixelSize", True) and enhanced.shape[:2] != img.shape[:2]:
        raise WorkerError("internal", "enhanced image does not have the original pixel size")
    enhance_ms = (time.perf_counter() - t) * 1000
    measured = enhanced
    if any(r.get("redraw", {}).get("redrawn") for r in panel_results):
        measured = enhanced.copy()
        for r in panel_results:
            if r.get("redraw", {}).get("redrawn"):
                x0, y0, x1, y1 = r["box"]
                measured[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    metrics = analyse.compare(img, measured, analysis["noise"])
    after = quality_after(img, enhanced)

    review: list[str] = []
    notes: list[str] = []
    if analysis["content"] == "text_or_diagram" or any(r["content"] == "text_or_diagram" for r in panel_results):
        review.append("text_or_diagram")
    if metrics["ssim"] < REVIEW_MIN_SSIM:
        review.append("large_change")
    if metrics["newEdgeRatio"] > REVIEW_MAX_NEW_EDGES:
        review.append("possible_artifacts")
    if metrics["lostEdgeRatio"] > REVIEW_MAX_LOST_EDGES:
        review.append("possible_detail_loss")
    if metrics["meanShift"] > REVIEW_MAX_MEAN_SHIFT:
        review.append("brightness_shift")
    if (analysis.get("blur", 0.0) > RELIABLE_BLUR or panel_results) and any(o["op"] == "deblur" for o in flatten(ops)) \
            and (not panel_results or any("heavy_blur" in r["review"] for r in panel_results)):
        review.append("heavy_blur")
    if any(o["op"] == "ai_photo" for o in flatten(ops)):
        review.append("ai_restored")
    if any(o["op"] == "ai_deblur_text" for o in flatten(ops)):
        review.append("ai_deblurred_text")
    if choice.get("lessReadable") or any("text_less_readable" in r["review"] for r in panel_results):
        review.append("text_less_readable")
    if any(r.get("redraw", {}).get("redrawn") for r in panel_results):
        review.append("redrawn_structure")
    if any(r.get("redraw", {}).get("redrawn") and not r["redraw"].get("proven", True) for r in panel_results):
        review.append("unproven_redraw")
    if redraw_result.get("redrawn"):
        # A redraw is a new drawing: pixel comparisons with the original do not measure its accuracy (the
        # chemical identity checks do), so the only flag is the required chemist approval.
        review = ["redrawn_structure"] + ([] if redraw_result.get("proven", True) else ["unproven_redraw"])
    text_check = _text_check(img, measured, analysis | ({"content": "text_or_diagram"} if panel_results else {}), settings) \
        if not redraw_result.get("redrawn") else {"skipped": True}
    if text_check.get("flag"):
        review.append("text_changed")

    out_format = args.get("format") or image_io.OUTPUT_FORMAT[info.format]
    if out_format not in image_io.RASTER_FORMATS:
        raise WorkerError("bad_request", f"unsupported output format: {out_format}")
    if out_format != info.format:
        notes.append(f"saved as {out_format.upper()} instead of {info.format.upper()}")

    t = time.perf_counter()
    data = image_io.encode(enhanced, out_format, info)
    if out_format == info.format == "jpeg":
        from_cmyk = info.details.get("components") == 4
        data = image_io.copy_jpeg_metadata(source, data, from_cmyk)
        if from_cmyk:
            review.append("color_conversion")
            notes.append("CMYK image converted to RGB")
    elif out_format == info.format == "png":
        data = image_io.copy_png_metadata(source, data)
        if info.details.get("colorType") == 3:
            notes.append("palette PNG saved as full-colour PNG")
    elif out_format == "webp" and info.details.get("hasMetadata"):
        notes.append("WebP metadata (colour profile/EXIF) is not preserved")
    del source
    image_io.write_atomic(output_path, data)
    encode_ms = (time.perf_counter() - t) * 1000

    if thumbs.get("enhanced"):
        result["enhancedThumbnail"] = image_io.write_thumbnail(enhanced, thumbs["enhanced"], thumb_side, orientation)

    result |= {
        "status": "completed",
        "operations": flatten(ops),
        "metrics": metrics,
        # Measured at the original pixel size so before and after are comparable.
        "after": after,
        "textCheck": text_check,
        "redraw": redraw_result,
        "panels": panel_results,
        # Which version was kept and how the versions measured (filters, AI clean, AI de-blur).
        "choice": choice,
        "needsReview": bool(review),
        "reviewReasons": review,
        "notes": notes,
        "output": {
            "path": output_path,
            "format": out_format,
            "width": int(enhanced.shape[1]),
            "height": int(enhanced.shape[0]),
            "fileSize": len(data),
        },
        "timings": {
            "decodeMs": round(decode_ms, 1),
            "enhanceMs": round(enhance_ms, 1),
            "encodeMs": round(encode_ms, 1),
            "totalMs": round((time.perf_counter() - started) * 1000, 1),
        },
        "memory": sysinfo.process_memory(),
    }
    used = {o["op"] for o in flatten(ops)}
    if used & {"ai_photo", "ai_jpeg", "ai_grain", "ai_deblur_text"}:
        # Where each AI model of this result ran (graphics chip or processor; chosen per model, see ai_runtime).
        names = {"ai_photo": "Real-ESRGAN", "ai_jpeg": "FBCNN", "ai_grain": "SCUNet", "ai_deblur_text": "DocRes"}
        devices = restore_ai.devices_used() | ({"Real-ESRGAN": photo_ai.device_used()} if photo_ai.device_used() else {})
        result["aiDevices"] = {names[op]: devices.get(names[op], "processor") for op in sorted(used) if op in names}
    return result


@command("thumbnail")
def thumbnail_cmd(args, ctx):
    """Preview for an image that is not being enhanced (e.g. skipped before decoding)."""
    path = _require_path(args, "input")
    info = image_io.probe(path)
    if info.kind != "raster":
        raise WorkerError("unsupported_format", "no preview available for this file type")
    limits = _merged(DEFAULT_LIMITS, args.get("limits"))
    if info.pixels and info.pixels > limits["maxPixels"]:
        if info.format != "jpeg":
            raise WorkerError("too_large", "image is too large to preview")
        # JPEG can be decoded at 1/8 scale without allocating the full image.
        img = cv2.imdecode(np.frombuffer(image_io.read_bytes(path), np.uint8), cv2.IMREAD_REDUCED_COLOR_8)
        if img is None:
            raise WorkerError("decode_failed", "the image data is corrupted")
    else:
        img = image_io.decode(path, int(args.get("frame") or 0))
    orientation = image_io.jpeg_orientation(image_io.read_bytes(path)) if info.format == "jpeg" else 1
    return image_io.write_thumbnail(img, _require_path(args, "output"), int(args.get("maxSide") or 256), orientation)


@command("assemble_tiff")
def assemble_tiff_cmd(args, ctx):
    """Builds a multi-page TIFF from per-page files (enhanced or original), in page order."""
    frames = args.get("frames")
    if not isinstance(frames, list) or not frames:
        raise WorkerError("bad_request", "'frames' must be a non-empty list of {path, frame}")
    output_path = _require_path(args, "output")
    source = args.get("source")
    info = image_io.probe(source) if source else None

    pages = []
    for i, entry in enumerate(frames):
        pages.append(image_io.decode(entry["path"], int(entry.get("frame") or 0)))
        ctx.progress(page=i + 1, total=len(frames))
    image_io.write_atomic(output_path, image_io.encode_multipage_tiff(pages, info))
    return {"path": output_path, "pages": len(pages)}


@command("configure")
def configure_cmd(args, ctx):
    """Worker-wide settings. With several workers, each should use one OpenCV thread."""
    threads = args.get("cvThreads")
    if threads is not None:
        cv2.setNumThreads(int(threads))
    return {"cvThreads": cv2.getNumThreads()}


@command("memory")
def memory_cmd(args, ctx):
    return sysinfo.process_memory() | {"availableBytes": sysinfo.available_memory(), "cpuCount": sysinfo.cpu_count()}
