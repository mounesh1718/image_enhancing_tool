"""Cheap image-quality measurements used to decide whether, and how gently, to enhance.

All measurements run on an 8-bit grayscale working copy that is downscaled (or cropped, for noise)
to at most ANALYSIS_MAX_SIDE pixels, so analysis cost stays roughly constant for huge images.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

ANALYSIS_MAX_SIDE = 1024
# A clean digital picture: grain at most this (sigma) ...
CLEAN_MAX_NOISE = 0.2
# ... this share of the background exactly one value (+-1) ...
CLEAN_MIN_EXACT = 0.6
# ... or, for text and diagrams on a shaded (gradient) background, this share of it without any grain ...
CLEAN_MIN_SMOOTH = 0.85
# ... white at least this light (99th percentile) and ink at least this dark (unless the ink is only greyed by blur).
CLEAN_MIN_WHITE = 235
CLEAN_MAX_INK = 60
CLEAN_BLURRED = 0.9


def to_gray8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def downscale(gray: np.ndarray, max_side: int = ANALYSIS_MAX_SIDE) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return gray
    return cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def center_crop(gray: np.ndarray, max_side: int = ANALYSIS_MAX_SIDE) -> np.ndarray:
    h, w = gray.shape[:2]
    ch, cw = min(h, max_side), min(w, max_side)
    y, x = (h - ch) // 2, (w - cw) // 2
    return gray[y:y + ch, x:x + cw]


def estimate_noise(gray: np.ndarray) -> float:
    """Noise sigma from the response of Immerkær's Laplacian-difference kernel on non-edge pixels.

    Uses the median (not the mean) of the response so fine texture and thin lines, which a mask cannot
    fully remove, do not read as noise. For Gaussian noise the kernel response has std 6*sigma.
    """
    if min(gray.shape[:2]) < 8:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = np.abs(cv2.filter2D(gray.astype(np.float32), -1, kernel))[1:-1, 1:-1]
    # Structure mask: coarse edges from a smoothed copy (robust to heavy noise) plus strong fine edges from
    # the raw image (thin lines that smoothing would erase).
    smoothed = cv2.GaussianBlur(gray, (0, 0), 2.0)
    edges = cv2.Canny(smoothed, 20, 60) | cv2.Canny(gray, 150, 300)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))[1:-1, 1:-1]
    flat = response[edges == 0]
    if flat.size < 64:
        return 0.0
    return float(np.median(flat) / (0.6745 * 6.0))


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian; low values mean a blurry image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_bilevel(img: np.ndarray) -> bool:
    """True for pure black/white images (e.g. 1-bit scans), where filtering cannot add detail."""
    h, w = img.shape[:2]
    step = max(1, int(math.sqrt(h * w / 1_000_000)))
    sample = img[::step, ::step]
    if sample.ndim == 3:
        sample = sample[:, :, :3] if sample.shape[2] == 4 else sample
        if not (np.array_equal(sample[:, :, 0], sample[:, :, 1]) and np.array_equal(sample[:, :, 1], sample[:, :, 2])):
            return False
        sample = sample[:, :, 0]
    return np.unique(sample).size <= 2


def analyse(img: np.ndarray) -> dict:
    gray_full = to_gray8(img)
    gray = downscale(gray_full)
    crop = center_crop(gray_full)
    blur = estimate_blur(crop)
    noise = estimate_noise(crop)
    sharp = sharpness(crop)
    # Random noise roughly halves when the image is averaged down 2x; real fine texture (crystals, fabric,
    # micrographs) does not. A high ratio means the "noise" is detail that must not be smoothed away.
    noise_half = estimate_noise(cv2.resize(crop, (max(1, crop.shape[1] // 2), max(1, crop.shape[0] // 2)),
                                           interpolation=cv2.INTER_AREA))
    noise_ratio = noise_half / noise if noise > 0.3 else 0.0
    del gray_full, crop

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    cumulative = np.cumsum(hist) / total
    p1 = int(np.searchsorted(cumulative, 0.01))
    p99 = int(np.searchsorted(cumulative, 0.99))
    # The darkest ink: in a sparse chart or drawing the darkest 1 % is mostly the soft edges of lines, not ink
    # (measured on the ink itself below, once the background is known).
    ink = int(np.searchsorted(cumulative, 0.002))
    # Black and white point of each colour channel: a faded photo has a coloured veil, its blacks raised in all three.
    channel_low = channel_high = None
    if img.ndim == 3 and img.shape[2] >= 3 and img.dtype == np.uint8:
        pixels = img[::max(1, img.shape[0] // 512), ::max(1, img.shape[1] // 512), :3].reshape(-1, 3)
        channel_low = [int(v) for v in np.percentile(pixels, 0.5, axis=0)]
        channel_high = [int(v) for v in np.percentile(pixels, 99.5, axis=0)]

    # Text, charts, chemical structures, diagrams and screenshots are dominated by one flat background.
    # Measured after a median filter (so scan noise does not hide a flat page) with a tolerance that
    # shrinks with the tonal range (so a low-contrast photo's narrow histogram is not mistaken for one).
    flat = cv2.medianBlur(gray, 5)
    flat_hist = cv2.calcHist([flat], [0], None, [256], [0, 256]).ravel()
    mode = int(flat_hist.argmax())
    if mode >= 128:
        # Measured on the marks clearly darker than the background, so a cell holding only "1.4" is measured on its
        # letters' core rather than their soft grey edges.
        marks = gray[gray < mode - 30]
        if marks.size >= 20:
            ink = min(ink, int(np.percentile(marks, 10)))
    tolerance = int(np.clip(round((p99 - p1) * 0.04), 2, 8))
    background_fraction = float(flat_hist[max(0, mode - tolerance):mode + tolerance + 1].sum() / total)
    # Scans and phone photos of pages have uneven lighting, which spreads the paper over many grey levels.
    # Measure again with the lighting evened out (paper estimated by a closing that removes dark strokes).
    if mode >= 100 and _colorfulness(img) < 20:
        k = max(15, int(min(flat.shape) / 30)) | 1
        paper = cv2.GaussianBlur(cv2.morphologyEx(flat, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (0, 0), k / 2)
        even = np.clip(flat.astype(np.float32) * 255.0 / np.maximum(paper.astype(np.float32), 1.0), 0, 255).astype(np.uint8)
        even_hist = cv2.calcHist([even], [0], None, [256], [0, 256]).ravel()
        even_mode = int(even_hist.argmax())
        even_fraction = float(even_hist[max(0, even_mode - 8):even_mode + 9].sum() / total)
        if even_fraction >= 0.6:  # clearly a page: most of it is one paper level once lighting is even
            background_fraction = max(background_fraction, even_fraction)
    del flat
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    content = "text_or_diagram" if background_fraction >= 0.45 else "photo"
    # A clean digital picture (a chart, table, diagram or drawing saved from software): no grain, its background one
    # exact value, real white and dark ink. It has nothing to clean: its colours, shading, lines and text are kept.
    raw = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    near = raw[max(0, mode - 12):mode + 13].sum()
    exact_background = float(raw[max(0, mode - 1):mode + 2].sum() / near) if near else 0.0
    # A shaded (gradient) background is not one value, but each pixel equals its neighbours: no grain at all.
    background = np.abs(gray.astype(np.int16) - mode) <= 12
    steady = np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 3).astype(np.int16)) <= 1
    smooth_background = float(steady[background].mean()) if background.any() else 0.0
    blur_measured = blur >= CLEAN_BLURRED
    # Faded: the ink itself is grey (a grey box or band with dark text on it is design, not fading), unless blur
    # alone greyed thin ink on white paper.
    faded = CLEAN_MAX_INK < ink < p99 - 20 and not (blur_measured and p99 >= CLEAN_MIN_WHITE)  # (blank: no ink)
    clean = noise <= CLEAN_MAX_NOISE and not faded and (
        exact_background >= CLEAN_MIN_EXACT or (content == "text_or_diagram" and smooth_background >= CLEAN_MIN_SMOOTH))
    document_fraction = float(document_mask(img).mean())
    colorfulness = _colorfulness(img)
    color_fraction = _color_fraction(img)

    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "channels": 1 if img.ndim == 2 else int(img.shape[2]),
        "dtype": str(img.dtype),
        "content": content,
        "noise": round(noise, 3),
        # Estimated Gaussian blur radius in pixels (0 = sharp); drives deblurring.
        "blur": round(blur, 2),
        "noiseRatio": round(noise_ratio, 3),
        "backgroundLevel": mode,
        # Share of the image that looks like a document page (scan, table, chart, text) rather than a photo.
        "documentFraction": round(document_fraction, 3),
        "colorfulness": round(colorfulness, 2),
        # Share of clearly coloured pixels: a table header or chart lines are a small share; a colour diagram more.
        "colorFraction": round(color_fraction, 3),
        # Measured at native resolution: downscaling would hide blur in large images.
        "sharpness": round(sharp, 2),
        "p1": p1,
        "p99": p99,
        "dynamicRange": p99 - p1,
        "inkLevel": ink,
        "channelLow": channel_low,
        "channelHigh": channel_high,
        "contrastStd": round(float(gray.std()), 2),
        "meanBrightness": round(float(gray.mean()), 2),
        "backgroundFraction": round(background_fraction, 3),
        "edgeDensity": round(edge_density, 4),
        "bilevel": is_bilevel(img),
        "exactBackground": round(exact_background, 3),
        "smoothBackground": round(smooth_background, 3),
        "clean": bool(clean),
    }


# Calibration of the re-blur ratio estimate against known Gaussian blur (text and lines, 780x420, light noise).
_BLUR_RAW = (0.79, 0.98, 1.10, 1.22, 1.40, 1.68, 1.80)
# Below this range of grey levels a picture is stretched before its blur is measured.
LOW_CONTRAST_RANGE = 120
_BLUR_TRUE = (0.0, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0)


def estimate_blur(gray: np.ndarray, s0: float = 1.0) -> float:
    """Gaussian blur radius (sigma, in pixels) of an image, from the re-blur gradient ratio at edges
    (Zhuo & Sim): at an edge |grad I| / |grad (I * G_s0)| = sqrt(s^2 + s0^2) / s. 0 means sharp."""
    if min(gray.shape[:2]) < 16:
        return 0.0
    # The edge finder below uses fixed thresholds: on a faded, low-contrast picture it finds almost no edges and the
    # image would wrongly count as sharp. The ratio measured at edges does not depend on contrast, so such pictures
    # are stretched to the full range first.
    lo, hi = np.percentile(gray, (0.5, 99.5))
    if hi - lo < LOW_CONTRAST_RANGE:
        gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
    g = gray.astype(np.float32)

    def grad(x):
        return np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3))

    m, m1 = grad(g), grad(cv2.GaussianBlur(g, (0, 0), s0))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.0), 20, 60) > 0
    if edges.sum() < 50:
        return 0.0
    strong = edges & (m > np.percentile(m[edges], 50))
    ratio = m[strong] / np.maximum(m1[strong], 1e-3)
    ratio = ratio[ratio > 1.01]
    if ratio.size < 50:
        return 0.0
    raw = float(np.median(s0 / np.sqrt(ratio ** 2 - 1)))
    return float(np.interp(raw, _BLUR_RAW, _BLUR_TRUE))


def paper_hue(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True where a colour is neutral, grey or yellowed like paper (LAB a/b offsets from 128). Pastel pink, blue
    and green backgrounds are design, not paper, and are excluded."""
    return (np.abs(a) <= 7) & (b >= -7) & (b <= 18)


def document_mask(img: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """Soft mask, 1.0 where the image looks like a document (light paper of a paper colour) and 0.0 in photo
    areas. Computed on a small copy in tiles, then smoothed so regions blend without seams.
    Returns the tile grid when size is None (cheap, for analysis); otherwise the mask at that (width, height)."""
    bgr = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if bgr.dtype != np.uint8:
        bgr = (bgr >> 8).astype(np.uint8) if bgr.dtype == np.uint16 else bgr.astype(np.uint8)
    h, w = bgr.shape[:2]
    scale = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    # A 3-pixel median: larger ones erase thin, faded text strokes once the image is reduced.
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 3)
    k = max(9, int(min(gray.shape) / 25)) | 1
    paper = cv2.GaussianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (0, 0), k / 3)
    even = gray.astype(np.float32) / np.maximum(paper.astype(np.float32), 1.0)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1] - 128, lab[:, :, 2] - 128
    saturated = np.hypot(a, b) > 40
    # Coloured areas (cells, leaves, skin) rather than thin coloured lines (chart curves, coloured text).
    colour_areas = cv2.morphologyEx(saturated.astype(np.uint8), cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    midtone = (even > 0.35) & (even < 0.85)
    tile = max(8, int(min(gray.shape) / 24))
    rows, cols = -(-gray.shape[0] // tile), -(-gray.shape[1] // tile)
    light = np.zeros((rows, cols), np.float32)
    paperish = np.zeros((rows, cols), np.float32)
    sat = np.zeros((rows, cols), np.float32)
    areas = np.zeros((rows, cols), np.float32)
    mid = np.zeros((rows, cols), np.float32)
    ink = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            sl = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
            light[i, j] = float(paper[sl].mean()) >= 150
            paperish[i, j] = float((even[sl] > 0.88).mean() >= 0.55
                                   and paper_hue(np.array(np.median(a[sl])), np.array(np.median(b[sl]))))
            sat[i, j] = float(saturated[sl].mean())
            areas[i, j] = float(colour_areas[sl].mean())
            mid[i, j] = float(midtone[sl].mean())
            ink[i, j] = float((even[sl] < 0.7).mean() > 0.005)
    # Photo regions grow from confident evidence (single tiles of a chart and of a micrograph can look alike):
    # clearly saturated colour, strongly mid-tone areas (black-and-white photos), or dark areas. Holes inside a
    # photo are filled, and photos extend only into neighbouring tiles that still contain saturated pixels, which
    # text, tables and charts on paper do not.
    near_mid = cv2.blur(mid, (3, 3), borderType=cv2.BORDER_REPLICATE)
    seeds = ((areas > 0.03) | ((mid > 0.5) & (near_mid > 0.42)) | (light < 0.5)).astype(np.uint8)
    photo = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    spotted = (sat > 0.003).astype(np.uint8)
    while True:  # grow until stable
        grown = np.maximum(photo, cv2.dilate(photo, np.ones((3, 3), np.uint8)) * spotted)
        if np.array_equal(grown, photo):
            break
        photo = grown
    photo = photo.astype(np.float32)
    grid = paperish * (1.0 - photo)
    # A paper region is a document only if it carries ink (text, lines, drawings): empty light areas such as
    # sky or walls in a photo are never "cleaned".
    count, labels = cv2.connectedComponents(grid.astype(np.uint8), connectivity=8)
    for label in range(1, count):
        region = labels == label
        if ink[region].sum() < max(2, 0.05 * region.sum()):
            grid[region] = 0
    grid = _drop_textured_regions(bgr, grid)
    # Grow documents by one tile so their edges (next to labels and frames) are included, but never into photos.
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # include strokes inside documents
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8)) * (1.0 - photo)
    if size is None:
        return grid
    mask = cv2.resize(grid, size, interpolation=cv2.INTER_NEAREST)
    edge = max(3, int(max(size) / 250)) | 1
    return cv2.GaussianBlur(mask, (0, 0), edge / 2)


# Paper, even noisy or speckled, is smooth at full resolution compared with photographed surfaces (crystals,
# fabric, shelves): measured median detail on paper areas 0.9-1.1 for scans vs 5.5 (fine) for a crystal photo
# and 5.0 (medium scale) for a shelf photo.
MAX_PAPER_FINE_DETAIL = 2.5
MAX_PAPER_MEDIUM_DETAIL = 3.5


def _drop_textured_regions(bgr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Removes "paper" regions whose light areas carry photographic texture at full resolution."""
    if not grid.any():
        return grid
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    fine = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.0))
    blur1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    medium = np.abs(blur1 - cv2.GaussianBlur(gray, (0, 0), 3.0))
    count, labels = cv2.connectedComponents(grid.astype(np.uint8), connectivity=8)
    full = cv2.resize(labels.astype(np.int32).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    for label in range(1, count):
        region = full == label
        values = gray[region]
        if values.size < 64:
            continue
        light = region & (gray > np.percentile(values, 40))
        if not light.any():
            continue
        if np.median(fine[light]) > MAX_PAPER_FINE_DETAIL or np.median(medium[light]) > MAX_PAPER_MEDIUM_DETAIL:
            grid[labels == label] = 0
    return grid


def _colorfulness(img: np.ndarray) -> float:
    """Hasler-Suesstrunk colourfulness (0 for grey images, ~15 slightly, >40 strongly colourful)."""
    if img.ndim == 2 or img.shape[2] < 3:
        return 0.0
    small = img[:, :, :3]
    h, w = small.shape[:2]
    scale = ANALYSIS_MAX_SIDE / max(h, w)
    if scale < 1:
        small = cv2.resize(small, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    b, g, r = [c.astype(np.float32) for c in cv2.split(small)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _color_fraction(img: np.ndarray) -> float:
    if img.ndim == 2 or img.shape[2] < 3:
        return 0.0
    small = img[:, :, :3]
    h, w = small.shape[:2]
    scale = ANALYSIS_MAX_SIDE / max(h, w)
    if scale < 1:
        small = cv2.resize(small, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    if small.dtype != np.uint8:  # 16-bit images: colour is judged on an 8-bit copy
        small = (small / 257).astype(np.uint8) if small.dtype == np.uint16 else small.astype(np.uint8)
    lab = cv2.cvtColor(np.ascontiguousarray(small), cv2.COLOR_BGR2LAB).astype(np.float32)
    return float((np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128) > 25).mean())


# --------------------------------------------------------------------------- comparison

def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity of two same-size 8-bit grayscale images (Gaussian window, sigma 1.5)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = blur(a * a) - mu_a2
    var_b = blur(b * b) - mu_b2
    cov = blur(a * b) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * cov + c2)) / ((mu_a2 + mu_b2 + c1) * (var_a + var_b + c2))
    return float(ssim_map.mean())


def _relative_edges(gray: np.ndarray) -> np.ndarray:
    """Canny edges with thresholds scaled to the image's own gradient strength, so a global contrast
    change alone does not create or remove edges."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    high = max(12.0, 3.0 * float((np.abs(gx) + np.abs(gy)).mean()))
    return cv2.Canny(gray, high / 2.5, high)


def compare(original: np.ndarray, result: np.ndarray, noise: float = 0.0) -> dict:
    """How much the enhancement changed the image *structure*. Used to flag results for human review.

    Both images are smoothed in proportion to the original's noise first, so removing noise is not
    counted as a structural change, while lost lines, halos and invented detail still are.
    """
    a = to_gray8(original)
    b = to_gray8(result)
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    a, b = downscale(a), downscale(b)
    sigma = 0.8 + noise / 6.0
    a = cv2.GaussianBlur(a, (0, 0), sigma)
    b = cv2.GaussianBlur(b, (0, 0), sigma)

    edges_a = _relative_edges(a)
    edges_b = _relative_edges(b)
    # Edges that appear away from any original edge: halos, ringing, amplified noise, invented structure.
    new_edges = np.count_nonzero(edges_b & ~cv2.dilate(edges_a, np.ones((3, 3), np.uint8)))
    # Original edges that vanished: lost thin lines, merged characters.
    lost_edges = np.count_nonzero(edges_a & ~cv2.dilate(edges_b, np.ones((3, 3), np.uint8)))
    # Floor of 1% of pixels so near-edgeless images (smooth gradients) do not produce huge ratios.
    base = max(int(np.count_nonzero(edges_a)), int(edges_a.size * 0.01), 1)

    return {
        "ssim": round(ssim(a, b), 4),
        "newEdgeRatio": round(float(new_edges) / base, 4),
        "lostEdgeRatio": round(float(lost_edges) / base, 4),
        "meanShift": round(abs(float(b.mean()) - float(a.mean())), 2),
    }
