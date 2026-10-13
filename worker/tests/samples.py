"""Synthetic test images, generated in code so the repository holds no confidential or binary fixtures."""

from __future__ import annotations

import math

import cv2
import numpy as np


def photo(h: int = 480, w: int = 640, seed: int = 1) -> np.ndarray:
    """Photo-like: smooth gradients, soft coloured shapes and fine texture; no dominant flat background."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        img[:, :, c] = 60 + 120 * (0.5 + 0.5 * np.sin(x / (w / (2 + c)) + y / (h / (1 + c)) + c))
    for _ in range(25):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(15, 90)
        colour = rng.uniform(0, 255, 3)
        mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r))[..., None]
        img = img * (1 - 0.7 * mask) + colour * 0.7 * mask
    for _ in range(15):
        centre = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        axes = (int(rng.uniform(10, 60)), int(rng.uniform(10, 60)))
        cv2.ellipse(img, centre, axes, rng.uniform(0, 180), 0, 360, rng.uniform(0, 255, 3).tolist(), -1, cv2.LINE_AA)
    texture = cv2.GaussianBlur(rng.normal(0, 25, (h, w)).astype(np.float32), (0, 0), 1.2)
    img += texture[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def good_photo(h: int = 480, w: int = 640) -> np.ndarray:
    """A photo that needs nothing: full tonal range, crisp fine detail, no noise."""
    p = photo(h, w).astype(np.float32)
    lo, hi = np.percentile(p, 1), np.percentile(p, 99)
    img = np.clip((p - lo) * 235 / (hi - lo) + 10, 0, 255).astype(np.uint8)
    for i in range(0, w, 7):
        cv2.line(img, (i, 0), (i + 120, h), (40, 40, 40), 1)
    return img


def line_art(h: int = 480, w: int = 640) -> np.ndarray:
    """Diagram-like: white background, black text, lines and a benzene-style ring."""
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(img, "C6H5-OH  Figure 3: Reaction", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "small label 12.5 mg/mL", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cx, cy, r = w // 3, h // 2, 70
    pts = [(int(cx + r * math.cos(math.pi / 3 * k + math.pi / 6)), int(cy + r * math.sin(math.pi / 3 * k + math.pi / 6))) for k in range(6)]
    for k in range(6):
        cv2.line(img, pts[k], pts[(k + 1) % 6], (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r // 2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(img, pts[0], (pts[0][0] + 60, pts[0][1]), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, (w // 2 + 20, h // 2), (w - 60, h // 2), (0, 0, 200), 2, cv2.LINE_AA)
    cv2.rectangle(img, (w - 200, 90), (w - 40, 200), (30, 30, 30), 1)
    return img


def add_noise(img: np.ndarray, sigma: float, seed: int = 7) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def low_contrast(img: np.ndarray, scale: float = 0.4, offset: float = 90) -> np.ndarray:
    return np.clip(img.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)


def faded_scan(img: np.ndarray) -> np.ndarray:
    """Line art as a poor scan: grey paper, faded ink, slight blur and noise."""
    return add_noise(blur(low_contrast(img, 0.45, 110), 1.2), 6)


def write(path, img: np.ndarray, ext: str | None = None, params=None) -> str:
    path = str(path)
    ok, buf = cv2.imencode(ext or path[path.rfind("."):], img, params or [])
    assert ok, path
    buf.tofile(path)
    return path


def poor_scan(seed: int = 1) -> np.ndarray:
    """A low-resolution scan of a text page: blurred, faded ink, yellowed paper, uneven lighting, noise, JPEG."""
    w, h = 1500, 900
    page = np.full((h, w), 255, np.uint8)
    for i in range(7):
        cv2.putText(page, "Salicylic acid (2.0 g, 14.5 mmol) was treated with acetic anhydride",
                    (50, 90 + i * 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    small = cv2.GaussianBlur(cv2.resize(page, (w // 3, h // 3), interpolation=cv2.INTER_AREA), (0, 0), 0.7).astype(np.float32)
    yy, xx = np.mgrid[0:small.shape[0], 0:small.shape[1]].astype(np.float32)
    light = 0.78 + 0.18 * (xx / xx.max()) - 0.08 * (yy / yy.max())
    gray = (70 + small / 255.0 * 155) * light + np.random.default_rng(seed).normal(0, 5, small.shape)
    bgr = np.dstack([gray * 0.86, gray * 0.95, gray]).clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
