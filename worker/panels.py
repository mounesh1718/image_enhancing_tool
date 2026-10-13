"""Finds the framed panels of a composite image (a figure sheet, a page of boxed images), so each panel can be
treated as an image of its own: a structure is redrawn or cleaned, a photo gets photo treatment, a table gets crisp
text. Without this, a sheet mixing photos, structures and text is processed as one "mixed" picture.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_PANEL = 0.02      # share of the image a panel covers at least
MAX_PANEL = 0.60      # ... and at most (a frame around the whole picture is not a panel)
MIN_COVERAGE = 0.35   # the panels together must cover this much of the image
RECTANGULAR = 0.85    # outline area / bounding box area
INSET = 4             # pixels left out at each panel edge (the frame line itself is kept as it is)


def detect(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Panel boxes (x0, y0, x1, y1) inside their frames, top-to-bottom then left-to-right; [] when the image is
    not a composite of at least two framed panels."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    area = float(h * w)
    boxes = []
    for threshold in (200, 170, 140):  # frames are drawn in light or dark grey
        lines = (gray < threshold).astype(np.uint8)
        contours, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            share = bw * bh / area
            if not MIN_PANEL <= share <= MAX_PANEL or min(bw, bh) < 48:
                continue
            outline = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(outline) != 4 or cv2.contourArea(contour) / float(bw * bh) < RECTANGULAR:
                continue
            boxes.append((x, y, x + bw, y + bh))
    boxes = _outermost(_distinct(boxes))
    if len(boxes) < 2 or sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes) / area < MIN_COVERAGE:
        return []
    inset = [(x0 + INSET, y0 + INSET, x1 - INSET, y1 - INSET) for x0, y0, x1, y1 in boxes]
    row_height = max(1.0, 0.05 * h)  # panels whose tops differ by a few pixels are in the same row
    rows = sorted(inset, key=lambda b: (round(b[1] / row_height), b[0]))
    return [tuple(int(v) for v in b) for b in rows]


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _distinct(boxes):
    kept = []
    for box in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(_iou(box, other) < 0.8 for other in kept):
            kept.append(box)
    return kept


def _outermost(boxes):
    """A table frame inside a panel is part of that panel, not a panel of its own."""
    def inside(a, b):
        return a != b and a[0] >= b[0] - 2 and a[1] >= b[1] - 2 and a[2] <= b[2] + 2 and a[3] <= b[3] + 2
    return [a for a in boxes if not any(inside(a, b) for b in boxes)]
