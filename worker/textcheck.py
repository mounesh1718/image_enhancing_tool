"""Text check: reads the text in an image before and after enhancement with OCR and reports what changed.

It only checks; it never rewrites text. A changed number, or many changed characters, flags the result for review,
so a letter or value that the enhancement made ambiguous (e.g. "8" vs "6", "l" vs "t") is caught.

OCR: RapidOCR (Apache-2.0, ONNX models bundled with the package, runs offline on the CPU). Optional: when it is not
installed, the check reports itself as unavailable and enhancement works as before.
"""
from __future__ import annotations

import difflib
import re
import threading

import cv2
import numpy as np

_engine = None
_engine_lock = threading.Lock()
_import_error: str | None = None

# A line read before enhancement counts for the number check only when OCR was at least this confident.
CONFIDENT = 0.80
# Share of stable characters that may read differently after enhancement before the result is flagged.
MAX_CHANGE_RATIO = 0.03
# Lines that read differently after enhancement are read again in the original with the outline widened by this
# share of the line height; only characters read the same way both times in the original are compared.
REFERENCE_MARGIN = 0.25
# ... and once more enlarged by this factor.
REFERENCE_SCALE = 1.5
# Dense pages: at most this many lines are read (spread over the page), which bounds the time per image.
MAX_LINES = 60
# Images with less text than this are not checked (a lone label or arrow is not worth the time).
MIN_CHARACTERS = 6
MAX_EXAMPLES = 6
_NOT_ALNUM = re.compile(r"[^0-9A-Za-z]")


def available() -> bool:
    return _get_engine() is not None


def unavailable_reason() -> str | None:
    _get_engine()
    return _import_error


# Newer reading models (PP-OCRv5, Apache-2.0, published by RapidOCR with these SHA-256), installed by setup_ai.py
# into models/ppocr5. Measured on small scanned document text (6-9 pt at 100-150 dpi, three fonts, clean and
# scan-like): 86.9 % of characters read correctly with the English model against 83.3 % for the bundled PP-OCRv3
# (7 pt scans: 95-97 % against 82-95 %). The Chinese+Latin model read Latin text slightly worse (81.7 %).
# Without them the bundled PP-OCRv3 models are used.
OCR5_BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5"
OCR5_MODELS = {
    "det": ("ch_PP-OCRv5_det_mobile.onnx", OCR5_BASE + "/det/ch_PP-OCRv5_det_mobile.onnx",
            "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"),
    "rec": ("en_PP-OCRv5_rec_mobile.onnx", OCR5_BASE + "/rec/en_PP-OCRv5_rec_mobile.onnx",
            "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8"),
}


def ocr5_path(part: str):
    import ai_runtime
    return ai_runtime.models_base() / "ppocr5" / OCR5_MODELS[part][0]


def model_version() -> str:
    return "PP-OCRv5" if all(ocr5_path(p).exists() for p in OCR5_MODELS) else "PP-OCRv3"


def _get_engine():
    global _engine, _import_error
    if _engine is not None or _import_error is not None:
        return _engine
    with _engine_lock:
        if _engine is None and _import_error is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                if model_version() == "PP-OCRv5":
                    _engine = RapidOCR(det_model_path=str(ocr5_path("det")), rec_model_path=str(ocr5_path("rec")))
                else:
                    _engine = RapidOCR()
                # Document text is upright. The 180° guess sometimes flipped a short line after sharpening
                # ("99.3 %" read upside down as "% M'66"), which looked like a changed value.
                _engine.use_angle_cls = False
            except Exception as exc:  # missing package or model: the check is simply unavailable
                _import_error = f"{type(exc).__name__}: {exc}"
    return _engine


def read(img: np.ndarray) -> list[dict]:
    """Text lines as {"box": (x0, y0, x1, y1), "text", "confidence"}, top to bottom."""
    engine = _get_engine()
    if engine is None:
        return []
    result, _ = engine(_bgr8(img))
    lines = []
    for box, text, confidence in result or []:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        lines.append({"box": (min(xs), min(ys), max(xs), max(ys)), "text": text.strip(), "confidence": float(confidence)})
    lines.sort(key=lambda l: (round(l["box"][1] / 10), l["box"][0]))
    return lines


def _bgr8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        img = np.ascontiguousarray(img[:, :, :3])
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
    return img


def _overlap(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _touches(a, b) -> bool:
    """Boxes on the same text line that share some width."""
    overlap_y = min(a[3], b[3]) - max(a[1], b[1])
    return overlap_y > 0.5 * min(a[3] - a[1], b[3] - b[1]) and min(a[2], b[2]) > max(a[0], b[0])


def _squash(text: str) -> str:
    return "".join(text.split())


def _letters_digits(text: str) -> str:
    """Punctuation is left out: OCR swaps "." and "," between readings of the same blurry text."""
    return _NOT_ALNUM.sub("", text)


def _match(line: dict, lines: list[dict], used: set[int]) -> str:
    """The text of the line at the same place in another reading ("" when none overlaps enough)."""
    best, best_overlap = None, 0.0
    for i, other in enumerate(lines):
        if i not in used and (o := _overlap(line["box"], other["box"])) > best_overlap:
            best, best_overlap = i, o
    if best is None or best_overlap < 0.3:
        return ""
    used.add(best)
    return lines[best]["text"]


def _stable(text: str, reference: str) -> list[bool]:
    """Which characters of `text` were read the same way in a second, independent reading of the original.
    Characters OCR is unsure about (a blurry "0" read as "o" one time and "0" the next) are not stable."""
    stable = [False] * len(text)
    for block in difflib.SequenceMatcher(None, text, reference, autojunk=False).get_matching_blocks():
        for k in range(block.a, block.a + block.size):
            stable[k] = True
    return stable


def compare(before: list[dict], after: list[dict], reference: list[dict] | list[list[dict]] | None = None) -> dict:
    """Matches lines by position (the enhanced image has the same pixel size) and lists what reads differently.
    With a second reading of the original (`reference`), only characters read the same way twice are compared."""
    total = changed = 0
    numbers_changed: list[str] = []
    number_lines: list[int] = []  # indices into `before` of the lines with a changed number
    examples: list[str] = []
    used_after: set[int] = set()
    if reference and isinstance(reference[0], dict):
        reference = [reference]
    used_reference: list[set[int]] = [set() for _ in reference or []]
    for index, line in enumerate(before):
        old = _letters_digits(line["text"])
        if not old:
            continue
        new = _letters_digits(_match(line, after, used_after))
        stable = [True] * len(old)
        for view, used in zip(reference or [], used_reference):
            seen = _stable(old, _letters_digits(_match(line, view, used)))
            stable = [a and b for a, b in zip(stable, seen)]
        total += sum(stable)
        for tag, i0, i1, j0, j1 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            lost = sum(stable[i0:i1])  # stable characters that now read differently or are missing
            if not lost:
                continue
            changed += lost
            if len(examples) < MAX_EXAMPLES:
                examples.append(f"{_context(old, i0, i1)} → {_context(new, j0, j1)}")
            # A stable digit that now reads differently or is missing is a changed value. A digit read only
            # afterwards (a blurry "o" that now reads "0") does not count.
            if line["confidence"] >= CONFIDENT and any(old[k].isdigit() and stable[k] for k in range(i0, i1)):
                numbers_changed.append(f"{_context(old, i0, i1)} → {_context(new, j0, j1) if j1 > j0 else 'missing'}")
                number_lines.append((index, (i0 + max(i1 - i0, 1) / 2) / max(len(old), 1)))
    ratio = changed / total if total else 0.0
    return {
        "characters": total,
        "changedCharacters": changed,
        "changeRatio": round(ratio, 4),
        "numbersChanged": numbers_changed[:MAX_EXAMPLES],
        "numberLines": number_lines[:MAX_EXAMPLES],
        "examples": examples,
        "confidenceBefore": round(float(np.mean([l["confidence"] for l in before])), 3) if before else None,
        "confidenceAfter": round(float(np.mean([l["confidence"] for l in after])), 3) if after else None,
        "linesBefore": len(before),
        "linesAfter": len(after),
    }


def _context(text: str, start: int, end: int, pad: int = 3) -> str:
    """The changed part with a few characters around it, e.g. "tr[e]ated"."""
    lo, hi = max(0, start - pad), min(len(text), end + pad)
    return f"{text[lo:start]}[{text[start:end]}]{text[end:hi]}"


def _detect(img: np.ndarray) -> list[np.ndarray]:
    """Text line outlines (4 corner points each), top to bottom; at most MAX_LINES, spread over the image."""
    engine = _get_engine()
    boxes, _ = engine.text_detector(img)
    if boxes is None or len(boxes) == 0:
        return []
    boxes = list(engine.sorted_boxes(boxes))
    if len(boxes) > MAX_LINES:
        picks = np.linspace(0, len(boxes) - 1, MAX_LINES).round().astype(int)
        boxes = [boxes[i] for i in sorted(set(picks))]
    return boxes


def _recognise(img: np.ndarray, boxes: list[np.ndarray], margin: float = 0.0, scale: float = 1.0) -> list[dict]:
    """Reads the given line outlines in an image. `margin` widens each outline by that share of its height, which
    gives OCR a slightly different view of the same line (used for the second reading of the original)."""
    if not boxes:
        return []
    engine = _get_engine()
    outlines = []
    for box in boxes:
        box = np.asarray(box, dtype=np.float32)
        if margin:
            centre = box.mean(axis=0)
            height = float(np.linalg.norm(box[3] - box[0]))
            direction = box - centre
            norms = np.linalg.norm(direction, axis=1, keepdims=True)
            box = box + direction / np.maximum(norms, 1e-6) * height * margin
            box[:, 0] = box[:, 0].clip(0, img.shape[1] - 1)
            box[:, 1] = box[:, 1].clip(0, img.shape[0] - 1)
        outlines.append(box)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        outlines = [o * scale for o in outlines]
    crops = engine.get_crop_img_list(img, np.array(outlines))
    texts, _ = engine.text_recognizer(crops)
    lines = []
    for box, (text, confidence) in zip(boxes, texts):
        xs, ys = box[:, 0], box[:, 1]
        lines.append({"box": (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                      "text": str(text).strip(), "confidence": float(confidence)})
    return lines


def _ink_intact(original: np.ndarray, enhanced: np.ndarray, box, position: float) -> bool:
    """True when, around the changed character (at `position` along the line, 0-1), the enhanced image still holds
    the original's strokes and shows no new ones: the glyph is the same, only read differently."""
    import analyse
    x0, y0, x1, y1 = (float(v) for v in box)
    height = y1 - y0
    centre = x0 + position * (x1 - x0)
    half = 0.9 * height  # about one and a half characters either side
    h, w = original.shape[:2]
    ya, yb = max(0, int(y0 - height / 4)), min(h, int(y1 + height / 4))
    xa, xb = max(0, int(centre - half)), min(w, int(centre + half))
    if yb - ya < 4 or xb - xa < 4:
        return False
    metrics = analyse.compare(original[ya:yb, xa:xb], enhanced[ya:yb, xa:xb])
    # Measured: a replaced digit loses ~10 % of the window's edges and adds ~12 %; a cleaned, unchanged glyph
    # loses ~2 % and adds ~4 %.
    return metrics["lostEdgeRatio"] <= 0.05 and metrics["newEdgeRatio"] <= 0.08


class Reader:
    """Reads one original once and scores several enhanced versions of it on the same line outlines, so the best
    version can be chosen (enhance._choose).

    For each version: how legible it is (characters read, weighted by OCR confidence), which characters the original
    read the same way twice now read differently (as in `check`), and which characters the version shows that the
    original did not show readably ("unverified"). An AI de-blur may only be kept with no unverified digit: a sharp
    but invented number is worse than a blurred true one."""

    def __init__(self, original: np.ndarray):
        self.original = _bgr8(original)
        self.boxes = _detect(self.original)
        self.before = _recognise(self.original, self.boxes)
        self.has_text = sum(len(_squash(l["text"])) for l in self.before) >= MIN_CHARACTERS
        self._references: list[list[dict]] | None = None

    def references(self) -> list[list[dict]]:
        if self._references is None:
            self._references = [_recognise(self.original, self.boxes, margin=REFERENCE_MARGIN),
                                _recognise(self.original, self.boxes, scale=REFERENCE_SCALE)]
        return self._references

    def _stable_lines(self) -> list[tuple[str, list[bool]]]:
        """Each original line's letters and digits with which of them every reading of the original agrees on."""
        lines = []
        for i, line in enumerate(self.before):
            old = _letters_digits(line["text"])
            stable = [True] * len(old)
            for view in self.references():
                seen = _stable(old, _letters_digits(view[i]["text"]) if i < len(view) else "")
                stable = [a and b for a, b in zip(stable, seen)]
            lines.append((old, stable))
        return lines

    @staticmethod
    def _unproven(new: str, sources: list[tuple[str, list[bool]]]) -> tuple[int, int]:
        """(digits, letters) of `new` that no stable character of the source lines accounts for."""
        proven = [False] * len(new)
        for old, stable in sources:
            for block in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks():
                for k in range(block.size):
                    if stable[block.a + k]:
                        proven[block.b + k] = True
        digits = sum(1 for ch, ok in zip(new, proven) if not ok and ch.isdigit())
        return digits, sum(1 for ok in proven if not ok) - digits

    def score(self, enhanced: np.ndarray, strict: bool = False) -> dict:
        """With `strict` (for AI-sharpened text) the lines are also found in the enhanced version itself, so text
        that appears where the original shows no readable line is caught too."""
        enhanced = _bgr8(enhanced)
        after = _recognise(enhanced, self.boxes)
        result = compare(self.before, after, self.references())
        result.pop("numberLines", None)
        stable_lines = self._stable_lines()
        unverified_digits = unverified_letters = shown = 0
        if strict:
            for line in _recognise(enhanced, _detect(enhanced)):
                new = _letters_digits(line["text"])
                sources = [stable_lines[i] for i, b in enumerate(self.before) if _touches(line["box"], b["box"])]
                digits, letters = self._unproven(new, sources)
                unverified_digits, unverified_letters, shown = unverified_digits + digits, unverified_letters + letters, shown + len(new)
        else:
            for i, (old, stable) in enumerate(stable_lines):
                new = _letters_digits(after[i]["text"]) if i < len(after) else ""
                digits, letters = self._unproven(new, [(old, stable)])
                unverified_digits, unverified_letters, shown = unverified_digits + digits, unverified_letters + letters, shown + len(new)
        result |= {
            "legibility": round(sum(l["confidence"] * len(_letters_digits(l["text"])) for l in after), 2),
            "charactersShown": shown,
            "unverifiedDigits": unverified_digits,
            "unverifiedLetters": unverified_letters,
        }
        return result


def check(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """Runs the check. "flag" is True when a number changed or too many characters read differently.

    The lines are found once in the original; the same line outlines are then read in the original and in the
    enhanced image, so the two readings line up exactly. Only lines that read differently get a second look at the
    original, to tell real changes from characters OCR is unsure about."""
    if not available():
        return {"available": False}
    original, enhanced = _bgr8(original), _bgr8(enhanced)
    boxes = _detect(original)
    before = _recognise(original, boxes)
    if sum(len(_squash(l["text"])) for l in before) < MIN_CHARACTERS:
        return {"available": True, "textFound": False}
    after = _recognise(enhanced, boxes)
    differing = [i for i, (b, a) in enumerate(zip(before, after)) if _letters_digits(b["text"]) != _letters_digits(a["text"])]
    # Lines that read the same need no second look. For the others, two further views of the original: a character
    # counts only when every reading of the original agrees on it.
    references = [list(before), list(before)]
    if differing:
        subset = [boxes[i] for i in differing]
        for view, lines in enumerate((_recognise(original, subset, margin=REFERENCE_MARGIN),
                                      _recognise(original, subset, scale=REFERENCE_SCALE))):
            for i, line in zip(differing, lines):
                references[view][i] = line
    result = compare(before, after, references)
    # A claimed changed number is checked against the ink itself: when the line's strokes are all still there
    # (edges kept), the digits were not lost; OCR read a symbol differently (a currency sign read as "7" in the
    # blurry original and correctly afterwards). Such cases are reported but do not flag the image.
    confirmed, uncertain = [], []
    for text, (index, position) in zip(result["numbersChanged"], result.pop("numberLines")):
        if _ink_intact(original, enhanced, before[index]["box"], position):
            uncertain.append(text)
        else:
            confirmed.append(text)
    result["numbersChanged"] = confirmed
    result["numbersUncertain"] = uncertain
    result |= {
        "available": True,
        "textFound": True,
        "reader": model_version(),
        "linesChecked": len(boxes),
        "flag": bool(result["numbersChanged"]) or result["changeRatio"] > MAX_CHANGE_RATIO,
    }
    return result
