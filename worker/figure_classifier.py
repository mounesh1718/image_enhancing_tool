"""Classifies extracted pictures into the groups the app shows: chemical diagrams, flowcharts and graphs, tables, and
other images (photos included). Composite pictures (several framed panels) have each panel classified as well.

The classifier is DocumentFigureClassifier v2.5 (docling-project, MIT licence: an EfficientNet-B0 trained on figures
from PDF documents, 26 figure types), run with ONNX Runtime (MIT) on the processor. It only classifies: pictures are
obtained by the existing extraction code (DOCX image parts copied byte for byte, PDF images exported as stored, image
files read as they are), never by the classifier.

A picture classified as a chemical diagram gets a second check: MolScribe tries to read its atoms and bonds. That is a
reading, not a verification. A picture MolScribe cannot read reliably stays a chemical diagram and is marked
"possible chemical diagram, needs review". MolScribe is never run on pictures classified as anything else.

The model is downloaded once by setup_ai.py (16.9 MB, checked against its SHA-256); nothing is downloaded at run time.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

import ai_runtime

MODEL_NAME = "DocumentFigureClassifier v2.5"
COMMIT = "82d5545115ed6aa799d45ff3e16e96d386b995a7"
SOURCE_URL = f"https://huggingface.co/docling-project/DocumentFigureClassifier-v2.5/resolve/{COMMIT}/model.onnx"
SOURCE_SHA256 = "27ffc48c27ae4e12c99b6f6de0dd730005245e47b70dd0c1339e62cbac3ec4c0"
MODEL_FILE = "DocumentFigureClassifier-v2.5.onnx"

# The model's 26 figure types, in output order (config.json id2label).
LABELS = (
    "logo", "photograph", "icon", "engineering_drawing", "line_chart", "bar_chart", "other", "table", "flow_chart",
    "screenshot_from_computer", "signature", "screenshot_from_manual", "geographical_map", "pie_chart",
    "page_thumbnail", "stamp", "music", "calendar", "qr_code", "bar_code", "full_page_image", "scatter_plot",
    "chemistry_structure", "topographical_map", "crossword_puzzle", "box_plot",
)

CHEMICAL, GRAPH, TABLE, OTHER = "chemical", "flowchart_graph", "table", "other"
GROUP_OF = {
    "chemistry_structure": CHEMICAL,
    "flow_chart": GRAPH, "line_chart": GRAPH, "bar_chart": GRAPH, "pie_chart": GRAPH, "scatter_plot": GRAPH,
    "box_plot": GRAPH, "engineering_drawing": GRAPH,
    "table": TABLE,
}  # every other figure type (photograph, logo, screenshot, map, ...) is an "other image"

# Preprocessing published with the model: RGB, 224 x 224, ImageNet mean and the model's own standard deviation.
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.47853944, 0.4732864, 0.47434163], np.float32)

# A picture that looks chemical at least this much is kept with the chemical diagrams for a chemist to look at,
# rather than lost among the other images.
CHEMICAL_MIN = 0.35
# Below this confidence a suggestion is marked uncertain (the user should check it).
UNCERTAIN = 0.60
# A picture classified at least this confidently as one figure (a table, a flowchart, a structure) is not split into
# panels: the framed cells of a table and the boxes of a flowchart are not separate figures (a cell holding "0.9999"
# must never be read as a molecule). Measured: single tables and flowcharts 0.80-1.00; sheets of several figures
# (structures, photos, tables, charts side by side) 0.44-0.68.
WHOLE_FIGURE = 0.75

_lock = threading.Lock()
_session = None
_error: str | None = None


def model_path() -> Path:
    return ai_runtime.models_base() / "figures" / MODEL_FILE


def available() -> bool:
    return _get() is not None


def unavailable_reason() -> str | None:
    _get()
    return _error


def _get():
    global _session, _error
    if _session is not None or _error is not None:
        return _session
    with _lock:
        if _session is None and _error is None:
            path = model_path()
            if not path.exists():
                _error = "the picture classifier is not installed (run worker/setup_ai.py)"
                return None
            try:
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                # The processor: a 224 x 224 picture takes a few milliseconds, and the graphics chip stays free.
                _session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            except Exception as exc:  # a broken install must not stop enhancement
                _error = f"the picture classifier could not be loaded ({type(exc).__name__}: {str(exc)[:160]})"
    return _session


def _prepare(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # Transparent areas are shown on white paper in documents.
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = cv2.cvtColor((img[:, :, :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8), cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = (rgb / (256 if rgb.dtype == np.uint16 else 1)).clip(0, 255).astype(np.uint8)
    shrink = min(rgb.shape[:2]) > SIZE
    small = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return ((small.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def probabilities(images: list[np.ndarray]) -> np.ndarray:
    """The model's probabilities for each picture (one row of 26 per picture)."""
    session = _get()
    if session is None:
        raise RuntimeError(_error or "the picture classifier is not available")
    batch = np.stack([_prepare(img) for img in images])
    logits = session.run(None, {session.get_inputs()[0].name: batch})[0].astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def suggest(p: np.ndarray) -> dict:
    """The group suggested by one row of probabilities: the group with the highest total probability, except that a
    picture looking chemical at least CHEMICAL_MIN goes to the chemical diagrams."""
    totals = {CHEMICAL: 0.0, GRAPH: 0.0, TABLE: 0.0, OTHER: 0.0}
    for label, value in zip(LABELS, p):
        totals[GROUP_OF.get(label, OTHER)] += float(value)
    chemical = float(p[LABELS.index("chemistry_structure")])
    group = CHEMICAL if chemical >= CHEMICAL_MIN else max(totals, key=totals.get)
    top = np.argsort(p)[::-1][:3]
    best = LABELS[int(top[0])]
    return {
        "category": group,
        "confidence": round(totals[group], 3),
        "label": best,
        "labelConfidence": round(float(p[top[0]]), 3),
        "top": [{"label": LABELS[int(i)], "confidence": round(float(p[i]), 3)} for i in top],
        "uncertain": totals[group] < UNCERTAIN,
    }


def chemical_check(img: np.ndarray) -> dict:
    """The second check for a picture classified as a chemical diagram: can MolScribe read its atoms and bonds
    reliably? "read" when the reading is confident, holds together as one molecule (or a salt), its bonds are lines
    drawn in the picture and RDKit accepts it as a molecule; otherwise "uncertain", with the reasons. Neither result
    proves that a reading or a redraw matches the original."""
    try:
        from redraw import molscribe_reader, pipeline
    except Exception as exc:
        return {"status": "unavailable", "reasons": [f"MolScribe could not be loaded ({type(exc).__name__})"]}
    if not molscribe_reader.available():
        return {"status": "unavailable", "reasons": [molscribe_reader.unavailable_reason() or "MolScribe is not installed"]}
    picture = img if img.ndim == 3 and img.shape[2] == 3 else (
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else np.ascontiguousarray(img[:, :, :3]))
    if picture.dtype != np.uint8:
        picture = (picture / 257).astype(np.uint8)
    reading = molscribe_reader.read(picture)
    atoms, bonds = reading.get("atoms", []), reading.get("bonds", [])
    confidence = float(reading.get("confidence", 0.0))
    weakest = min((float(a.get("confidence", 0.0)) for a in atoms), default=0.0)
    reasons: list[str] = []
    if len(atoms) < 2 or not bonds:
        reasons.append("MolScribe found no atoms and bonds to read")
    else:
        if confidence < pipeline.MIN_CONFIDENCE:
            reasons.append(f"MolScribe is only {confidence * 100:.0f} % sure of its reading "
                           f"(at least {pipeline.MIN_CONFIDENCE * 100:.0f} % is needed)")
        elif weakest < pipeline.MIN_ATOM_CONFIDENCE:
            reasons.append(f"MolScribe is unsure about an atom ({weakest * 100:.0f} % sure)")
        if pipeline._pieces(reading) > pipeline.MAX_PIECES:
            reasons.append("the reading falls apart into separate pieces")
        labels = pipeline.read_labels(picture)
        drawing = pipeline.check_drawing(picture, reading, [line["box"] for line in labels])
        if "bonds" in drawing and (drawing["bonds"] < pipeline.MIN_DRAWN_BONDS
                                   or drawing["bondsChecked"] < 0.5 * drawing["bonds"]):
            reasons.append("the bonds MolScribe read are not lines drawn in the picture")
    found: dict = {}
    if not reasons:
        mol, problems = pipeline.build_molecule(reading)
        if mol is None:
            reasons.append("the reading is not a valid molecule" + (f" ({problems[0]})" if problems else ""))
        else:
            found = pipeline.identity(mol)
    result = {"status": "uncertain" if reasons else "read", "atoms": len(atoms), "bonds": len(bonds),
              "confidence": round(confidence, 3), "reasons": reasons}
    if found.get("formula"):
        result["formula"] = found["formula"]
    if found.get("inchiKey"):
        result["inchiKey"] = found["inchiKey"]  # to tell whether two readings are the same molecule
    return result


def classify(img: np.ndarray, panels: list[tuple[int, int, int, int]] | None = None,
             check_chemistry: bool = True) -> dict:
    """Classification of one extracted picture and of its panels (boxes x0, y0, x1, y1), with the MolScribe check
    for whatever is classified as chemical."""
    boxes = list(panels or [])
    views = [img] + [img[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
    rows = probabilities(views)
    whole = suggest(rows[0])
    result: dict = {"available": True, "model": MODEL_NAME} | whole
    result["panels"] = []
    if boxes and whole["confidence"] >= WHOLE_FIGURE:
        kind = {TABLE: "the cells of a table", GRAPH: "the boxes of one flowchart or graph",
                CHEMICAL: "parts of one chemical diagram"}.get(whole["category"], "parts of one picture")
        result["panelsNote"] = f"the framed areas are {kind}, not separate figures"
    else:
        panels = [{"panel": number, "box": [int(v) for v in box]} | suggest(row)
                  for number, (box, row) in enumerate(zip(boxes, rows[1:]), start=1)]
        if panels and all(p["category"] == whole["category"] for p in panels):
            # Every framed area is the same kind as the whole picture (the cells of a table): one figure.
            result["panelsNote"] = "the framed areas are parts of one picture of the same kind, not separate figures"
            panels = []
        for panel in panels:
            if check_chemistry and panel["category"] == CHEMICAL:
                x0, y0, x1, y1 = panel["box"]
                panel["chemicalCheck"] = chemical_check(img[y0:y1, x0:x1])
            result["panels"].append(panel)
    if check_chemistry and result["category"] == CHEMICAL and not result["panels"]:
        result["chemicalCheck"] = chemical_check(img)
    return result


def groups(result: dict) -> list[str]:
    """Every group a classified picture belongs to: its own and its panels' (a sheet with a structure and a table is
    found under both)."""
    found = [result.get("category") or OTHER]
    for panel in result.get("panels") or []:
        if panel.get("category") and panel["category"] not in found:
            found.append(panel["category"])
    return found


def panel_category(classification: dict | None, box) -> str | None:
    """The category of the panel at `box` in a classification sent by the app (whose categories already include the
    user's own choices), or the picture's category when the panel is not listed. None: not classified."""
    if not classification:
        return None
    for panel in classification.get("panels") or []:
        if [int(v) for v in panel.get("box", [])] == [int(v) for v in box]:
            return panel.get("category")
    return classification.get("category")
