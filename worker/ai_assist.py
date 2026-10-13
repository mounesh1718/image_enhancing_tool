"""Optional AI enhancement, tried by the user after the normal (OpenCV) processing.

Suggestions come from the picture's measurements, its group (see figure_classifier.py) and the installed models:

- SCUNet (grain, Apache-2.0) when real grain is measured;
- FBCNN (JPEG blocks, Apache-2.0) for JPEGs saved at low quality;
- DocRes (blurred document text, MIT) for blurred text and diagrams; its letters are checked against the original;
- Real-ESRGAN (BSD-3-Clause) for photos only (the "other images" group with photo content): never for text,
  tables, graphs or chemical diagrams, and only when none of the faithful repairs above applies.

A suggestion says a model *may* help; nothing is promised. The AI version is made from the original (the AI step,
then the remaining normal steps) and is checked like any result: the text check (no number or more than a few
letters may read differently), detail preservation (it may not lose or add much more detail than the OpenCV result)
and, for chemical diagrams, chemical safety (the structure must read as the same molecule before and after). A
version failing a check is reported as rejected, with the reason. The OpenCV result is never replaced: the user
chooses which version to keep.
"""
from __future__ import annotations

import numpy as np

import analyse
import enhance
import figure_classifier
import photo_ai
import restore_ai
import textcheck

MODEL_OF = {"grain": "SCUNet", "jpeg": "FBCNN", "deblur_text": "DocRes", "photo": "Real-ESRGAN"}
OP_MODEL = {"ai_grain": "SCUNet", "ai_jpeg": "FBCNN", "ai_deblur_text": "DocRes", "ai_photo": "Real-ESRGAN"}
SUGGESTION = {
    "grain": "Grain detected — SCUNet may help.",
    "jpeg": "Strong JPEG compression detected — FBCNN may help.",
    "deblur_text": "Blurred text detected — DocRes may help (every letter it shows is checked against the original).",
    "photo": "Soft or noisy photo — Real-ESRGAN may make it look clearer (it can add fine texture, so compare closely).",
}
NONE_SUITABLE = "No suitable AI repair detected."


def _installed(kind: str) -> bool:
    try:
        return photo_ai.model_path().exists() if kind == "photo" else restore_ai.installed(kind)
    except Exception:
        return False


def suggestions(analysis: dict, category: str | None, has_panels: bool = False) -> list[dict]:
    """What AI could try on a picture, from its measurements and group: [{"kind", "model", "text", "installed"}].
    Empty when no model suits it (clean pictures, pure black and white, sheets of several figures)."""
    if analysis.get("bilevel") or analysis.get("dtype", "uint8") != "uint8" or has_panels:
        return []
    pixels = int(analysis.get("width", 0)) * int(analysis.get("height", 0))
    kinds: list[str] = []
    if pixels <= enhance.AI_REPAIR_MAX_PIXELS:
        quality = analysis.get("jpegQuality")
        if quality is not None and quality <= enhance.JPEG_MAX_QUALITY:
            kinds.append("jpeg")
        if analysis.get("noise", 0.0) > enhance.GRAIN_MIN_NOISE and analysis.get("noiseRatio", 1.0) < enhance.MAX_NOISE_RATIO:
            kinds.append("grain")
        if analysis.get("content") == "text_or_diagram" and analysis.get("blur", 0.0) >= enhance.DOC_DEBLUR_MIN_BLUR:
            kinds.append("deblur_text")
    # The photo model invents fine texture: photos only (never text, tables, graphs or chemical diagrams), and only
    # when no faithful repair applies.
    if not kinds and category == figure_classifier.OTHER and analysis.get("content") == "photo" \
            and pixels <= photo_ai.MAX_PIXELS \
            and (analysis.get("blur", 0.0) >= enhance.MIN_DEBLUR or analysis.get("sharpness", 1e9) < 600
                 or (analysis.get("noise", 0.0) > 2.0 and analysis.get("noiseRatio", 1.0) < enhance.MAX_NOISE_RATIO)):
        kinds.append("photo")
    return [{"kind": k, "model": MODEL_OF[k], "text": SUGGESTION[k], "installed": _installed(k)} for k in kinds]


def _candidates(img: np.ndarray, analysis: dict, settings: dict, kinds: set[str]) -> list[dict]:
    """The AI versions: the AI step on the original, then the remaining normal steps."""
    found: list[dict] = []
    pre = ([{"op": "ai_jpeg"}] if "jpeg" in kinds else []) + ([{"op": "ai_grain"}] if "grain" in kinds else [])
    cleaned = img
    if pre:
        cleaned = enhance.apply(img, pre)
        rest = enhance.plan(enhance._reanalyse(cleaned, analysis), settings)
        found.append({"name": "ai_clean", "ops": pre + rest, "image": enhance.apply(cleaned, rest) if rest else cleaned})
    if "deblur_text" in kinds:
        sharp = enhance.apply(cleaned, [{"op": "ai_deblur_text"}])
        rest = enhance.plan(enhance._reanalyse(sharp, analysis), settings | {"sharpen": False})
        found.append({"name": "ai_deblur", "ops": pre + [{"op": "ai_deblur_text"}] + rest,
                      "image": enhance.apply(sharp, rest) if rest else sharp})
    if "photo" in kinds:
        ops = enhance.plan(analysis | {"aiPhoto": True}, settings)
        if not any(o["op"] == "ai_photo" for o in ops):
            ops = [{"op": "ai_photo"}]
        found.append({"name": "ai_photo", "ops": ops, "image": enhance.apply(img, ops)})
    return found


def _chemical_safety(original: np.ndarray, result: np.ndarray) -> str | None:
    """None when the structure reads as the same molecule before and after; otherwise why the AI version is unsafe."""
    before = figure_classifier.chemical_check(original)
    if before.get("status") != "read" or not before.get("inchiKey"):
        return ("on a chemical diagram an AI change is only kept when the structure can be read before and after, "
                "and this one cannot be read reliably")
    after = figure_classifier.chemical_check(result)
    if after.get("status") != "read" or after.get("inchiKey") != before["inchiKey"]:
        return "the chemical structure reads differently after the AI change"
    return None


def try_ai(img: np.ndarray, analysis: dict, settings: dict, classification: dict | None,
           opencv: np.ndarray | None) -> dict:
    """Makes and checks the AI version of one picture. Returns {"status": "accepted" | "rejected" | "not_eligible" |
    "unavailable", "image": the AI version (accepted or rejected), "models", "operations", "reasons", "checks",
    "suggestions"}."""
    category = (classification or {}).get("category")
    has_panels = bool((classification or {}).get("panels"))
    suggested = suggestions(analysis, category, has_panels)
    if not suggested:
        return {"status": "not_eligible", "reasons": [NONE_SUITABLE], "suggestions": []}
    usable = [s for s in suggested if s["installed"]]
    if not usable:
        return {"status": "unavailable", "suggestions": suggested,
                "reasons": [f"{s['model']} is not installed (run worker/setup_ai.py), so it could not be tried." for s in suggested]}

    settings = enhance._merged(enhance.DEFAULT_SETTINGS, settings) | {"mode": "auto", "aiRepair": False, "aiPhoto": False}
    analysis = dict(analysis)
    analysis.setdefault("upscaleAllowed", True)
    analysis.setdefault("autoUpscale", 1.0)
    candidates = _candidates(img, analysis, settings, {s["kind"] for s in usable})
    baseline_ops = enhance.plan(analysis, settings)
    if opencv is not None and opencv.shape == img.shape:
        baseline = {"name": "filters" if baseline_ops else "original", "ops": baseline_ops, "image": opencv}
    else:
        baseline = {"name": "filters", "ops": baseline_ops, "image": enhance.apply(img, baseline_ops)} if baseline_ops \
            else {"name": "original", "ops": [], "image": img}

    # The existing checks between versions: text (numbers, changed letters, unverified letters for DocRes) and
    # detail preservation against the OpenCV result.
    _, info = enhance._choose(img, analysis, [baseline] + candidates, settings, {})
    entries = {e["name"]: e for e in info.get("candidates", [])}
    valid = [c for c in candidates if "rejected" not in entries.get(c["name"], {})]
    reasons = [f"{_models(c)}: {entries[c['name']]['rejected']}" for c in candidates if "rejected" in entries.get(c["name"], {})]
    if valid:
        textual = "legibility" in entries.get(valid[0]["name"], {})
        best = max(valid, key=lambda c: entries[c["name"]].get("legibility", 0.0)) if textual \
            else min(valid, key=lambda c: entries[c["name"]].get("noise", 0.0))
    else:
        best = candidates[0]
    checks: dict = {"versions": info.get("candidates", [])}

    status = "accepted" if valid else "rejected"
    if status == "accepted":
        text = enhance._text_check(img, best["image"], analysis, settings)
        checks["text"] = text
        if text.get("flag"):
            status = "rejected"
            reasons = [f"{_models(best)}: the text reader reads some letters or numbers differently from the original"]
    if status == "accepted" and (category == figure_classifier.CHEMICAL
                                 or any(p.get("category") == figure_classifier.CHEMICAL for p in (classification or {}).get("panels") or [])):
        problem = _chemical_safety(img, best["image"])
        checks["chemical"] = problem or "the structure reads as the same molecule before and after"
        if problem:
            status = "rejected"
            reasons = [f"{_models(best)}: {problem}"]
    checks["detail"] = analyse.compare(img, best["image"], analysis.get("noise", 0.0))
    return {"status": status, "image": best["image"], "models": _model_list(best), "operations": enhance.flatten(best["ops"]),
            "reasons": reasons if status == "rejected" else [], "checks": checks, "suggestions": suggested}


def _model_list(candidate: dict) -> list[str]:
    return [OP_MODEL[o["op"]] for o in enhance.flatten(candidate["ops"]) if o["op"] in OP_MODEL]


def _models(candidate: dict) -> str:
    return " + ".join(_model_list(candidate)) or "AI"


def available_models() -> dict:
    """Which AI models are installed (for the app's explanations)."""
    return {MODEL_OF[k]: _installed(k) for k in MODEL_OF}


def text_reader_available() -> bool:
    return textcheck.available()
