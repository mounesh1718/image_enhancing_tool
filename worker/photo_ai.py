"""AI clarity for photos: Real-ESRGAN "general" model (BSD-3-Clause, Xintao Wang) run with ONNX Runtime (MIT).

The model restores a photo at 4x resolution (removing noise, compression blocks and blur); the result is reduced
back to exactly the original pixel size, so the photo comes out cleaner and sharper but no larger.

Photos only: the model can invent fine detail (texture, hair), which is harmless in a photo but would be wrong in
text, numbers or chemical structures, so it is never used on those. Every AI-restored photo is flagged for review.

The model is converted once to ONNX by worker/setup_ai.py (needs PyTorch then); at run time only ONNX Runtime is
needed. Nothing is downloaded at run time.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

import ai_runtime

SOURCE_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
SOURCE_FILE = "realesr-general-x4v3.pth"
SOURCE_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"
MODEL_FILE = "realesr-general-x4v3.onnx"
SCALE = 4
TILE = 192          # input tile size; bounds memory (the output tile is 4x larger in each direction)
OVERLAP = 12
# Above this many pixels the photo is not processed (time grows with size; about 1 s per 0.1 MP on a laptop CPU).
MAX_PIXELS = 2_500_000

_lock = threading.Lock()
_session = None
_error: str | None = None


def models_dir() -> Path:
    return ai_runtime.models_base() / "realesrgan"


def model_path() -> Path:
    return models_dir() / MODEL_FILE


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
            try:
                import onnxruntime  # noqa: F401  (reported as missing when not installed)
                if not model_path().exists():
                    raise FileNotFoundError(f"AI photo model not installed at {model_path()} (run worker/setup_ai.py)")
                # Graphics chip when there is one (much faster), otherwise the processor.
                _session = ai_runtime.session(str(model_path()), shape=(1, 3, TILE, TILE))
            except Exception as exc:
                _error = f"{type(exc).__name__}: {exc}"
    return _session


def device_used() -> str | None:
    return ai_runtime.where(_session) if _session is not None else None


def restore(img: np.ndarray) -> np.ndarray:
    """The photo restored at 4x and reduced back to its own size (BGR uint8 in, BGR uint8 out)."""
    global _session
    session = _get()
    if session is None:
        raise RuntimeError(_error or "AI photo model is not available")
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = np.zeros((h * SCALE, w * SCALE, 3), np.float32)
    weight = np.zeros((h * SCALE, w * SCALE, 1), np.float32)
    name = session.get_inputs()[0].name
    step = TILE - 2 * OVERLAP
    for y in range(0, h, step):
        for x in range(0, w, step):
            y0, x0 = max(0, y - OVERLAP), max(0, x - OVERLAP)
            y1, x1 = min(h, y + step + OVERLAP), min(w, x + step + OVERLAP)
            tile = rgb[y0:y1, x0:x1].transpose(2, 0, 1)[None]
            with _lock:
                outputs, session = ai_runtime.run(session, str(model_path()), {name: np.ascontiguousarray(tile)})
                _session = session  # the processor session, if the graphics chip failed
            result = outputs[0][0].transpose(1, 2, 0)
            out[y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] += result
            weight[y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] += 1.0
    restored = np.clip(out / np.maximum(weight, 1e-6), 0, 1)
    small = cv2.resize(restored, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor((small * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- setup (needs PyTorch)

def export_onnx(source: Path, target: Path) -> None:
    """Builds the model (SRVGGNetCompact, as published with Real-ESRGAN), loads its weights strictly and exports it."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class SRVGGNetCompact(nn.Module):
        def __init__(self, num_feat=64, num_conv=32, upscale=4):
            super().__init__()
            self.upscale = upscale
            self.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
            for _ in range(num_conv):
                self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                self.body.append(nn.PReLU(num_parameters=num_feat))
            self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")

    states = torch.load(str(source), map_location="cpu", weights_only=True)
    weights = states.get("params_ema") or states.get("params") or states
    model = SRVGGNetCompact()
    model.load_state_dict(weights, strict=True)
    model.eval()
    example = torch.rand(1, 3, 64, 64)
    torch.onnx.export(model, example, str(target), input_names=["image"], output_names=["restored"], opset_version=17,
                      dynamic_axes={"image": {2: "height", 3: "width"}, "restored": {2: "height", 3: "width"}},
                      dynamo=False)
