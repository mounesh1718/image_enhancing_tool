"""AI repair models that restore an image at its own size (nothing is enlarged):

- "grain": SCUNet real-image denoiser (Apache-2.0, Kai Zhang et al.). Removes scan and camera grain.
- "jpeg":  FBCNN (Apache-2.0, Jiaxi Jiang et al.). Removes JPEG blocks and ringing; estimates the quality itself.
- "deblur_text": DocRes deblurring task (MIT, Jiaxin Zhang et al.). Sharpens blurred document text.

Measured against the existing filters (characters read correctly by OCR on synthetic text with known content):
heavy grain 95.1 % -> 100 %, severe JPEG 98.8 % -> 100 %, blur sigma 2.5 92.7 % -> 97.6 %, sigma 3.0 26.8 % -> 91.5 %.
On a page blurred beyond recognition DocRes produced sharp but WRONG digits, so enhance.py only keeps its result when
the text check proves every letter and digit against the original (see enhance._choose).

Speed. Each model is kept in up to three ONNX files, made once by worker/setup_ai.py:
- "_flex": any size; used on the processor with tiles sized to the image (a small picture is one tile; a 1000x700
  picture takes 6 tiles of 384 instead of 20 of 256: measured 1.6x faster for SCUNet);
- "_<tile>": a fixed tile size, for graphics chips (drivers crashed on SCUNet at variable sizes, on an AMD and an
  NVIDIA chip alike);
- "_fp16": the same in half precision, for graphics chips (SCUNet on an RTX 3050: 3.5x faster, at most 0.6 of 255
  levels different; FBCNN was slower in half precision there).
ai_runtime.pick tries each graphics-chip file once per computer and keeps the fastest one whose result matches the
processor's. Tiles overlap and only their inner part is kept, so there are no seams. With `regions`, only tiles
touching those areas are run (DocRes runs on the text lines only). Nothing is downloaded while the app runs.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import ai_runtime


@dataclass(frozen=True)
class Spec:
    folder: str
    stem: str
    tile: int        # tile size for graphics chips (fixed-size file)
    overlap: int
    multiple: int    # sides of a flexible model's input must be a multiple of this
    fixed: bool      # whether a fixed-size file can be made (the PyTorch exporter crashed on DocRes at 384)
    source_url: str
    source_file: str
    source_sha256: str
    name: str        # for the report


MODELS = {
    "grain": Spec("scunet", "scunet_color_real_psnr", 256, 16, 64, True,
                  "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
                  "scunet_color_real_psnr.pth",
                  "fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e", "SCUNet"),
    "jpeg": Spec("fbcnn", "fbcnn_color", 256, 16, 8, True,
                 "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth",
                 "fbcnn_color.pth",
                 "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8", "FBCNN"),
    # The authors publish the weights on OneDrive, which now needs a sign-in; this is the identical re-hosted file
    # (pinned by SHA-256). An official copy can be given to setup_ai.py with --docres-file.
    "deblur_text": Spec("docres", "docres_deblur", 384, 32, 8, False,
                        "https://huggingface.co/presencesw/DocRes/resolve/main/docres.pkl",
                        "docres.pkl",
                        "1d6a89d754fe1e58ffd1865eab0ef3f03344798d39197b2d9a77ce4fbc8c02fd", "DocRes"),
}

# Processor tiles: sizes tried (pixels per side) and the fixed cost of one tile, in pixel equivalents.
CPU_TILES = range(128, 513, 64)
TILE_OVERHEAD = 20_000

_lock = threading.Lock()
_state: dict[str, dict] = {}


def files(kind: str) -> dict[str, Path]:
    """The model's files: "flex" (any size, processor), "fixed" and "fixed16" or "flex16" (graphics chips)."""
    spec = MODELS[kind]
    base = ai_runtime.models_base() / spec.folder
    found = {"flex": base / f"{spec.stem}_flex.onnx"}
    if spec.fixed:
        found["fixed"] = base / f"{spec.stem}_{spec.tile}.onnx"
        found["fixed16"] = base / f"{spec.stem}_{spec.tile}_fp16.onnx"
    else:
        found["flex16"] = base / f"{spec.stem}_flex_fp16.onnx"
    return found


def model_path(kind: str) -> Path:
    """The model file the processor uses (always present when the model is installed)."""
    return files(kind)["flex"]


def installed(kind: str) -> bool:
    return model_path(kind).exists()


def available(kind: str) -> bool:
    if not installed(kind):
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _channels(kind: str) -> int:
    return 6 if kind == "deblur_text" else 3


def variants(kind: str) -> list[ai_runtime.Variant]:
    """The ways this model can run here; the processor (reference) is last."""
    found = files(kind)
    options = []
    for key, label in (("fixed16", "graphics chip, half precision"), ("fixed", "graphics chip"),
                       ("flex16", "graphics chip, half precision"), ("flex", "graphics chip, any size")):
        if key in found and found[key].exists():
            options.append(ai_runtime.Variant(label, str(found[key]), True))
    options.append(ai_runtime.Variant("processor", str(found["flex"]), False))
    return options


def _runner(kind: str) -> dict:
    with _lock:
        if kind not in _state:
            spec = MODELS[kind]
            variant = ai_runtime.pick(variants(kind), (1, _channels(kind), spec.tile, spec.tile))
            _state[kind] = {"variant": variant, "session": ai_runtime.open_session(variant)}
        return _state[kind]


def _fall_back(kind: str, exc: Exception) -> dict:
    """The graphics chip failed: this model runs on the processor from now on."""
    ai_runtime._mark_gpu_failed(exc)
    cpu = variants(kind)[-1]
    with _lock:
        _state[kind] = {"variant": cpu, "session": ai_runtime.open_session(cpu)}
        return _state[kind]


def devices_used() -> dict:
    """{model name: where it ran} for the models used so far in this worker."""
    with _lock:
        return {MODELS[k].name: ai_runtime.where(s["session"]) for k, s in _state.items()}


def _deblur_prompt(bgr: np.ndarray) -> np.ndarray:
    """DocRes's deblurring hint: the image's edge strength (as in the authors' inference code)."""
    x = cv2.convertScaleAbs(cv2.Sobel(bgr, cv2.CV_16S, 1, 0))
    y = cv2.convertScaleAbs(cv2.Sobel(bgr, cv2.CV_16S, 0, 1))
    edges = cv2.cvtColor(cv2.addWeighted(x, 0.5, y, 0.5, 0), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def run(kind: str, img: np.ndarray, regions: list | None = None) -> np.ndarray:
    """Repairs an 8-bit image (grey or BGR) with one model; same size and channels out. With `regions` (boxes
    x0, y0, x1, y1) only the tiles touching them are repaired; the rest is returned unchanged."""
    if img.dtype != np.uint8:
        raise ValueError("AI repair needs an 8-bit image")
    gray = img.ndim == 2
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if gray else img
    if kind == "deblur_text":
        planes = np.concatenate([bgr, _deblur_prompt(bgr)], axis=2)  # DocRes works in BGR order
    else:
        planes = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = _tiled(kind, planes.astype(np.float32) / 255.0, regions)
    result = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if kind != "deblur_text":
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if gray else result


def tile_size(kind: str, gpu: bool, h: int, w: int) -> int:
    """Graphics chips: the fixed tile. Processor: the tile size with the least work for this image (fewest padded
    pixels plus a fixed cost per tile)."""
    spec = MODELS[kind]
    if gpu:
        return spec.tile
    def cost(t: int) -> float:
        step = t - 2 * spec.overlap
        return math.ceil(h / step) * math.ceil(w / step) * (t * t + TILE_OVERHEAD)
    sizes = [t for t in CPU_TILES if t % spec.multiple == 0 and t > 2 * spec.overlap]
    return min(sizes, key=cost)


def _touches(regions, y0: int, x0: int, y1: int, x1: int) -> bool:
    return any(bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0 for bx0, by0, bx1, by1 in regions)


def _tiled(kind: str, planes: np.ndarray, regions) -> np.ndarray:
    spec = MODELS[kind]
    state = _runner(kind)
    h, w = planes.shape[:2]
    tile, overlap = tile_size(kind, state["variant"].gpu, h, w), spec.overlap
    step = tile - 2 * overlap
    full_h, full_w = -(-h // step) * step, -(-w // step) * step
    border = cv2.BORDER_REFLECT_101 if min(h, w) > overlap + 1 else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(planes, overlap, full_h - h + overlap, overlap, full_w - w + overlap, border)
    out = np.ascontiguousarray(padded[overlap:overlap + full_h, overlap:overlap + full_w, :3]).copy()
    for y in range(0, full_h, step):
        for x in range(0, full_w, step):
            if regions is not None and not _touches(regions, y, x, y + step, x + step):
                continue  # nothing to repair here: the tile keeps its own pixels
            chunk = np.ascontiguousarray(padded[y:y + tile, x:x + tile].transpose(2, 0, 1)[None])
            sess = state["session"]
            try:
                result = sess.run(None, {sess.get_inputs()[0].name: chunk})[0]
            except Exception as exc:
                if not state["variant"].gpu:
                    raise
                state = _fall_back(kind, exc)
                return _tiled(kind, planes, regions)  # again, with the processor's tiles
            out[y:y + step, x:x + step] = result[0].transpose(1, 2, 0)[overlap:overlap + step, overlap:overlap + step]
    return out[:h, :w]


# --------------------------------------------------------------------------- setup (needs PyTorch)

def _network(kind: str, source: Path):
    """The model built from its published weights (strict loading), returning only the restored image."""
    import torch
    if kind == "deblur_text":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "docres"))
        import restormer_arch
        net = restormer_arch.Restormer(inp_channels=6, out_channels=3, dim=48, num_blocks=[2, 3, 3, 4],
                                       num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                                       bias=False, LayerNorm_type="WithBias", dual_pixel_task=True)
        state = torch.load(str(source), map_location="cpu", weights_only=True)["model_state"]
        net.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state.items()}, strict=True)
    else:
        from spandrel import ModelLoader
        state = torch.load(str(source), map_location="cpu", weights_only=True)
        net = ModelLoader().load_from_state_dict(state.get("params", state)).model

    class FirstOutput(torch.nn.Module):  # FBCNN also returns its quality estimate
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            r = self.inner(x)
            return r[0] if isinstance(r, (tuple, list)) else r

    return FirstOutput(net).eval()


def export_onnx(kind: str, source: Path, which: str) -> float:
    """Makes one model file ("flex", "fixed", "fixed16" or "flex16") and returns the largest difference (0-255 levels)
    between it and the original model on a test tile. Run in a separate process (setup_ai does): the PyTorch exporter
    can crash outright. Half precision is made from the matching full-precision file and checked later against the
    processor by ai_runtime.pick."""
    spec = MODELS[kind]
    target = files(kind)[which]
    target.parent.mkdir(parents=True, exist_ok=True)
    if which in ("fixed16", "flex16"):
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
        full = files(kind)["fixed" if which == "fixed16" else "flex"]
        onnx.save(convert_float_to_float16(onnx.load(str(full)), keep_io_types=True), str(target))
        return 0.0
    import onnxruntime
    import torch
    model = _network(kind, source)
    if which == "flex":
        # Traced small with flexible sides (tracing at the tile size crashed the exporter for DocRes).
        torch.onnx.export(model, torch.rand(1, _channels(kind), 128, 128), str(target), input_names=["image"],
                          output_names=["restored"], opset_version=17, dynamo=False,
                          dynamic_axes={"image": {2: "height", 3: "width"}, "restored": {2: "height", 3: "width"}})
    else:
        torch.onnx.export(model, torch.rand(1, _channels(kind), spec.tile, spec.tile), str(target),
                          input_names=["image"], output_names=["restored"], opset_version=17, dynamo=False)
    example = torch.rand(1, _channels(kind), spec.tile, spec.tile, generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        expected = model(example).numpy()
    got = onnxruntime.InferenceSession(str(target), providers=["CPUExecutionProvider"]).run(
        None, {"image": example.numpy()})[0]
    return float(np.abs(got - expected).max() * 255.0)


if __name__ == "__main__":  # python restore_ai.py export <kind> <source> <which>  (used by setup_ai.py)
    import sys
    if len(sys.argv) == 5 and sys.argv[1] == "export":
        print(export_onnx(sys.argv[2], Path(sys.argv[3]), sys.argv[4]))
