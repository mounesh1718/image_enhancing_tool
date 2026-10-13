"""Runs the ONNX AI models on the fastest device in the computer.

- A graphics chip (NVIDIA, AMD or Intel) through DirectML when the DirectML build of ONNX Runtime is installed
  (`onnxruntime-directml`, Windows 10/11). Measured on a laptop: an RTX 3050 ran the photo model 12x faster than the
  processor, the built-in Radeon chip 5x faster, with the same result (largest difference 0.004 of 255 levels).
- Otherwise the processor. Every model also works there, only slower.

Windows lists the graphics chips in no particular order (often the slow built-in chip first), so the chip with the most
memory of its own is chosen. A chip that fails is never fatal: the model falls back to the processor.

Each model is tried once on the chip in a separate process before it is used there (a graphics driver bug can crash a
process outright, which no error handling can catch) and timed against the processor; the faster device is remembered
for that model, chip and engine version in models/device_choice.json. Measured on a laptop's built-in Radeon chip:
FBCNN and DocRes ran faster than on the processor, SCUNet slower, so SCUNet stays on the processor there.
Set IQE_AI_DEVICE=cpu to always use the processor.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_lock = threading.Lock()
_adapters: list[dict] | None = None
_gpu_failed: str | None = None


def adapters() -> list[dict]:
    """Hardware graphics chips in DirectML's device order: [{"index", "name", "memoryGB"}]. Empty off Windows."""
    global _adapters
    if _adapters is None:
        try:
            _adapters = _enumerate_dxgi() if sys.platform == "win32" else []
        except Exception:
            _adapters = []
    return _adapters


def _enumerate_dxgi() -> list[dict]:
    from ctypes import POINTER, byref, c_void_p

    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_ulong), ("b", ctypes.c_ushort), ("c", ctypes.c_ushort), ("d", ctypes.c_ubyte * 8)]

    class DESC1(ctypes.Structure):
        _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                    ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                    ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                    ("Luid", ctypes.c_int64), ("Flags", ctypes.c_uint)]

    iid = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    factory = c_void_p()
    if ctypes.WinDLL("dxgi").CreateDXGIFactory1(byref(iid), byref(factory)) != 0:
        return []
    vtable = ctypes.cast(factory, POINTER(POINTER(c_void_p))).contents
    enum_adapters1 = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, ctypes.c_uint, POINTER(c_void_p))(vtable[12])
    found, index = [], 0
    while True:
        adapter = c_void_p()
        if enum_adapters1(factory, index, byref(adapter)) != 0:
            break
        avt = ctypes.cast(adapter, POINTER(POINTER(c_void_p))).contents
        desc = DESC1()
        ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(DESC1))(avt[10])(adapter, byref(desc))
        ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(avt[2])(adapter)  # Release
        if not desc.Flags & 2:  # skip the software renderer
            found.append({"index": index, "name": desc.Description.strip(),
                          "memoryGB": round(desc.DedicatedVideoMemory / 2**30, 1)})
        index += 1
    ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(vtable[2])(factory)
    return found


def best_adapter() -> dict | None:
    """The graphics chip to use, or None when the models run on the processor."""
    if os.environ.get("IQE_AI_DEVICE", "").lower() == "cpu" or _gpu_failed:
        return None
    try:
        import onnxruntime
        if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
            return None
    except ImportError:
        return None
    chips = adapters()
    return max(chips, key=lambda a: a["memoryGB"]) if chips else None


def models_base() -> Path:
    """Folder holding every AI model (one sub-folder per model)."""
    base = os.environ.get("IQE_MODELS") or os.path.join(os.environ.get("LOCALAPPDATA", str(Path.home())),
                                                        "ImageQualityEnhancer", "models")
    return Path(base)


_PROBE = r"""
import sys, time, numpy as np, onnxruntime as ort
path, shape, device = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3])
o = ort.SessionOptions(); o.enable_mem_pattern = False; o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
s = ort.InferenceSession(path, o, providers=[("DmlExecutionProvider", {"device_id": device})])
if "DmlExecutionProvider" not in s.get_providers(): sys.exit(3)  # quietly fell back to the processor
x = {s.get_inputs()[0].name: np.random.default_rng(0).random(shape, dtype=np.float32)}
y = s.run(None, x)[0]; best = 1e9
for _ in range(3):
    t = time.perf_counter(); s.run(None, x); best = min(best, time.perf_counter() - t)
print(best)
if len(sys.argv) > 4: np.save(sys.argv[4], y)
"""


@dataclass(frozen=True)
class Variant:
    """One way to run a model: a model file on the graphics chip or on the processor."""
    label: str   # e.g. "graphics chip, half precision" / "processor"
    path: str
    gpu: bool


def _file_key(path: str) -> str:
    try:
        stat = os.stat(path)
        return f"{Path(path).name}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        return f"{Path(path).name}:missing"


def _cpu_run(path: str, shape: tuple):
    """(seconds for one run, output) on the processor, with a fixed random input."""
    import numpy as np
    import onnxruntime
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
    sess = onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    feeds = {sess.get_inputs()[0].name: np.random.default_rng(0).random(shape, dtype=np.float32)}
    out = sess.run(None, feeds)[0]
    best = float("inf")
    for _ in range(2):  # the best of several runs: one slow run (a busy or power-saving moment) must not decide
        start = time.perf_counter()
        sess.run(None, feeds)
        best = min(best, time.perf_counter() - start)
    return best, out


def _probe(path: str, shape: tuple, chip: dict):
    """(seconds, output) on the graphics chip in a separate process (the best of three runs, as laptop chips slow down
    for moments to save power), or (None, None) if it failed."""
    import numpy as np
    handle, out_file = tempfile.mkstemp(suffix=".npy")
    os.close(handle)
    try:
        done = subprocess.run([sys.executable, "-B", "-c", _PROBE, path, ",".join(map(str, shape)), str(chip["index"]),
                               out_file], capture_output=True, text=True, timeout=600,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if done.returncode != 0:
            return None, None
        return float(done.stdout.strip().splitlines()[-1]), np.load(out_file)
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None, None
    finally:
        try:
            os.remove(out_file)
        except OSError:
            pass


def pick(variants: list[Variant], shape: tuple, tolerance: float = 1.0) -> Variant:
    """The fastest accurate way to run a model on this computer, among `variants` (the last processor variant is
    the reference). Each graphics-chip variant is tried once in its own process, so a driver crash cannot reach the
    app; its result must match the processor's within `tolerance` (0-255 levels, which allows half precision only
    where it is visually identical) and it must be clearly faster (time per pixel, at least 10 %). The choice is
    remembered per model files, chip and engine version."""
    cpu = [v for v in variants if not v.gpu][-1]
    chip = best_adapter()
    gpus = [v for v in variants if v.gpu and os.path.exists(v.path)] if chip else []
    if not gpus:
        return cpu
    import onnxruntime
    key = "|".join(_file_key(v.path) for v in variants) + f"|{chip['name']}|{onnxruntime.__version__}|{shape}"
    remembered = _load_choices().get(key)
    for v in variants:
        if v.label == remembered:
            return v
    import numpy as np
    cpu_seconds, reference = _cpu_run(cpu.path, shape)
    best, best_seconds = cpu, 0.9 * cpu_seconds
    for v in gpus:
        seconds, out = _probe(v.path, shape, chip)
        if seconds is None or out is None or out.shape != reference.shape \
                or float(np.abs(out - reference).max()) * 255.0 > tolerance:
            continue
        if seconds < best_seconds:
            best, best_seconds = v, seconds
    _save_choice(key, best.label)
    return best


def open_session(variant: Variant, threads: int | None = None):
    """An ONNX Runtime session for a variant chosen by `pick`."""
    import onnxruntime
    options = onnxruntime.SessionOptions()
    chip = best_adapter() if variant.gpu else None
    if chip:
        options.enable_mem_pattern = False
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        return onnxruntime.InferenceSession(variant.path, options, providers=[
            ("DmlExecutionProvider", {"device_id": chip["index"]}), "CPUExecutionProvider"])
    options.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2) // 2)
    return onnxruntime.InferenceSession(variant.path, options, providers=["CPUExecutionProvider"])




def _load_choices() -> dict:
    try:
        return json.loads((models_base() / "device_choice.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_choice(key: str, value: str) -> None:
    path = models_base() / "device_choice.json"
    with _lock:
        choices = _load_choices()
        choices[key] = value
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(choices, indent=1), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass  # not remembered: the trial simply runs again next time




def chip_for(model_path: str, shape: tuple) -> dict | None:
    """The graphics chip for a single-file model, or None to use the processor (crashing, failing, inaccurate or
    slower on the chip)."""
    chosen = pick([Variant("graphics chip", model_path, True), Variant("processor", model_path, False)], shape)
    return best_adapter() if chosen.gpu else None


def device() -> dict:
    """Where the AI models run, for the app and the report: {"kind": "gpu"|"cpu", "name": ...}."""
    chip = best_adapter()
    if chip:
        return {"kind": "gpu", "name": chip["name"]}
    info = {"kind": "cpu", "name": "processor"}
    if _gpu_failed:
        info["gpuError"] = _gpu_failed
    return info


def session(model_path: str, threads: int | None = None, shape: tuple | None = None):
    """An ONNX Runtime session on the best device for this model. With `shape` (the model's input shape) the model is
    first tried and timed on the graphics chip (see chip_for). Inputs then go through `run` (below)."""
    import onnxruntime
    options = onnxruntime.SessionOptions()
    chip = chip_for(model_path, shape) if shape else best_adapter()
    if chip:
        # DirectML needs sequential execution without memory patterns (ONNX Runtime documentation).
        options.enable_mem_pattern = False
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        try:
            return onnxruntime.InferenceSession(model_path, options, providers=[
                ("DmlExecutionProvider", {"device_id": chip["index"]}), "CPUExecutionProvider"])
        except Exception as exc:  # driver or chip problem: use the processor instead
            _mark_gpu_failed(exc)
            options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2) // 2)
    return onnxruntime.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])


def on_gpu(sess) -> bool:
    return "DmlExecutionProvider" in sess.get_providers()


def where(sess) -> str:
    """Where a session runs, for the report: the graphics chip's name or "processor"."""
    chip = best_adapter() if on_gpu(sess) else None
    return chip["name"] if chip else "processor"


def run(sess, model_path: str, feeds: dict, threads: int | None = None):
    """Runs the model; if the graphics chip fails, retries on the processor. Returns (outputs, session to keep)."""
    try:
        return sess.run(None, feeds), sess
    except Exception as exc:
        if not on_gpu(sess):
            raise
        _mark_gpu_failed(exc)
        cpu = session(model_path, threads)
        return cpu.run(None, feeds), cpu


def _mark_gpu_failed(exc: Exception) -> None:
    global _gpu_failed
    with _lock:
        _gpu_failed = f"{type(exc).__name__}: {str(exc)[:200]}"
