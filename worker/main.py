"""Entry point of the local image/PDF worker process. See protocol.py for the message format."""

from __future__ import annotations

import os
import platform
import sys

# Hard safety cap on decoded image size, applied by OpenCV itself before allocating pixels. The per-request
# limits in enhance.py are lower; this only catches files whose headers could not be read in advance.
# Must be set before cv2 is imported.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(1 << 28))
# OpenCV warnings (e.g. "PNG input buffer is incomplete") duplicate the errors we report; keep stderr readable.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import enhance  # noqa: E402,F401 - registers image commands
import pdf_ops  # noqa: E402,F401 - registers PDF commands
import sysinfo  # noqa: E402
import textcheck  # noqa: E402
from protocol import ShutdownRequested, command, serve  # noqa: E402


@command("ping")
def ping(args, ctx):
    return {"pong": True, "pid": os.getpid(), "echo": args.get("echo")}


@command("info")
def info(args, ctx):
    """Library versions, so the app can explain a missing dependency before any job starts."""

    def version_of(module_name: str, attr: str = "__version__"):
        try:
            module = __import__(module_name)
            return str(getattr(module, attr, "unknown"))
        except ImportError:
            return None

    return {
        "python": platform.python_version(),
        "numpy": version_of("numpy"),
        "opencv": version_of("cv2"),
        "pymupdf": version_of("pymupdf", "VersionBind"),
        "textCheck": textcheck.available(),
        "redraw": _redraw_available(),
        # The picture classifier (chemical diagrams, flowcharts and graphs, tables, other images).
        "classifier": _classifier_available(),
        "aiPhoto": _photo_ai_available(),
        # AI repair models installed (SCUNet grain, FBCNN JPEG, DocRes text de-blur) and where AI models run.
        "aiRepair": _repair_available(),
        "aiDevice": _ai_device(),
        "cpuCount": sysinfo.cpu_count(),
        "availableMemoryBytes": sysinfo.available_memory(),
    }


def _redraw_available() -> bool:
    try:
        from redraw import molscribe_reader
        return molscribe_reader.model_path().exists()  # loading the model itself is left to the first redraw
    except Exception:
        return False


def _classifier_available() -> bool:
    try:
        import figure_classifier
        return figure_classifier.model_path().exists()
    except Exception:
        return False


def _repair_available() -> dict:
    try:
        import restore_ai
        return {kind: restore_ai.installed(kind) for kind in restore_ai.MODELS}
    except Exception:
        return {}


def _ai_device() -> dict:
    try:
        import ai_runtime
        return ai_runtime.device()
    except Exception:
        return {"kind": "cpu", "name": "processor"}


def _photo_ai_available() -> bool:
    try:
        import photo_ai
        return photo_ai.model_path().exists()
    except Exception:
        return False


@command("shutdown")
def shutdown(args, ctx):
    raise ShutdownRequested()


def main() -> int:
    # The app reads stdout as UTF-8 lines; make that independent of the OS code page.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    return serve()


if __name__ == "__main__":
    sys.exit(main())
