"""One-time setup of the optional AI features (run with the Python the app uses):

    python worker/setup_ai.py                 # everything below
    python worker/setup_ai.py --ocr-only      # text check only (about 30 MB)

- Graphics chips: on Windows the DirectML build of ONNX Runtime (MIT) is installed, so the AI models run on any
  NVIDIA, AMD or Intel graphics chip, and on the processor when there is none.
- AI repair: SCUNet (grain, Apache-2.0), FBCNN (JPEG blocks, Apache-2.0) and DocRes (blurred document text, MIT).
  Downloaded, checked against their SHA-256, converted once to ONNX (Spandrel, MIT, and einops, MIT, are needed for
  that step only) and checked against the original model. About 430 MB on disk.

- AI clarity for photos: Real-ESRGAN (BSD-3-Clause) run with ONNX Runtime (MIT). The 5 MB model is downloaded,
  checked against its SHA-256 and converted once to ONNX (PyTorch and onnx are needed for that step only).

- Picture classification: DocumentFigureClassifier v2.5 (docling-project, MIT; 16.9 MB ONNX model, checked against
  its SHA-256) run with ONNX Runtime. It sorts extracted pictures into chemical diagrams, flowcharts and graphs,
  tables and other images. Installed with every option, --ocr-only included.
- Text check: RapidOCR (Apache-2.0) with the newer PP-OCRv5 models (Apache-2.0, about 13 MB, checked against
  their SHA-256; the package's own PP-OCRv3 models are the fallback).
- Redraw: RDKit (BSD), PyTorch CPU (BSD), timm (Apache-2.0), SmilesPE (Apache-2.0) and MolScribe (MIT). MolScribe's
  package pins libraries that do not exist for current Python, so it and SmilesPE are installed without their
  dependency pins; worker/redraw/molscribe_reader.py runs its model directly.
- The MolScribe model (1.08 GB) is downloaded from Hugging Face, checked against its published SHA-256, reduced to
  the reading weights (366 MB) and the download is deleted. Disk space needed: about 2.5 GB during setup.

After setup everything works offline; nothing is downloaded while the app runs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MOLSCRIBE_SOURCE = "https://github.com/thomas0809/MolScribe/archive/refs/heads/main.zip"


def pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", *args])


def download(url: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 22):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:,.0f} of {total / 2**20:,.0f} MB", end="", flush=True)
    print()
    partial.replace(target)


def installed(distribution: str) -> bool:
    from importlib import metadata
    try:
        metadata.version(distribution)
        return True
    except metadata.PackageNotFoundError:
        return False


def install_ocr5() -> bool:
    """The newer text-reading models (PP-OCRv5 detection and English recognition, about 13 MB), checked by SHA-256."""
    import hashlib
    import textcheck
    for part, (name, url, sha256) in textcheck.OCR5_MODELS.items():
        target = textcheck.ocr5_path(part)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".download")
        download(url, partial)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != sha256:
            partial.unlink()
            print(f"The downloaded {name} does not match its SHA-256; it was deleted. The bundled PP-OCRv3 stays in use.")
            return False
        partial.replace(target)
    print("Text check uses PP-OCRv5.")
    return True


def install_figure_classifier() -> bool:
    """The picture classifier (16.9 MB), checked by SHA-256."""
    import hashlib
    import figure_classifier
    target = figure_classifier.model_path()
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == figure_classifier.SOURCE_SHA256:
        print(f"Picture classifier already installed: {target}")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".download")
    download(figure_classifier.SOURCE_URL, partial)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != figure_classifier.SOURCE_SHA256:
        partial.unlink()
        print("The downloaded picture classifier does not match its SHA-256; it was deleted and not installed.")
        return False
    partial.replace(target)
    print("Picture classifier installed.")
    return True


def ensure_gpu_engine() -> None:
    """Windows: the DirectML build of ONNX Runtime runs the AI models on any graphics chip (NVIDIA, AMD, Intel) and
    still on the processor when there is none. Both builds install into the same folder, so the processor-only build
    (pulled in by other packages) is removed first; uninstalling either one alone would break the other."""
    if sys.platform != "win32":
        return
    if installed("onnxruntime-directml") and not installed("onnxruntime"):
        print("Graphics-chip engine (ONNX Runtime DirectML) already installed.")
        return
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-directml"])
    pip("--force-reinstall", "--no-deps", "onnxruntime-directml")
    pip("onnxruntime-directml")  # its own dependencies, if any are missing
    print("Graphics-chip engine (ONNX Runtime DirectML) installed.")


def install_repair_models(docres_file: str | None) -> bool:
    """SCUNet, FBCNN and DocRes: download, verify, make the model files (processor, graphics chip, half precision;
    see restore_ai), verify each conversion against the original model, then delete the download."""
    import hashlib
    import restore_ai
    pip("spandrel", "einops")
    ok = True
    for kind, spec in restore_ai.MODELS.items():
        wanted = restore_ai.files(kind)
        folder = wanted["flex"].parent
        folder.mkdir(parents=True, exist_ok=True)
        # Earlier versions kept DocRes's flexible file under another name.
        old = folder / "docres_deblur_384.onnx"
        if kind == "deblur_text" and old.exists() and not wanted["flex"].exists():
            old.replace(wanted["flex"])
        missing = [which for which, path in wanted.items() if not path.exists()]
        if not missing:
            print(f"{spec.name} already installed.")
            continue
        source = folder / spec.source_file
        if any(which in ("flex", "fixed") for which in missing):
            if kind == "deblur_text" and docres_file:
                shutil.copyfile(docres_file, source)
            elif not source.exists():
                download(spec.source_url, source)
            if hashlib.sha256(source.read_bytes()).hexdigest() != spec.source_sha256:
                source.unlink()
                print(f"The downloaded {spec.name} model does not match its SHA-256; it was deleted and not installed.")
                ok = False
                continue
        # Full-precision files first: the half-precision ones are made from them.
        for which in sorted(missing, key=lambda w: w.endswith("16")):
            # Each file in its own process: the PyTorch exporter can crash outright on some sizes.
            done = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("restore_ai.py")), "export",
                                   kind, str(source), which], capture_output=True, text=True)
            try:
                difference = float(done.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                difference = None
            if done.returncode != 0 or difference is None or difference > 1.0:
                wanted[which].unlink(missing_ok=True)
                if which == "flex":
                    print(f"{spec.name}: the conversion failed or differs from the original; not installed.")
                    ok = False
                    break
                print(f"{spec.name}: the {which} file could not be made; the model runs without it.")
                continue
            print(f"{spec.name}: {wanted[which].name} made (largest difference {difference:.3f} of 255 levels).")
        source.unlink(missing_ok=True)
    return ok


def choose_devices() -> None:
    """Tries each model once on the graphics chip (if any) and remembers the fastest accurate way to run it, so the
    first job does not wait for it."""
    import ai_runtime
    import photo_ai
    import restore_ai
    if ai_runtime.best_adapter() is None:
        print("No usable graphics chip: the AI models run on the processor.")
        return
    if photo_ai.model_path().exists():
        where = "graphics chip" if ai_runtime.chip_for(str(photo_ai.model_path()), (1, 3, photo_ai.TILE, photo_ai.TILE)) \
            else "processor (faster or safer here)"
        print(f"Real-ESRGAN: runs on the {where}.")
    for kind, spec in restore_ai.MODELS.items():
        if restore_ai.installed(kind):
            chosen = ai_runtime.pick(restore_ai.variants(kind), (1, restore_ai._channels(kind), spec.tile, spec.tile))
            print(f"{spec.name}: runs on the {chosen.label}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ocr-only", action="store_true", help="install the text check only")
    parser.add_argument("--keep-download", action="store_true", help="keep the full 1.08 GB MolScribe checkpoint")
    parser.add_argument("--docres-file", help="an official copy of docres.pkl (from the authors' OneDrive) to use "
                                              "instead of downloading it")
    args = parser.parse_args()

    pip("rapidocr-onnxruntime==1.2.3")
    install_ocr5()
    classifier_ok = install_figure_classifier()
    if args.ocr_only:
        ensure_gpu_engine()
        print("Text check and picture classification installed." if classifier_ok else "Text check installed.")
        return 0 if classifier_ok else 1

    pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    pip("onnx")
    import hashlib
    import photo_ai
    photo_ai.models_dir().mkdir(parents=True, exist_ok=True)
    if not photo_ai.model_path().exists():
        source = photo_ai.models_dir() / photo_ai.SOURCE_FILE
        download(photo_ai.SOURCE_URL, source)
        if hashlib.sha256(source.read_bytes()).hexdigest() != photo_ai.SOURCE_SHA256:
            source.unlink()
            print("The downloaded photo model does not match its SHA-256; it was deleted. Nothing was installed.")
            return 1
        photo_ai.export_onnx(source, photo_ai.model_path())
        source.unlink()
    print("AI clarity for photos installed.")

    repair_ok = install_repair_models(args.docres_file)

    pip("rdkit", "timm")
    pip("--no-deps", "SmilesPE==0.0.3")
    pip("--no-deps", MOLSCRIBE_SOURCE)

    from redraw import molscribe_reader
    folder = molscribe_reader.model_path().parent
    folder.mkdir(parents=True, exist_ok=True)
    if molscribe_reader.model_path().exists():
        print(f"MolScribe weights already installed: {molscribe_reader.model_path()}")
    else:
        if shutil.disk_usage(folder).free < 2 * 1024**3:
            print("Not enough free disk space: about 2 GB is needed for the MolScribe download.")
            return 1
        source = folder / molscribe_reader.SOURCE_FILE
        if not molscribe_reader.verify_source(source):
            download(molscribe_reader.SOURCE_URL, source)
        if not molscribe_reader.verify_source(source):
            source.unlink(missing_ok=True)
            print("The downloaded model does not match its published SHA-256; it was deleted. Nothing was installed.")
            return 1
        molscribe_reader.slim(source, molscribe_reader.model_path())
        if not args.keep_download:
            source.unlink()
    if not molscribe_reader.available():
        print("Redraw could not be loaded: " + (molscribe_reader.unavailable_reason() or "unknown error"))
        return 1
    print("Text check and redraw installed.")
    # Last: other packages may have pulled in the processor-only engine.
    ensure_gpu_engine()
    choose_devices()
    return 0 if repair_ok and classifier_ok else 1


if __name__ == "__main__":
    sys.exit(main())
