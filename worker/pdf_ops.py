"""PDF scanning, image extraction and image replacement with PyMuPDF.

Rules:
- Documents are opened from file paths (MuPDF reads objects lazily), processed page by page, and closed
  at the end of every command. A document object is never shared between threads or commands.
- Images are replaced in place: the image object's stream and size keys change; its object number,
  /SMask, colour space and every placement (position, size, rotation, clipping) stay as they were.
  Vector graphics, text, fonts and untouched images are never rewritten.
- Output goes to a temp file, is validated against the original, and only then renamed into place.
"""

from __future__ import annotations

import gc
import hashlib
import os
import re
import shutil

import cv2
import numpy as np
import pymupdf

import enhance
import image_io
import pdf_content
from protocol import WorkerError, command

MIN_IMAGE_SIDE = 16
FULL_PAGE_FRACTION = 0.85
# ASCII85 and ASCIIHex are lossless text wrappers (ReportLab and other PDF writers store images this way); MuPDF
# decodes the whole filter chain, and a replaced image is always written back as DCT or Flate.
SUPPORTED_FILTERS = {"", "DCTDecode", "FlateDecode", "LZWDecode", "RunLengthDecode", "JPXDecode",
                     "ASCII85Decode", "ASCIIHexDecode"}
LOSSY_FILTERS = {"DCTDecode", "JPXDecode"}
EXTRACT_EXTENSIONS = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "jpx": ".jp2", "jp2": ".jp2"}

# MuPDF keeps objects it has read or modified in memory until the document is closed, so memory grows with
# document size. Long operations close and reopen the document after this much data (measured: 1000 images,
# 257 MB PDF -> replacement peak 322 MB without chunking, 116 MB with it).
DEFAULT_CHUNK_BYTES = 128 * 1024**2


# --------------------------------------------------------------------------- opening

def open_pdf(path: str, password: str | None = None) -> pymupdf.Document:
    if not os.path.isfile(path):
        raise WorkerError("file_not_found", "PDF file does not exist")
    with open(path, "rb") as f:
        if b"%PDF-" not in f.read(1024):
            raise WorkerError("corrupt_pdf", "the file is not a PDF document")
    doc = None
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError):
        pass
    if doc is None:
        # A failed open can leave MuPDF's file handle reachable from the exception; collect it so the
        # user's file is not left locked on Windows. Raised outside the except block for the same reason.
        gc.collect()
        raise WorkerError("corrupt_pdf", "the PDF file is damaged and cannot be opened")
    if not doc.is_pdf or doc.page_count == 0:
        doc.close()
        raise WorkerError("corrupt_pdf", "the PDF file is damaged or has no pages")
    if doc.needs_pass:
        if not password:
            doc.close()
            raise WorkerError("password_required", "the PDF is password-protected")
        if not doc.authenticate(password):
            doc.close()
            raise WorkerError("wrong_password", "the PDF password is incorrect")
    return doc


class ReopeningDocument:
    """A read-only document handle that is closed and reopened after chunk_bytes of data, bounding memory."""

    def __init__(self, path: str, password: str | None, chunk_bytes: int):
        self.path, self.password, self.chunk_bytes = path, password, chunk_bytes
        self.doc = open_pdf(path, password)
        self.used = 0

    def consumed(self, nbytes: int) -> None:
        self.used += nbytes
        if self.used >= self.chunk_bytes:
            self.doc.close()
            # MuPDF's decoded-image cache is process-wide (default 256 MB) and survives closing the document.
            pymupdf.TOOLS.store_shrink(100)
            self.doc = open_pdf(self.path, self.password)
            self.used = 0

    def close(self) -> None:
        self.doc.close()


# --------------------------------------------------------------------------- image properties

def colorspace_components(doc: pymupdf.Document, xref: int) -> int | None:
    """Colour components of an image's colour space, or None for spaces we do not modify
    (Indexed, CMYK, Separation, DeviceN, Lab, Pattern)."""
    kind, value = doc.xref_get_key(xref, "ColorSpace")
    if kind == "xref":
        value = doc.xref_object(int(value.split()[0]), compressed=True)
        kind = "array" if value.startswith("[") else "name"
    if kind == "name":
        return {"/DeviceRGB": 3, "/DeviceGray": 1, "/CalRGB": 3, "/CalGray": 1}.get(value.strip())
    if kind == "array":
        parts = value.strip("[] ").split()
        if not parts:
            return None
        family = parts[0]
        if family == "/ICCBased" and len(parts) >= 2:
            _, n = doc.xref_get_key(int(parts[1]), "N")
            n = int(n) if n.isdigit() else None
            return n if n in (1, 3) else None
        if family in ("/CalRGB",):
            return 3
        if family in ("/CalGray",):
            return 1
    return None


def filter_names(doc: pymupdf.Document, xref: int) -> str:
    """Every filter of an image stream, e.g. "ASCII85Decode DCTDecode" (get_images reports only the first)."""
    kind, value = doc.xref_get_key(xref, "Filter")
    if kind not in ("name", "array"):
        return ""
    return " ".join(re.findall(r"/([A-Za-z0-9]+)", value))


def image_properties(doc: pymupdf.Document, item: tuple) -> dict:
    """Describes one image object and decides whether it may be enhanced (without decoding it)."""
    xref, smask, width, height, bpc, cs_name, _alt, _name, _first_filter, _ref = item
    filters = filter_names(doc, xref)
    props = {
        "xref": xref,
        "width": width,
        "height": height,
        "bitsPerComponent": bpc,
        "colorspace": cs_name,
        "filter": filters,
        "smask": smask,
        "eligible": False,
    }
    is_mask = doc.xref_get_key(xref, "ImageMask")[1] == "true"
    has_decode = doc.xref_get_key(xref, "Decode")[0] not in ("null", "none")
    components = None if is_mask else colorspace_components(doc, xref)
    filter_set = set(filters.split()) if filters else {""}

    if is_mask or bpc == 1 or filter_set & {"JBIG2Decode", "CCITTFaxDecode"}:
        props["skipReason"], props["message"] = "bilevel", "black-and-white image; filtering cannot add detail"
    elif bpc != 8:
        props["skipReason"], props["message"] = "high_bit_depth", f"{bpc}-bit images are kept unchanged"
    elif components is None:
        props["skipReason"], props["message"] = "unsupported_colorspace", f"{cs_name} colour images are kept unchanged"
    elif has_decode:
        props["skipReason"], props["message"] = "custom_decode", "image uses a custom decode array; kept unchanged"
    elif not filter_set <= SUPPORTED_FILTERS:
        props["skipReason"], props["message"] = "unsupported_filter", f"{filters} images are kept unchanged"
    elif min(width, height) < MIN_IMAGE_SIDE:
        props["skipReason"], props["message"] = "too_small", "image is too small to benefit from enhancement"
    else:
        props["eligible"] = True
        props["components"] = components
        # Lossy originals are re-embedded as JPEG, lossless ones as Flate-compressed pixels.
        props["outputFormat"] = "jpeg" if filter_set & LOSSY_FILTERS else "png"
    return props


def resource_key(doc: pymupdf.Document, xref: int) -> str:
    """Content hash of the encoded image, so identical images stored as separate objects are enhanced once."""
    digest = hashlib.sha1(doc.xref_stream_raw(xref) or b"")
    digest.update(f"{doc.xref_get_key(xref, 'Width')[1]}x{doc.xref_get_key(xref, 'Height')[1]}".encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------- commands

@command("pdf_scan")
def pdf_scan(args, ctx):
    """Lists every image placement and every distinct image object, page by page.

    Optional pageStart/pageEnd let the app scan large documents in chunks (progress, cancellation).
    """
    doc = open_pdf(_path(args, "path"), args.get("password"))
    try:
        page_count = doc.page_count
        start = int(args.get("pageStart") or 0)
        end = min(int(args.get("pageEnd") or page_count), page_count)
        occurrences: list[dict] = []
        images: dict[int, dict] = {}
        for index in range(start, end):
            page = doc.load_page(index)
            page_area = abs(page.rect)
            items = {item[0]: item for item in page.get_images(full=True)}
            for placement in pdf_content.find_placements(doc, page):
                xref = placement["xref"]
                rect = pdf_content.placement_rect(page, placement["matrix"])
                a, b, c, d, _, _ = placement["matrix"]
                occurrence = {
                    "page": index,
                    "xref": xref,
                    "rect": [round(v, 3) for v in rect],
                    "matrix": [round(v, 6) for v in placement["matrix"]],
                    "inForm": placement["form"] != 0,
                    "fullPage": page_area > 0 and abs(rect & page.rect) >= FULL_PAGE_FRACTION * page_area,
                }
                if xref == 0:
                    occurrence["skipReason"] = "inline_image"
                elif xref not in images and xref in items:
                    images[xref] = image_properties(doc, items[xref])
                    images[xref]["resourceKey"] = resource_key(doc, xref)
                occurrences.append(occurrence)
            del page
            if (index - start) % 10 == 0 or index == end - 1:
                ctx.progress(stage="scan", page=index + 1, total=page_count)
        return {
            "pageCount": page_count,
            "pageStart": start,
            "pageEnd": end,
            "repaired": bool(doc.is_repaired),
            # Changing any image invalidates a digital signature; the app warns before exporting.
            "signed": doc.get_sigflags() > 0,
            "encrypted": bool(doc.is_encrypted or doc.metadata.get("encryption")),
            "occurrences": occurrences,
            "images": list(images.values()),
        }
    finally:
        doc.close()


@command("pdf_extract")
def pdf_extract(args, ctx):
    """Writes each requested image object to outputDir as x<xref>.<ext>, without re-encoding.

    Soft masks (transparency) stay in the PDF and are not extracted: only colour data is enhanced.
    One image at a time; a failure on one image does not stop the others.
    """
    output_dir = _path(args, "outputDir")
    if not os.path.isdir(output_dir):
        raise WorkerError("bad_request", "outputDir does not exist")
    xrefs = [int(x) for x in args.get("xrefs") or []]
    source = ReopeningDocument(_path(args, "path"), args.get("password"),
                               int(args.get("chunkBytes") or DEFAULT_CHUNK_BYTES))
    results = []
    try:
        for i, xref in enumerate(xrefs):
            try:
                info = source.doc.extract_image(xref)
                if not info or not info.get("image"):
                    raise WorkerError("extract_failed", "image data could not be read")
                ext = EXTRACT_EXTENSIONS.get(info["ext"])
                if ext is None:
                    raise WorkerError("unsupported_format", f"embedded {info['ext']} images are not supported")
                data = info["image"]
                if info["colorspace"] == 1 and ext == ".png":
                    # MuPDF exports grayscale images as RGB PNGs; store them as 1-channel (lossless) so the
                    # enhanced result matches the PDF image's colour space.
                    img = image_io.decode_buffer(data)
                    if img.ndim == 3:
                        data = image_io.encode(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), "png")
                    del img
                target = os.path.join(output_dir, f"x{xref}{ext}")
                image_io.write_atomic(target, data)
                source.consumed(len(info["image"]))
                results.append({
                    "xref": xref, "path": target, "format": info["ext"], "width": info["width"],
                    "height": info["height"], "components": info["colorspace"], "fileSize": len(data),
                    "estimatedMemoryBytes": enhance.estimate_memory(info["width"], info["height"], info["colorspace"]),
                })
            except WorkerError as exc:
                results.append({"xref": xref, "error": {"code": exc.code, "message": exc.message}})
            except (RuntimeError, ValueError) as exc:
                results.append({"xref": xref, "error": {"code": "extract_failed", "message": str(exc)}})
            ctx.progress(stage="extract", done=i + 1, total=len(xrefs))
        return {"images": results}
    finally:
        source.close()


def _replace_one(doc: pymupdf.Document, xref: int, path: str) -> int:
    """Swaps the pixels of one image object, keeping its object number, mask and colour space.

    Returns the number of bytes now held in memory for the new stream.
    """
    if doc.xref_get_key(xref, "Subtype")[1] != "/Image":
        raise WorkerError("replace_failed", "object is not an image")
    expected = colorspace_components(doc, xref)
    data = image_io.read_bytes(path)
    info = image_io.probe(path)

    if info.format == "jpeg":
        components = info.details.get("components")
        if components != expected:
            raise WorkerError("replace_failed", f"enhanced JPEG has {components} channels, the PDF image needs {expected}")
        # A JPEG can be embedded as-is: no decode, no second compression.
        doc.update_stream(xref, data, compress=0)
        doc.xref_set_key(xref, "Filter", "/DCTDecode")
        width, height = info.width, info.height
        added = len(data)
    else:
        img = image_io.decode_buffer(data)
        del data
        channels = 1 if img.ndim == 2 else img.shape[2]
        if channels != expected or img.dtype != np.uint8:
            raise WorkerError("replace_failed", f"enhanced image has {channels} channels, the PDF image needs {expected}")
        if channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        added = img.nbytes
        doc.update_stream(xref, np.ascontiguousarray(img).tobytes(), compress=1)
        del img
        doc.xref_set_key(xref, "Filter", "/FlateDecode")

    doc.xref_set_key(xref, "DecodeParms", "null")
    doc.xref_set_key(xref, "Width", str(width))
    doc.xref_set_key(xref, "Height", str(height))
    doc.xref_set_key(xref, "BitsPerComponent", "8")
    return added


@command("pdf_replace")
def pdf_replace(args, ctx):
    """Builds a new PDF in which the approved images are replaced, then validates it.

    replacements: [{xref, path}]. Images that cannot be replaced are reported and left unchanged.
    The output is written only if validation passes.
    """
    source = _path(args, "source")
    output = _path(args, "output")
    password = args.get("password")
    replacements = args.get("replacements") or []
    if os.path.abspath(source) == os.path.abspath(output):
        raise WorkerError("bad_request", "output must differ from the source PDF")

    tmp = output + ".tmp"
    work = output + ".work"
    chunk_bytes = int(args.get("chunkBytes") or DEFAULT_CHUNK_BYTES)
    replaced: list[int] = []
    failed: list[dict] = []
    doc = None
    try:
        # Work on a copy so pending changes can be flushed with incremental saves and the document reopened,
        # which releases MuPDF's in-memory copies of the new image data.
        try:
            shutil.copyfile(source, work)
        except OSError as exc:
            raise WorkerError(_write_error_code(exc), f"cannot create a working copy: {exc.strerror}") from None
        doc = open_pdf(work, password)
        incremental = doc.can_save_incrementally()
        pending = 0
        for i, entry in enumerate(replacements):
            xref = int(entry["xref"])
            try:
                pending += _replace_one(doc, xref, entry["path"])
                replaced.append(xref)
            except WorkerError as exc:
                failed.append({"xref": xref, "code": exc.code, "message": exc.message})
            if incremental and pending >= chunk_bytes:
                doc.saveIncr()
                doc.close()
                doc = open_pdf(work, password)
                pending = 0
            ctx.progress(stage="replace", done=i + 1, total=len(replacements))
        ctx.progress(stage="save")
        try:
            # Every object is kept with its number, including ones nothing refers to (often left by earlier
            # edits), so the untouched parts stay byte-identical and validation can prove it. Encryption is kept.
            doc.save(tmp, garbage=0, encryption=pymupdf.PDF_ENCRYPT_KEEP)
        except (RuntimeError, ValueError) as exc:
            raise WorkerError("save_failed", f"could not write the PDF: {exc}") from None
    except BaseException:
        _remove(tmp)
        raise
    finally:
        if doc is not None:
            doc.close()
        _remove(work)

    ctx.progress(stage="validate")
    validation = validate(source, tmp, password, replaced, chunk_bytes)
    if not validation["ok"]:
        _remove(tmp)
        raise WorkerError("validation_failed", "; ".join(validation["problems"][:5]))
    os.replace(tmp, output)
    return {"output": output, "replaced": replaced, "failed": failed, "validation": validation,
            "fileSize": os.path.getsize(output)}


def validate(original_path: str, output_path: str, password: str | None, replaced: list[int],
             chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> dict:
    """Checks the output against the original:
    - same page count, page sizes, rotation, drawing instructions, annotations and links;
    - every stream other than the replaced images is byte-identical (text, fonts, vector graphics,
      unapproved images);
    - replaced images keep their mask and colour space, decode, and match their declared size;
    - every page holding a replaced image still renders.
    Both documents are reopened periodically so memory stays bounded on very large files.
    """
    problems: list[str] = []
    replaced_set = set(replaced)
    original = ReopeningDocument(original_path, password, chunk_bytes)
    try:
        output = ReopeningDocument(output_path, password, chunk_bytes)
    except WorkerError as exc:
        original.close()
        return {"ok": False, "problems": [f"output cannot be opened: {exc.message}"]}

    def note(problem: str) -> bool:
        problems.append(problem)
        return len(problems) >= 50  # enough to explain the failure

    try:
        page_count = original.doc.page_count
        pages_to_render = []
        if page_count != output.doc.page_count:
            note(f"page count changed from {page_count} to {output.doc.page_count}")
        else:
            for i in range(page_count):
                a, b = original.doc.load_page(i), output.doc.load_page(i)
                contents = a.read_contents()
                if a.rect != b.rect or a.rotation != b.rotation:
                    note(f"page {i + 1} size or rotation changed")
                # Catches added or removed content streams, not just edited ones.
                if contents != b.read_contents():
                    note(f"page {i + 1} content changed")
                if len(list(a.annots())) != len(list(b.annots())) or len(a.get_links()) != len(b.get_links()):
                    note(f"page {i + 1} annotations or links changed")
                if replaced_set and any(item[0] in replaced_set for item in b.get_images(full=True)):
                    pages_to_render.append(i)
                del a, b
                original.consumed(len(contents))
                output.consumed(len(contents))

        checked = 0
        xref_count = original.doc.xref_length()
        for xref in range(1, xref_count):
            if len(problems) >= 50:
                break
            if xref in replaced_set or not original.doc.xref_is_stream(xref):
                continue
            if original.doc.xref_get_key(xref, "Type")[1] in ("/ObjStm", "/XRef"):
                continue  # file-structure streams are legitimately rebuilt on save
            if not output.doc.xref_is_stream(xref):
                note(f"object {xref} is missing from the output")
                continue
            raw = original.doc.xref_stream_raw(xref) or b""
            if raw != output.doc.xref_stream_raw(xref):
                note(f"object {xref} changed but was not meant to")
            checked += 1
            original.consumed(len(raw))
            output.consumed(len(raw))

        for xref in replaced:
            for key in ("SMask", "ColorSpace", "Subtype"):
                if original.doc.xref_get_key(xref, key) != output.doc.xref_get_key(xref, key):
                    note(f"image {xref}: /{key} changed")
            try:
                info = output.doc.extract_image(xref)
                if (str(info["width"]), str(info["height"])) != (
                        output.doc.xref_get_key(xref, "Width")[1], output.doc.xref_get_key(xref, "Height")[1]):
                    note(f"image {xref}: decoded size does not match its declared size")
                output.consumed(len(info["image"]))
            except (RuntimeError, ValueError, KeyError, TypeError):
                note(f"image {xref}: cannot be decoded")

        pymupdf.TOOLS.mupdf_warnings()  # clear earlier messages
        for i in pages_to_render:
            try:
                output.doc.load_page(i).get_pixmap(matrix=pymupdf.Matrix(0.2, 0.2), alpha=False)
                # Rendering decodes the page's images at full size; count roughly one decoded image per page.
                output.consumed(8 * 1024**2)
            except (RuntimeError, ValueError) as exc:
                note(f"page {i + 1} cannot be rendered: {exc}")
        warnings = pymupdf.TOOLS.mupdf_warnings()
        if warnings:
            note(f"rendering reported problems: {warnings.splitlines()[0]}")

        return {"ok": not problems, "problems": problems, "streamsCompared": checked,
                "pagesRendered": len(pages_to_render)}
    finally:
        original.close()
        output.close()


@command("pdf_validate")
def pdf_validate(args, ctx):
    return validate(_path(args, "original"), _path(args, "output"), args.get("password"),
                    [int(x) for x in args.get("replaced") or []])


# --------------------------------------------------------------------------- helpers

def _path(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise WorkerError("bad_request", f"'{key}' must be a file path")
    return value


def _write_error_code(exc: OSError) -> str:
    return "disk_full" if exc.errno == 28 or getattr(exc, "winerror", None) == 112 else "write_failed"


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
