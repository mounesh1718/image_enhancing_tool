"""Image file detection, header probing, decoding and encoding with OpenCV only.

- Formats are detected from magic bytes, not file extensions.
- Pixel dimensions are read from headers so size limits apply *before* decoding.
- Files are read/written through NumPy buffers so non-ASCII Windows paths work.
- Outputs are written to a temp file and atomically moved into place.
- JPEG/PNG metadata (EXIF orientation, ICC profile, DPI, text) is copied from the source,
  because OpenCV's encoders drop it.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import cv2
import numpy as np

from protocol import WorkerError

# format -> (OpenCV encode extension, can OpenCV decode it)
RASTER_FORMATS = {
    "jpeg": ".jpg",
    "png": ".png",
    "bmp": ".bmp",
    "tiff": ".tif",
    "webp": ".webp",
    "jp2": ".jp2",
    "gif": ".gif",
    "pnm": ".ppm",
}
VECTOR_FORMATS = {"svg", "emf", "wmf"}
UNSUPPORTED_FORMATS = {"heic", "avif"}

# Formats that keep their own format on output. GIF is palette-limited (256 colours), so an enhanced GIF
# would be re-quantised; it is written as PNG instead.
OUTPUT_FORMAT = {fmt: fmt for fmt in RASTER_FORMATS} | {"gif": "png"}


HEADER_BYTES = 64 * 1024


@dataclass
class ProbeInfo:
    format: str | None
    kind: str  # raster | vector | unsupported | unknown
    width: int | None = None
    height: int | None = None
    frames: int = 1
    bits_per_channel: int = 8
    file_size: int = 0
    # Format-specific details used when writing the output back in the same format.
    details: dict = field(default_factory=dict)

    @property
    def pixels(self) -> int | None:
        return self.width * self.height if self.width and self.height else None

    def to_json(self) -> dict:
        return {
            "format": self.format,
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "bitsPerChannel": self.bits_per_channel,
            "fileSize": self.file_size,
            "outputFormat": OUTPUT_FORMAT.get(self.format) if self.kind == "raster" else None,
        }


# --------------------------------------------------------------------------- detection / probing

def detect_format(head: bytes) -> tuple[str | None, str]:
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg", "raster"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "raster"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif", "raster"
    if head.startswith(b"BM"):
        return "bmp", "raster"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff", "raster"
    if head[:4] in (b"II+\x00", b"MM\x00+"):
        return "tiff", "raster"  # BigTIFF: decodable, header not parsed
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp", "raster"
    if head.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n") or head.startswith(b"\xff\x4f\xff\x51"):
        return "jp2", "raster"
    if len(head) >= 2 and head[0:1] == b"P" and head[1:2] in b"123456":
        return "pnm", "raster"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"avif", b"avis"):
            return "avif", "unsupported"
        if brand in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1"):
            return "heic", "unsupported"
    if head[40:44] == b" EMF":
        return "emf", "vector"
    if head.startswith(b"\xd7\xcd\xc6\x9a") or head.startswith(b"\x01\x00\x09\x00"):
        return "wmf", "vector"
    text = head[:2048].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if text.startswith(b"<svg") or (text.startswith(b"<?xml") and b"<svg" in text):
        return "svg", "vector"
    return None, "unknown"


def probe(path: str) -> ProbeInfo:
    """Reads only the file header. Never decodes pixel data."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(HEADER_BYTES)
    except FileNotFoundError:
        raise WorkerError("file_not_found", "input file does not exist") from None
    except OSError as exc:
        raise WorkerError("read_failed", f"cannot read input file: {exc.strerror}") from None

    fmt, kind = detect_format(head)
    info = ProbeInfo(format=fmt, kind=kind, file_size=size)
    if kind != "raster":
        return info

    try:
        parser = _HEADER_PARSERS.get(fmt)
        if parser is not None:
            parser(head, info, path)
    except (struct.error, IndexError, ValueError):
        pass  # Unparseable header: dimensions stay unknown; decoding will decide.
    return info


def _probe_png(head: bytes, info: ProbeInfo, path: str) -> None:
    info.width, info.height = struct.unpack(">II", head[16:24])
    info.bits_per_channel = max(head[24], 8)
    info.details["colorType"] = head[25]


def _probe_gif(head: bytes, info: ProbeInfo, path: str) -> None:
    info.width, info.height = struct.unpack("<HH", head[6:10])


def _probe_bmp(head: bytes, info: ProbeInfo, path: str) -> None:
    w, h = struct.unpack("<ii", head[18:26])
    info.width, info.height = abs(w), abs(h)


# JPEG Annex K luminance table (natural order) and the zigzag order tables are stored in.
_STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                      14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                      49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.float32)
_ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14,
           21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60,
           61, 54, 47, 55, 62, 63]


def _jpeg_quality(segment: bytes) -> int | None:
    """Estimated save quality (1-100, libjpeg scale) from the first luminance quantisation table of a DQT segment."""
    pos = 0
    while pos < len(segment):
        precision, table_id = segment[pos] >> 4, segment[pos] & 15
        size = 128 if precision else 64
        values = np.frombuffer(segment[pos + 1:pos + 1 + size], dtype=">u2" if precision else np.uint8).astype(np.float32)
        pos += 1 + size
        if table_id != 0 or values.size != 64:
            continue
        natural = np.empty(64, np.float32)
        natural[_ZIGZAG] = values
        scale = float(np.median(natural * 100.0 / _STD_LUMA))
        quality = 5000.0 / scale if scale > 100 else (200.0 - scale) / 2.0
        return int(round(min(100.0, max(1.0, quality))))
    return None


def _probe_jpeg(head: bytes, info: ProbeInfo, path: str) -> None:
    # Walk markers until a start-of-frame. APP segments can exceed the header buffer, so read from the file.
    with open(path, "rb") as f:
        f.seek(2)
        while True:
            marker = f.read(2)
            if len(marker) < 2 or marker[0] != 0xFF:
                return
            code = marker[1]
            if code == 0xFF:
                f.seek(-1, os.SEEK_CUR)
                continue
            if code in (0xD8, 0x01) or 0xD0 <= code <= 0xD7:
                continue
            length = struct.unpack(">H", f.read(2))[0]
            if code == 0xDB:  # quantisation tables: how strongly the image was compressed
                quality = _jpeg_quality(f.read(length - 2))
                if quality is not None and "quality" not in info.details:
                    info.details["quality"] = quality
                continue
            if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                precision, h, w, components = struct.unpack(">BHHB", f.read(6))
                info.width, info.height = w, h
                info.bits_per_channel = precision
                info.details["components"] = components
                return
            f.seek(length - 2, os.SEEK_CUR)


def _probe_webp(head: bytes, info: ProbeInfo, path: str) -> None:
    chunk = head[12:16]
    if chunk == b"VP8 ":
        w, h = struct.unpack("<HH", head[26:30])
        info.width, info.height = w & 0x3FFF, h & 0x3FFF
        info.details["lossless"] = False
    elif chunk == b"VP8L":
        bits = struct.unpack("<I", head[21:25])[0]
        info.width, info.height = (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        info.details["lossless"] = True
    elif chunk == b"VP8X":
        flags = head[20]
        info.width = int.from_bytes(head[24:27], "little") + 1
        info.height = int.from_bytes(head[27:30], "little") + 1
        info.details["lossless"] = b"VP8L" in head
        info.details["hasMetadata"] = bool(flags & 0x2C)  # ICC, EXIF, XMP
        if flags & 0x02:
            info.frames = 2  # animated; exact count not needed
            info.details["animated"] = True


def _probe_tiff(head: bytes, info: ProbeInfo, path: str) -> None:
    endian = "<" if head[:2] == b"II" else ">"
    if struct.unpack(endian + "H", head[2:4])[0] != 42:
        return  # BigTIFF
    type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}

    with open(path, "rb") as f:
        def read_at(offset: int, n: int) -> bytes:
            f.seek(offset)
            data = f.read(n)
            if len(data) != n:
                raise ValueError("truncated TIFF")
            return data

        offset = struct.unpack(endian + "I", head[4:8])[0]
        frames = 0
        seen = set()
        while offset and offset not in seen and frames < 10000:
            seen.add(offset)
            count = struct.unpack(endian + "H", read_at(offset, 2))[0]
            entries = read_at(offset + 2, count * 12)
            if frames == 0:
                for i in range(count):
                    tag, typ, n = struct.unpack(endian + "HHI", entries[i * 12:i * 12 + 8])
                    raw = entries[i * 12 + 8:i * 12 + 12]
                    value = _tiff_value(endian, typ, n, raw, type_sizes, read_at)
                    if tag == 256:
                        info.width = int(value)
                    elif tag == 257:
                        info.height = int(value)
                    elif tag == 258:
                        info.bits_per_channel = int(value)
                    elif tag == 259:
                        info.details["compression"] = int(value)
                    elif tag == 282:
                        info.details["xResolution"] = value
                    elif tag == 283:
                        info.details["yResolution"] = value
                    elif tag == 296:
                        info.details["resolutionUnit"] = int(value)
            frames += 1
            offset = struct.unpack(endian + "I", read_at(offset + 2 + count * 12, 4))[0]
        info.frames = max(frames, 1)


def _tiff_value(endian, typ, n, raw, type_sizes, read_at):
    """First value of a TIFF tag (enough for the scalar tags we need)."""
    size = type_sizes.get(typ, 1) * n
    data = raw if size <= 4 else read_at(struct.unpack(endian + "I", raw)[0], type_sizes.get(typ, 1))
    if typ == 3:
        return struct.unpack(endian + "H", data[:2])[0]
    if typ == 4:
        return struct.unpack(endian + "I", data[:4])[0]
    if typ == 5:
        num, den = struct.unpack(endian + "II", data[:8])
        return num / den if den else 0.0
    return data[0]


def _probe_jp2(head: bytes, info: ProbeInfo, path: str) -> None:
    i = head.find(b"ihdr")
    if i >= 0:
        info.height, info.width = struct.unpack(">II", head[i + 4:i + 12])
        return
    i = head.find(b"\xff\x51")  # raw codestream SIZ marker
    if i >= 0:
        xsiz, ysiz, xo, yo = struct.unpack(">IIII", head[i + 6:i + 22])
        info.width, info.height = xsiz - xo, ysiz - yo


def _probe_pnm(head: bytes, info: ProbeInfo, path: str) -> None:
    tokens: list[bytes] = []
    for line in head[2:512].splitlines():
        line = line.split(b"#", 1)[0]
        tokens.extend(line.split())
        if len(tokens) >= 3:
            break
    info.width, info.height = int(tokens[0]), int(tokens[1])
    if head[1:2] in b"14":
        info.details["bilevel"] = True


_HEADER_PARSERS = {
    "png": _probe_png, "gif": _probe_gif, "bmp": _probe_bmp, "jpeg": _probe_jpeg,
    "webp": _probe_webp, "tiff": _probe_tiff, "jp2": _probe_jp2, "pnm": _probe_pnm,
}


# --------------------------------------------------------------------------- decode / encode

def read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        raise WorkerError("file_not_found", "input file does not exist") from None
    except OSError as exc:
        raise WorkerError("read_failed", f"cannot read input file: {exc.strerror}") from None


def decode_buffer(data: bytes, frame: int = 0) -> np.ndarray:
    """Decodes one image (or one frame of a multi-frame file) keeping alpha and bit depth.

    EXIF orientation is NOT applied: pixels stay as stored so the copied EXIF tag stays correct.
    Only the requested frame is decoded.
    """
    buf = np.frombuffer(data, dtype=np.uint8)  # no copy
    try:
        if frame:
            ok, mats = cv2.imdecodemulti(buf, cv2.IMREAD_UNCHANGED, range=(frame, frame + 1))
            img = mats[0] if ok and mats else None
        else:
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except cv2.error:
        img = None
    if img is None or img.size == 0:
        raise WorkerError("decode_failed", "the image data is corrupted or not a supported image")
    return img


def decode(path: str, frame: int = 0) -> np.ndarray:
    return decode_buffer(read_bytes(path), frame)


def encode(img: np.ndarray, fmt: str, probe_info: ProbeInfo | None = None) -> bytes:
    ext = RASTER_FORMATS[fmt]
    params: list[int] = []
    details = probe_info.details if probe_info and probe_info.format == fmt else {}

    if fmt == "jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif fmt == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif fmt == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 101 if details.get("lossless") else 95]
    elif fmt == "tiff":
        compression = {1: 1, 5: 5, 8: 8, 32946: 8}.get(details.get("compression"), 5)
        params = [cv2.IMWRITE_TIFF_COMPRESSION, compression]
        if details.get("xResolution"):
            params += [
                cv2.IMWRITE_TIFF_RESUNIT, details.get("resolutionUnit", 2),
                cv2.IMWRITE_TIFF_XDPI, int(round(details["xResolution"])),
                cv2.IMWRITE_TIFF_YDPI, int(round(details.get("yResolution", details["xResolution"]))),
            ]
    elif fmt == "pnm":
        ext = ".pgm" if img.ndim == 2 else ".ppm"
    elif fmt == "jp2":
        params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000]

    try:
        ok, out = cv2.imencode(ext, img, params)
    except cv2.error as exc:
        raise WorkerError("encode_failed", f"could not encode {fmt}: {str(exc).splitlines()[-1]}") from None
    if not ok:
        raise WorkerError("encode_failed", f"could not encode {fmt}")
    return out.tobytes()


def encode_multipage_tiff(frames: list[np.ndarray], probe_info: ProbeInfo | None) -> bytes:
    compression = {1: 1, 5: 5, 8: 8, 32946: 8}.get((probe_info.details if probe_info else {}).get("compression"), 5)
    ok, out = cv2.imencodemulti(".tif", frames, [cv2.IMWRITE_TIFF_COMPRESSION, compression])
    if not ok:
        raise WorkerError("encode_failed", "could not encode multi-page TIFF")
    return out.tobytes()


def write_atomic(path: str, data: bytes) -> None:
    """Writes to path.tmp then renames, so a crash never leaves a half-written file under the real name."""
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise WorkerError("write_failed", "output directory does not exist")
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        try:
            os.remove(tmp)
        except OSError:
            pass
        code = "disk_full" if getattr(exc, "errno", None) == 28 or getattr(exc, "winerror", None) == 112 else "write_failed"
        raise WorkerError(code, f"cannot write output file: {exc.strerror}") from None


# --------------------------------------------------------------------------- metadata

def _jpeg_header_segments(data: bytes) -> list[tuple[int, bytes]]:
    """(marker, full segment bytes) for segments before the first SOF/SOS/DQT/DHT."""
    segments = []
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        code = data[i + 1]
        if not (0xE0 <= code <= 0xEF or code == 0xFE):
            break
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        segments.append((code, data[i:i + 2 + length]))
        i += 2 + length
    return segments


def copy_jpeg_metadata(source: bytes, encoded: bytes, converted_from_cmyk: bool) -> bytes:
    """Replaces the encoder's APPn segments with the source's EXIF/XMP/ICC/JFIF/comment segments."""
    keep = []
    for code, seg in _jpeg_header_segments(source):
        if code == 0xEE:  # Adobe APP14 describes the source colour transform; not valid for the new data.
            continue
        if code == 0xE2 and converted_from_cmyk:  # A CMYK ICC profile does not describe RGB output.
            continue
        keep.append(seg)
    if not keep:
        return encoded
    encoded_meta_len = sum(len(seg) for _, seg in _jpeg_header_segments(encoded))
    return b"\xff\xd8" + b"".join(keep) + encoded[2 + encoded_meta_len:]


_PNG_COPY_CHUNKS = {b"iCCP", b"sRGB", b"gAMA", b"cHRM", b"pHYs", b"eXIf", b"tEXt", b"zTXt", b"iTXt"}


def _png_chunks(data: bytes):
    i = 8
    while i + 8 <= len(data):
        length = struct.unpack(">I", data[i:i + 4])[0]
        ctype = data[i + 4:i + 8]
        yield ctype, data[i:i + 12 + length]
        i += 12 + length
        if ctype == b"IEND":
            return


def copy_png_metadata(source: bytes, encoded: bytes) -> bytes:
    """Inserts the source's colour/DPI/text chunks right after the encoder's IHDR."""
    keep = [chunk for ctype, chunk in _png_chunks(source) if ctype in _PNG_COPY_CHUNKS]
    if not keep:
        return encoded
    existing = {ctype for ctype, _ in _png_chunks(encoded)}
    keep = [chunk for chunk in keep if chunk[4:8] not in existing]
    ihdr_end = 8 + 12 + struct.unpack(">I", encoded[8:12])[0]
    return encoded[:ihdr_end] + b"".join(keep) + encoded[ihdr_end:]


def jpeg_orientation(data: bytes) -> int:
    """EXIF orientation (1-8) of a JPEG, 1 if absent. Used only to orient previews."""
    for code, seg in _jpeg_header_segments(data[:HEADER_BYTES * 4]):
        if code != 0xE1 or seg[4:10] != b"Exif\x00\x00":
            continue
        tiff = seg[10:]
        endian = "<" if tiff[:2] == b"II" else ">"
        try:
            ifd = struct.unpack(endian + "I", tiff[4:8])[0]
            count = struct.unpack(endian + "H", tiff[ifd:ifd + 2])[0]
            for k in range(count):
                entry = tiff[ifd + 2 + k * 12: ifd + 14 + k * 12]
                if struct.unpack(endian + "H", entry[:2])[0] == 0x0112:
                    value = struct.unpack(endian + "H", entry[8:10])[0]
                    return value if 1 <= value <= 8 else 1
        except struct.error:
            return 1
    return 1


def apply_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# --------------------------------------------------------------------------- previews

def to_preview_bgr(img: np.ndarray) -> np.ndarray:
    """8-bit 3-channel copy for display: 16-bit scaled down, alpha composited on white."""
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        out = img[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return out.astype(np.uint8)
    return img


def write_thumbnail(img: np.ndarray, path: str, max_side: int, orientation: int = 1) -> dict:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = img
    if scale < 1.0:
        small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    small = apply_orientation(to_preview_bgr(small), orientation)
    ok, out = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise WorkerError("encode_failed", "could not encode thumbnail")
    write_atomic(path, out.tobytes())
    return {"path": path, "width": small.shape[1], "height": small.shape[0]}
