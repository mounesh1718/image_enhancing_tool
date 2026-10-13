import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import image_io  # noqa: E402
import samples  # noqa: E402
from protocol import WorkerError  # noqa: E402


def png_chunk(ctype: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)


def jpeg_with_orientation(img: np.ndarray, orientation: int) -> bytes:
    """A JPEG carrying a minimal EXIF APP1 segment with the given orientation tag."""
    ok, buf = cv2.imencode(".jpg", img)
    data = buf.tobytes()
    tiff = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHIHH", 0x0112, 3, 1, orientation, 0) + struct.pack("<I", 0)
    payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return data[:2] + app1 + data[2:]


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.img = samples.photo(120, 200)

    def tearDown(self):
        self.tmp.cleanup()

    def test_detects_format_and_size_for_every_supported_raster(self):
        cases = {
            "a.jpg": "jpeg", "a.png": "png", "a.bmp": "bmp", "a.tif": "tiff", "a.webp": "webp",
            "a.jp2": "jp2", "a.gif": "gif", "a.ppm": "pnm",
        }
        for name, fmt in cases.items():
            with self.subTest(name):
                path = samples.write(self.dir / name, self.img)
                info = image_io.probe(path)
                self.assertEqual((info.format, info.kind), (fmt, "raster"))
                self.assertEqual((info.width, info.height), (200, 120))

    def test_detection_ignores_misleading_extension(self):
        path = samples.write(self.dir / "really_png.jpg", self.img, ext=".png")
        self.assertEqual(image_io.probe(path).format, "png")

    def test_webp_lossless_flag(self):
        lossless = samples.write(self.dir / "l.webp", self.img, params=[cv2.IMWRITE_WEBP_QUALITY, 101])
        lossy = samples.write(self.dir / "q.webp", self.img, params=[cv2.IMWRITE_WEBP_QUALITY, 80])
        self.assertTrue(image_io.probe(lossless).details["lossless"])
        self.assertFalse(image_io.probe(lossy).details["lossless"])

    def test_multipage_tiff_frame_count(self):
        ok, buf = cv2.imencodemulti(".tif", [self.img, self.img[:50, :60], self.img[:10, :10]])
        path = self.dir / "multi.tif"
        buf.tofile(str(path))
        info = image_io.probe(str(path))
        self.assertEqual(info.frames, 3)
        self.assertEqual(image_io.decode(str(path), 1).shape[:2], (50, 60))

    def test_vector_and_unsupported_formats(self):
        files = {
            "a.svg": (b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"></svg>', ("svg", "vector")),
            "a.heic": (b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic", ("heic", "unsupported")),
            "a.wmf": (b"\xd7\xcd\xc6\x9a" + b"\x00" * 60, ("wmf", "vector")),
            "a.emf": (b"\x01\x00\x00\x00" + b"\x00" * 36 + b" EMF" + b"\x00" * 20, ("emf", "vector")),
            "a.txt": (b"hello world", (None, "unknown")),
        }
        for name, (data, expected) in files.items():
            with self.subTest(name):
                (self.dir / name).write_bytes(data)
                info = image_io.probe(str(self.dir / name))
                self.assertEqual((info.format, info.kind), expected)

    def test_missing_file(self):
        with self.assertRaises(WorkerError) as ctx:
            image_io.probe(str(self.dir / "nope.png"))
        self.assertEqual(ctx.exception.code, "file_not_found")


class DecodeEncodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_non_ascii_path_round_trip(self):
        folder = self.dir / "Rapport_é_化学"
        folder.mkdir()
        img = samples.photo(64, 64)
        path = str(folder / "图像.png")
        image_io.write_atomic(path, image_io.encode(img, "png"))
        self.assertTrue(np.array_equal(image_io.decode(path), img))

    def test_corrupt_image_raises_decode_failed(self):
        ok, buf = cv2.imencode(".png", samples.photo(64, 64))
        data = buf.tobytes()
        path = self.dir / "broken.png"
        path.write_bytes(data[:60] + b"\x00" * 200)
        with self.assertRaises(WorkerError) as ctx:
            image_io.decode(str(path))
        self.assertEqual(ctx.exception.code, "decode_failed")

    def test_alpha_and_16bit_kept_on_decode(self):
        rgba = np.dstack([samples.photo(32, 32), np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)])
        self.assertEqual(image_io.decode(samples.write(self.dir / "a.png", rgba)).shape, (32, 32, 4))
        deep = (np.random.default_rng(0).random((16, 16)) * 65535).astype(np.uint16)
        self.assertEqual(image_io.decode(samples.write(self.dir / "d.tif", deep)).dtype, np.uint16)

    def test_write_atomic_leaves_no_temp_file(self):
        target = self.dir / "out.bin"
        image_io.write_atomic(str(target), b"abc")
        self.assertEqual(target.read_bytes(), b"abc")
        self.assertEqual(os.listdir(self.dir), ["out.bin"])

    def test_write_atomic_into_missing_directory_fails_cleanly(self):
        with self.assertRaises(WorkerError):
            image_io.write_atomic(str(self.dir / "missing" / "x.png"), b"abc")


class MetadataTests(unittest.TestCase):
    def test_jpeg_exif_is_copied_and_orientation_read(self):
        img = samples.photo(40, 60)
        source = jpeg_with_orientation(img, 6)
        self.assertEqual(image_io.jpeg_orientation(source), 6)

        encoded = image_io.encode(img, "jpeg")
        self.assertEqual(image_io.jpeg_orientation(encoded), 1)
        merged = image_io.copy_jpeg_metadata(source, encoded, converted_from_cmyk=False)
        self.assertEqual(image_io.jpeg_orientation(merged), 6)
        self.assertEqual(cv2.imdecode(np.frombuffer(merged, np.uint8), cv2.IMREAD_UNCHANGED).shape, (40, 60, 3))

    def test_png_dpi_and_text_chunks_are_copied(self):
        img = samples.photo(20, 30)
        plain = image_io.encode(img, "png")
        ihdr_end = 8 + 12 + 13
        phys = png_chunk(b"pHYs", struct.pack(">IIB", 11811, 11811, 1))  # 300 DPI
        text = png_chunk(b"tEXt", b"Comment\x00scan")
        source = plain[:ihdr_end] + phys + text + plain[ihdr_end:]

        merged = image_io.copy_png_metadata(source, image_io.encode(img, "png"))
        self.assertIn(phys, merged)
        self.assertIn(text, merged)
        self.assertTrue(np.array_equal(cv2.imdecode(np.frombuffer(merged, np.uint8), cv2.IMREAD_UNCHANGED), img))

    def test_orientation_rotates_preview(self):
        img = np.zeros((10, 20, 3), np.uint8)
        self.assertEqual(image_io.apply_orientation(img, 6).shape[:2], (20, 10))
        self.assertEqual(image_io.apply_orientation(img, 1).shape[:2], (10, 20))


if __name__ == "__main__":
    unittest.main()
