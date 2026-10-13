import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import enhance  # noqa: E402
import image_io  # noqa: E402
import pdf_content  # noqa: E402
import pdf_ops  # noqa: E402
import pdf_samples  # noqa: E402
import samples  # noqa: E402
from protocol import WorkerError  # noqa: E402


class Ctx:
    def __init__(self):
        self.events = []

    def progress(self, **data):
        self.events.append(data)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TokenizerTests(unittest.TestCase):
    def ops(self, data: bytes):
        return [(k, v) for k, v in pdf_content.tokens(data) if k in ("op", "inline", "name")]

    def test_strings_and_comments_do_not_produce_operators(self):
        data = b"(not an operator: Do q Q (nested) \\) still string) Tj % comment Do\n/Im1 Do"
        self.assertEqual(self.ops(data), [("op", b"Tj"), ("name", "Im1"), ("op", b"Do")])

    def test_name_escapes(self):
        self.assertEqual(self.ops(b"/My#20Image Do")[0], ("name", "My Image"))

    def test_inline_image_data_containing_EI(self):
        data = b"BI /W 2 /H 2 /CS /G /BPC 8 ID \x00EIx\xff EI Q"
        kinds = [k if k != "op" else v for k, v in self.ops(data)]
        self.assertEqual(kinds, [b"BI", "name", "name", "name", "name", "name", "inline", b"Q"])

    def test_matrix_multiply(self):
        scale = (2, 0, 0, 3, 0, 0)
        move = (1, 0, 0, 1, 10, 20)
        self.assertEqual(pdf_content.multiply(scale, move), (2, 0, 0, 3, 10, 20))
        self.assertEqual(pdf_content.multiply(move, scale), (2, 0, 0, 3, 20, 60))


class PdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.pdf = str(cls.dir / "feature.pdf")
        cls.x = pdf_samples.build_feature_pdf(cls.pdf)
        cls.scan = pdf_ops.pdf_scan({"path": cls.pdf}, Ctx())
        cls.images = {i["xref"]: i for i in cls.scan["images"]}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def occurrences(self, xref):
        return [o for o in self.scan["occurrences"] if o["xref"] == xref]

    # ---- scanning

    def test_scan_finds_every_placement_with_correct_position(self):
        self.assertEqual(self.scan["pageCount"], 5)
        self.assertEqual(self.occurrences(self.x["photo"])[0]["rect"], [72, 320, 312, 500])
        # Reused object: two placements, listed once.
        self.assertEqual([o["page"] for o in self.occurrences(self.x["photo"])], [0, 1])
        self.assertEqual(len([i for i in self.scan["images"] if i["xref"] == self.x["photo"]]), 1)

    def test_scan_positions_agree_with_pymupdf_for_unambiguous_images(self):
        doc = pymupdf.open(self.pdf)
        for xref in (self.x["table_image"], self.x["alpha"], self.x["gray"], self.x["scan"]):
            occ = self.occurrences(xref)[0]
            expected = doc[occ["page"]].get_image_rects(xref)[0]
            self.assertTrue(np.allclose(occ["rect"], list(expected), atol=0.01), (xref, occ["rect"], expected))
        doc.close()

    def test_scan_special_cases(self):
        rotated = self.occurrences(self.x["rotated"])[0]
        self.assertEqual(rotated["matrix"][0], 0.0)  # 90 degree rotation
        self.assertTrue(self.occurrences(self.x["scan"])[0]["fullPage"])
        self.assertTrue(any(o["inForm"] for o in self.scan["occurrences"]))
        self.assertTrue(any(o.get("skipReason") == "inline_image" for o in self.scan["occurrences"]))
        self.assertEqual(self.images[self.x["alpha"]]["smask"] > 0, True)
        self.assertEqual(self.images[self.x["gray"]]["components"], 1)
        self.assertEqual(self.images[self.x["cmyk"]]["skipReason"], "unsupported_colorspace")
        self.assertEqual(self.images[self.x["icon"]]["skipReason"], "too_small")
        self.assertEqual(self.images[self.x["photo"]]["outputFormat"], "jpeg")
        self.assertEqual(self.images[self.x["table_image"]]["outputFormat"], "png")

    def test_identical_images_in_separate_objects_share_resource_key(self):
        self.assertEqual(self.images[self.x["photo"]]["resourceKey"], self.images[self.x["duplicate"]]["resourceKey"])
        self.assertNotEqual(self.images[self.x["photo"]]["resourceKey"], self.images[self.x["rotated"]]["resourceKey"])

    def test_chunked_scan_matches_full_scan(self):
        parts = [pdf_ops.pdf_scan({"path": self.pdf, "pageStart": s, "pageEnd": s + 2}, Ctx()) for s in (0, 2, 4)]
        self.assertEqual(sum((p["occurrences"] for p in parts), []), self.scan["occurrences"])

    # ---- extraction

    def test_extract_writes_jpeg_streams_unchanged(self):
        out = self.dir / "extract1"
        out.mkdir()
        result = pdf_ops.pdf_extract({"path": self.pdf, "xrefs": [self.x["photo"], self.x["alpha"]], "outputDir": str(out)}, Ctx())
        by_xref = {r["xref"]: r for r in result["images"]}
        doc = pymupdf.open(self.pdf)
        self.assertEqual(Path(by_xref[self.x["photo"]]["path"]).read_bytes(), doc.xref_stream_raw(self.x["photo"]))
        doc.close()
        # Transparent image: only the colour data is extracted; the mask stays in the PDF.
        self.assertEqual(image_io.decode(by_xref[self.x["alpha"]]["path"]).shape, (150, 200, 3))

    def test_extract_keeps_lossless_images_pixel_identical(self):
        """Pictures are classified and enhanced from what extraction gives: a Flate (PNG) image must come out with
        exactly the PDF's pixels, at its own size (no resampling, no re-compression loss)."""
        out = self.dir / "extract_lossless"
        out.mkdir()
        xrefs = [self.x["gray"], self.x["table_image"]]
        result = pdf_ops.pdf_extract({"path": self.pdf, "xrefs": xrefs, "outputDir": str(out)}, Ctx())
        doc = pymupdf.open(self.pdf)
        for item in result["images"]:
            pix = pymupdf.Pixmap(doc, item["xref"])
            stored = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            extracted = image_io.decode(item["path"])
            extracted = extracted if extracted.ndim == 3 else extracted[:, :, None]
            if pix.n == 3:
                extracted = extracted[:, :, ::-1]  # BGR -> RGB
            self.assertEqual(extracted.shape, stored.shape, item["xref"])
            self.assertTrue(np.array_equal(extracted, stored), item["xref"])
        doc.close()

    def test_extract_reports_per_image_failures(self):
        out = self.dir / "extract2"
        out.mkdir()
        result = pdf_ops.pdf_extract({"path": self.pdf, "xrefs": [999999, self.x["gray"]], "outputDir": str(out)}, Ctx())
        self.assertIn("error", result["images"][0])
        self.assertIn("path", result["images"][1])

    # ---- full pipeline

    def run_pipeline(self, name: str, approve: list[int]) -> tuple[str, dict, dict]:
        work = self.dir / name
        (work / "extracted").mkdir(parents=True)
        (work / "enhanced").mkdir()
        extracted = pdf_ops.pdf_extract({"path": self.pdf, "xrefs": approve, "outputDir": str(work / "extracted")}, Ctx())
        replacements = []
        for item in extracted["images"]:
            fmt = self.images[item["xref"]]["outputFormat"]
            target = work / "enhanced" / f"x{item['xref']}{'.jpg' if fmt == 'jpeg' else '.png'}"
            # Force a visible change so replacement is observable even for already-good images.
            result = enhance.enhance_cmd({"input": item["path"], "output": str(target), "format": fmt,
                                          "settings": {"mode": "manual", "sharpenAmount": 0.8}}, None)
            self.assertEqual(result["status"], "completed", result)
            replacements.append({"xref": item["xref"], "path": str(target)})
        output = str(work / "out.pdf")
        result = pdf_ops.pdf_replace({"source": self.pdf, "output": output, "replacements": replacements}, Ctx())
        return output, result, {r["xref"]: r for r in replacements}

    def test_replace_changes_only_approved_images_and_preserves_layout(self):
        before = sha256(self.pdf)
        approve = [self.x["photo"], self.x["alpha"], self.x["gray"], self.x["table_image"], self.x["scan"], self.x["rotated"]]
        output, result, _ = self.run_pipeline("full", approve)

        self.assertEqual(sha256(self.pdf), before, "original must never change")
        self.assertEqual(sorted(result["replaced"]), sorted(approve))
        self.assertEqual(result["failed"], [])
        self.assertTrue(result["validation"]["ok"])

        a, b = pymupdf.open(self.pdf), pymupdf.open(output)
        try:
            for i in range(a.page_count):
                pa, pb = a[i], b[i]
                self.assertEqual(pa.get_text(), pb.get_text())
                self.assertEqual(len(pa.get_drawings()), len(pb.get_drawings()))
                self.assertEqual([(p["xref"], p["matrix"]) for p in pdf_content.find_placements(a, pa)],
                                 [(p["xref"], p["matrix"]) for p in pdf_content.find_placements(b, pb)])
            for xref in approve:
                self.assertNotEqual(a.xref_stream_raw(xref), b.xref_stream_raw(xref), xref)
                self.assertEqual(a.xref_get_key(xref, "SMask"), b.xref_get_key(xref, "SMask"))
            for xref in (self.x["duplicate"], self.x["cmyk"], self.x["icon"]):
                self.assertEqual(a.xref_stream_raw(xref), b.xref_stream_raw(xref), xref)
            # The table-cell image keeps its displayed box.
            self.assertEqual(b[0].get_image_rects(self.x["table_image"])[0], a[0].get_image_rects(self.x["table_image"])[0])
        finally:
            a.close()
            b.close()

    def test_memory_chunking_gives_identical_result(self):
        """chunkBytes=1 forces an incremental save + reopen after every image and during validation."""
        work = self.dir / "chunk"
        (work / "x").mkdir(parents=True)
        xrefs = [self.x["photo"], self.x["alpha"], self.x["gray"], self.x["scan"]]
        items = pdf_ops.pdf_extract({"path": self.pdf, "xrefs": xrefs, "outputDir": str(work / "x"), "chunkBytes": 1}, Ctx())["images"]
        reps = [{"xref": i["xref"], "path": i["path"]} for i in items]
        normal = pdf_ops.pdf_replace({"source": self.pdf, "output": str(work / "a.pdf"), "replacements": reps}, Ctx())
        chunked = pdf_ops.pdf_replace({"source": self.pdf, "output": str(work / "b.pdf"), "replacements": reps, "chunkBytes": 1}, Ctx())
        self.assertTrue(chunked["validation"]["ok"])
        self.assertEqual(normal["replaced"], chunked["replaced"])
        a, b = pymupdf.open(str(work / "a.pdf")), pymupdf.open(str(work / "b.pdf"))
        for xref in xrefs:
            self.assertEqual(a.xref_stream(xref), b.xref_stream(xref))
        a.close()
        b.close()
        self.assertEqual(sorted(os.listdir(work)), ["a.pdf", "b.pdf", "x"])  # no .work/.tmp left behind

    def test_resized_image_keeps_displayed_size(self):
        work = self.dir / "resize"
        (work / "x").mkdir(parents=True)
        xref = self.x["photo"]
        item = pdf_ops.pdf_extract({"path": self.pdf, "xrefs": [xref], "outputDir": str(work / "x")}, Ctx())["images"][0]
        target = str(work / "big.jpg")
        enhance.enhance_cmd({"input": item["path"], "output": target, "format": "jpeg",
                             "settings": {"mode": "manual", "sharpenAmount": 0.3, "resizeFactor": 2,
                                          "keepPixelSize": False}}, None)
        output = str(work / "out.pdf")
        pdf_ops.pdf_replace({"source": self.pdf, "output": output, "replacements": [{"xref": xref, "path": target}]}, Ctx())
        a, b = pymupdf.open(self.pdf), pymupdf.open(output)
        self.assertEqual(b.extract_image(xref)["width"], 640)
        self.assertEqual(a[0].get_image_rects(xref), b[0].get_image_rects(xref))
        a.close()
        b.close()

    def test_bad_replacement_fails_alone(self):
        work = self.dir / "bad"
        work.mkdir()
        wrong = samples.write(work / "rgb.png", samples.photo(100, 140))  # RGB for a grayscale slot
        good_src = pdf_ops.pdf_extract({"path": self.pdf, "xrefs": [self.x["alpha"]], "outputDir": str(work)}, Ctx())["images"][0]["path"]
        output = str(work / "out.pdf")
        result = pdf_ops.pdf_replace({"source": self.pdf, "output": output, "replacements": [
            {"xref": self.x["gray"], "path": wrong},
            {"xref": self.x["alpha"], "path": good_src},
        ]}, Ctx())
        self.assertEqual([f["xref"] for f in result["failed"]], [self.x["gray"]])
        self.assertEqual(result["replaced"], [self.x["alpha"]])
        a, b = pymupdf.open(self.pdf), pymupdf.open(output)
        self.assertEqual(a.xref_stream_raw(self.x["gray"]), b.xref_stream_raw(self.x["gray"]))
        a.close()
        b.close()

    def test_validation_detects_unexpected_changes(self):
        work = self.dir / "tamper"
        work.mkdir()
        tampered = str(work / "t.pdf")
        doc = pymupdf.open(self.pdf)
        page = doc[0]
        page.insert_text((72, 700), "extra text")
        doc.save(tampered)
        doc.close()
        result = pdf_ops.validate(self.pdf, tampered, None, [])
        self.assertFalse(result["ok"])
        self.assertTrue(any("changed" in p for p in result["problems"]))

    def test_output_is_not_written_when_validation_fails(self):
        work = self.dir / "novalid"
        work.mkdir()
        output = work / "out.pdf"
        original_validate = pdf_ops.validate
        pdf_ops.validate = lambda *a: {"ok": False, "problems": ["simulated"]}
        try:
            with self.assertRaises(WorkerError) as ctx:
                pdf_ops.pdf_replace({"source": self.pdf, "output": str(output), "replacements": []}, Ctx())
        finally:
            pdf_ops.validate = original_validate
        self.assertEqual(ctx.exception.code, "validation_failed")
        self.assertEqual(os.listdir(work), [])

    def test_output_must_differ_from_source(self):
        with self.assertRaises(WorkerError):
            pdf_ops.pdf_replace({"source": self.pdf, "output": self.pdf, "replacements": []}, Ctx())


class PdfErrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def code(self, args) -> str:
        with self.assertRaises(WorkerError) as ctx:
            pdf_ops.pdf_scan(args, Ctx())
        return ctx.exception.code

    def test_corrupt_and_non_pdf_files(self):
        (self.dir / "junk.pdf").write_bytes(b"%PDF-1.7\n" + os.urandom(2000))
        samples.write(self.dir / "picture.pdf", samples.photo(50, 50), ext=".png")
        (self.dir / "empty.pdf").write_bytes(b"")
        for name in ("junk.pdf", "picture.pdf", "empty.pdf"):
            with self.subTest(name):
                self.assertIn(self.code({"path": str(self.dir / name)}), ("corrupt_pdf",))
        self.assertEqual(self.code({"path": str(self.dir / "missing.pdf")}), "file_not_found")

    def test_signed_pdf_is_reported(self):
        path = str(self.dir / "signed.pdf")
        doc = pymupdf.open()
        page = doc.new_page()
        widget = pymupdf.Widget()
        widget.field_type = pymupdf.PDF_WIDGET_TYPE_SIGNATURE
        widget.field_name = "Approval"
        widget.rect = pymupdf.Rect(50, 50, 200, 100)
        page.add_widget(widget)
        doc.save(path)
        doc.close()
        self.assertTrue(pdf_ops.pdf_scan({"path": path}, Ctx())["signed"])
        unsigned = str(self.dir / "plain.pdf")
        pdf_samples.build_feature_pdf(unsigned)
        self.assertFalse(pdf_ops.pdf_scan({"path": unsigned}, Ctx())["signed"])

    def test_password_protected_pdf(self):
        path = str(self.dir / "secret.pdf")
        x = pdf_samples.build_feature_pdf(path, password="s3cret")
        self.assertEqual(self.code({"path": path}), "password_required")
        self.assertEqual(self.code({"path": path, "password": "nope"}), "wrong_password")
        scan = pdf_ops.pdf_scan({"path": path, "password": "s3cret"}, Ctx())
        self.assertEqual(scan["pageCount"], 5)

        (self.dir / "x").mkdir()
        item = pdf_ops.pdf_extract({"path": path, "password": "s3cret", "xrefs": [x["photo"]], "outputDir": str(self.dir / "x")}, Ctx())["images"][0]
        output = str(self.dir / "out.pdf")
        result = pdf_ops.pdf_replace({"source": path, "output": output, "password": "s3cret",
                                      "replacements": [{"xref": x["photo"], "path": item["path"]}]}, Ctx())
        self.assertTrue(result["validation"]["ok"])
        out = pymupdf.open(output)
        self.assertTrue(out.needs_pass, "encryption must be kept")
        out.close()


class TextEncodedImageTests(unittest.TestCase):
    """Images wrapped in ASCII85 or ASCIIHex text encoding, as ReportLab and other PDF writers store them."""

    @classmethod
    def setUpClass(cls):
        import base64
        import zlib
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.pdf = str(cls.dir / "text_encoded.pdf")
        rgb = samples.add_noise(samples.photo(180, 240, seed=4), 10)
        gray = cv2.cvtColor(samples.faded_scan(samples.line_art(180, 240)), cv2.COLOR_BGR2GRAY)
        jpeg = image_io.encode(rgb, "jpeg")
        a85 = lambda data: base64.a85encode(data) + b"~>"
        hexa = lambda data: data.hex().encode() + b">"
        doc = pymupdf.open()
        page = doc.new_page()
        cls.images = {}
        streams = [
            # name, placeholder, encoded stream, filter, colour space
            ("raw_a85", rgb, a85(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).tobytes()), "/ASCII85Decode", "/DeviceRGB"),
            # (PyMuPDF reuses one object for identical inserted pictures, so each placeholder differs.)
            ("jpeg_a85", np.ascontiguousarray(rgb[::-1]), a85(jpeg), "[/ASCII85Decode /DCTDecode]", "/DeviceRGB"),
            ("flate_hex", gray, hexa(zlib.compress(gray.tobytes())), "[/ASCIIHexDecode /FlateDecode]", "/DeviceGray"),
        ]
        for k, (name, placeholder, data, filters, colorspace) in enumerate(streams):
            xref = page.insert_image(pymupdf.Rect(50, 50 + k * 200, 290, 230 + k * 200),
                                     stream=image_io.encode(placeholder, "png"))
            doc.update_stream(xref, data, compress=0)
            doc.xref_set_key(xref, "Filter", filters)
            doc.xref_set_key(xref, "DecodeParms", "null")
            doc.xref_set_key(xref, "ColorSpace", colorspace)
            cls.images[name] = xref
        assert len(set(cls.images.values())) == len(streams)
        # The placeholders' colour profiles are now unused objects, as earlier edits often leave in real PDFs.
        doc.save(cls.pdf)
        doc.close()
        cls.rgb, cls.gray = rgb, gray

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_text_encoded_images_are_enhanced_and_replaced(self):
        scan = {i["xref"]: i for i in pdf_ops.pdf_scan({"path": self.pdf}, Ctx())["images"]}
        for name, xref in self.images.items():
            self.assertTrue(scan[xref]["eligible"], (name, scan[xref]))
        self.assertEqual(scan[self.images["jpeg_a85"]]["outputFormat"], "jpeg")
        self.assertEqual(scan[self.images["raw_a85"]]["outputFormat"], "png")

        out = self.dir / "extracted"
        out.mkdir()
        extracted = pdf_ops.pdf_extract({"path": self.pdf, "xrefs": list(self.images.values()), "outputDir": str(out)}, Ctx())
        paths = {r["xref"]: r["path"] for r in extracted["images"]}
        self.assertTrue(np.array_equal(image_io.decode(paths[self.images["raw_a85"]]), self.rgb))  # lossless
        self.assertTrue(np.array_equal(image_io.decode(paths[self.images["flate_hex"]]), self.gray))

        replacements = []
        for name, xref in self.images.items():
            fmt = scan[xref]["outputFormat"]
            target = self.dir / f"e{xref}{'.jpg' if fmt == 'jpeg' else '.png'}"
            result = enhance.enhance_cmd({"input": paths[xref], "output": str(target), "format": fmt,
                                          "settings": {"mode": "manual", "sharpenAmount": 0.8}}, None)
            self.assertEqual(result["status"], "completed", result)
            replacements.append({"xref": xref, "path": str(target)})
        output = str(self.dir / "out.pdf")
        result = pdf_ops.pdf_replace({"source": self.pdf, "output": output, "replacements": replacements}, Ctx())
        self.assertEqual(sorted(result["replaced"]), sorted(self.images.values()))
        self.assertTrue(result["validation"]["ok"], result["validation"])
        doc = pymupdf.open(output)
        try:
            self.assertEqual(doc.xref_get_key(self.images["jpeg_a85"], "Filter")[1], "/DCTDecode")
            self.assertEqual(doc.xref_get_key(self.images["raw_a85"], "Filter")[1], "/FlateDecode")
            for xref in self.images.values():
                info = doc.extract_image(xref)
                self.assertEqual((info["width"], info["height"]), (240, 180))
        finally:
            doc.close()


if __name__ == "__main__":
    unittest.main()
