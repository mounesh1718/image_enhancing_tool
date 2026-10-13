import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai_runtime  # noqa: E402
import analyse  # noqa: E402
import enhance  # noqa: E402
import image_io  # noqa: E402
import restore_ai  # noqa: E402
import samples  # noqa: E402
import textcheck  # noqa: E402

LINES = ["Batch 4521: assay 99.3 % w/w", "Water content 0.42 % (KF)", "Lot 7781  Exp 03/2028"]


def text_page(lines=LINES, width=640) -> np.ndarray:
    page = np.full((60 + 55 * len(lines), width, 3), 235, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(page, line, (20, 50 + i * 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2, cv2.LINE_AA)
    return page


def grainy(img: np.ndarray, sigma: float = 30) -> np.ndarray:
    return np.clip(img + np.random.default_rng(2).normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


class DeviceTests(unittest.TestCase):
    def test_processor_can_be_forced(self):
        with mock.patch.dict(os.environ, {"IQE_AI_DEVICE": "cpu"}):
            self.assertIsNone(ai_runtime.best_adapter())
            self.assertEqual(ai_runtime.device()["kind"], "cpu")

    def test_adapters_are_listed_without_the_software_renderer(self):
        for chip in ai_runtime.adapters():
            self.assertNotIn("Basic Render", chip["name"])

    @unittest.skipUnless(__import__("importlib").util.find_spec("onnx"), "onnx not installed")
    def test_a_chip_that_fails_is_not_used_and_the_choice_is_remembered(self):
        import onnx
        from onnx import TensorProto, helper
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"IQE_MODELS": tmp}):
            graph = helper.make_graph([helper.make_node("Identity", ["image"], ["restored"])], "identity",
                                      [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])],
                                      [helper.make_tensor_value_info("restored", TensorProto.FLOAT, [1, 3, 8, 8])])
            model = os.path.join(tmp, "identity.onnx")
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), model)
            # A chip number that does not exist: the trial in its own process fails, so the processor is used.
            fake = {"index": 97, "name": "Missing chip", "memoryGB": 1.0}
            with mock.patch.object(ai_runtime, "best_adapter", return_value=fake):
                self.assertIsNone(ai_runtime.chip_for(model, (1, 3, 8, 8)))
                choices = json.loads((Path(tmp) / "device_choice.json").read_text(encoding="utf-8"))
                self.assertEqual(list(choices.values()), ["processor"])
                with mock.patch.object(ai_runtime.subprocess, "run", side_effect=AssertionError("tried again")):
                    self.assertIsNone(ai_runtime.chip_for(model, (1, 3, 8, 8)))  # remembered, not tried again
                sess = ai_runtime.session(model, shape=(1, 3, 8, 8))
                self.assertEqual(ai_runtime.where(sess), "processor")


class MeasurementTests(unittest.TestCase):
    def test_jpeg_save_quality_is_read_from_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            for quality in (30, 60, 90):
                path = os.path.join(tmp, f"q{quality}.jpg")
                cv2.imwrite(path, samples.good_photo(), [cv2.IMWRITE_JPEG_QUALITY, quality])
                self.assertAlmostEqual(image_io.probe(path).details["quality"], quality, delta=2)

    def test_blur_is_measured_on_faded_low_contrast_pictures(self):
        line_art = cv2.cvtColor(samples.line_art(420, 780), cv2.COLOR_BGR2GRAY).astype(np.float32)
        faded = (140 + cv2.GaussianBlur(line_art, (0, 0), 2.0) * (60 / 255.0)).astype(np.uint8)
        self.assertGreater(analyse.estimate_blur(faded), 1.6)  # was measured as sharp (0) before
        self.assertEqual(analyse.estimate_blur((140 + line_art * (60 / 255.0)).astype(np.uint8)), 0.0)


@unittest.skipUnless(textcheck.available(), "OCR (rapidocr-onnxruntime) not installed")
class ChoiceTests(unittest.TestCase):
    """The safety rules of enhance._choose, with hand-made versions (no AI model needed)."""

    settings = dict(enhance.DEFAULT_SETTINGS)

    def choose(self, original, **versions):
        analysis = analyse.analyse(original)
        candidates = [{"name": name, "ops": [], "image": image} for name, image in versions.items()]
        return enhance._choose(original, analysis, candidates, self.settings)

    def test_a_sharpened_version_with_a_changed_number_is_never_kept(self):
        original = text_page()
        invented = text_page(["Batch 4527: assay 99.3 % w/w", "Water content 0.42 % (KF)", "Lot 7781  Exp 03/2028"])
        chosen, info = self.choose(original, filters=original.copy(), ai_deblur=invented)
        self.assertEqual(chosen["name"], "filters")
        rejected = {c["name"]: c.get("rejected") for c in info["candidates"]}
        self.assertIn("number", rejected["ai_deblur"])

    def test_digits_the_original_does_not_show_readably_are_not_accepted(self):
        original = text_page()
        original[150:185, 20:120] = cv2.GaussianBlur(original[150:185, 20:120], (0, 0), 6)  # "Lot 7781" unreadable
        chosen, info = self.choose(original, filters=original.copy(), ai_deblur=text_page())
        self.assertEqual(chosen["name"], "filters")
        self.assertIsNotNone(next(c for c in info["candidates"] if c["name"] == "ai_deblur").get("rejected"))

    def test_an_equally_readable_cleaner_version_is_kept(self):
        original = grainy(text_page(), 25)
        clean = text_page()
        chosen, info = self.choose(original, filters=original.copy(), ai_clean=clean)
        self.assertEqual(chosen["name"], "ai_clean")
        self.assertEqual(info["chosen"], "ai_clean")

    def test_pictures_without_text_keep_the_filters_when_ai_loses_detail(self):
        photo = samples.add_noise(samples.good_photo(), 12)
        filters = cv2.fastNlMeansDenoisingColored(photo, None, 8, 8, 7, 21)
        smeared = cv2.GaussianBlur(photo, (0, 0), 4)
        chosen, info = self.choose(photo, filters=filters, ai_clean=smeared)
        self.assertEqual(chosen["name"], "filters")
        self.assertIn("detail", info["candidates"][1]["rejected"])


def installed(*kinds):
    return all(restore_ai.installed(k) for k in kinds)


@unittest.skipUnless(installed("grain", "jpeg", "deblur_text"), "AI repair models not installed (worker/setup_ai.py)")
class RepairModelTests(unittest.TestCase):
    def test_grain_is_removed_at_the_same_size_in_colour_and_grey(self):
        noisy = grainy(text_page(width=320)[:160])
        out = restore_ai.run("grain", noisy)
        self.assertEqual(out.shape, noisy.shape)
        self.assertLess(analyse.estimate_noise(analyse.to_gray8(out)), analyse.estimate_noise(analyse.to_gray8(noisy)) / 4)
        grey = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
        self.assertEqual(restore_ai.run("grain", grey).shape, grey.shape)

    def test_jpeg_blocks_are_removed(self):
        clean = text_page(width=320)[:160]
        ok, buf = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, 10])
        blocky = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.assertGreater(cv2.PSNR(restore_ai.run("jpeg", blocky), clean), cv2.PSNR(blocky, clean))

    def test_blurred_text_gets_sharper(self):
        blurred = cv2.GaussianBlur(text_page(width=384)[:160], (0, 0), 2.0)
        self.assertGreater(analyse.sharpness(analyse.to_gray8(restore_ai.run("deblur_text", blurred))),
                           2 * analyse.sharpness(analyse.to_gray8(blurred)))


@unittest.skipUnless(installed("grain", "jpeg") and textcheck.available(), "AI repair models or OCR not installed")
class EnhanceWithAiTests(unittest.TestCase):
    def run_enhance(self, path, out, ai):
        return enhance.enhance_cmd({"input": path, "output": out, "settings": {"aiRepair": ai}}, None)

    def test_grainy_text_is_cleaned_by_ai_and_the_text_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "grain.png", grainy(text_page(), 30))
            result = self.run_enhance(src, str(Path(tmp) / "a.png"), True)
            self.assertEqual(result["choice"]["chosen"], "ai_clean")
            self.assertEqual(result["operations"][0]["op"], "ai_grain")
            self.assertFalse(result["textCheck"]["flag"], result["textCheck"])
            self.assertIn("SCUNet", result["aiDevices"])
            off = self.run_enhance(src, str(Path(tmp) / "b.png"), False)
            self.assertFalse(any(o["op"].startswith("ai_") for o in off["operations"]))
            self.assertEqual(off["choice"], {})

    def test_strongly_compressed_jpeg_text_is_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = str(Path(tmp) / "low.jpg")
            cv2.imwrite(src, text_page(), [cv2.IMWRITE_JPEG_QUALITY, 8])
            result = self.run_enhance(src, str(Path(tmp) / "o.jpg"), True)
            self.assertEqual(result["analysis"]["jpegQuality"] <= enhance.JPEG_MAX_QUALITY, True)
            self.assertIn("ai_jpeg", [o["op"] for o in result["operations"]])

    def test_unreadable_text_is_never_sharpened_by_ai(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "blur.png", cv2.GaussianBlur(text_page(), (0, 0), 6))
            result = self.run_enhance(src, str(Path(tmp) / "o.png"), True)
            self.assertNotIn("ai_deblur_text", [o["op"] for o in result["operations"]])


if __name__ == "__main__":
    unittest.main()
