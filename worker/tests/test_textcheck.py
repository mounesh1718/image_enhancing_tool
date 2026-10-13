import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import enhance  # noqa: E402
import samples  # noqa: E402
import textcheck  # noqa: E402


def line(text, x0=0, confidence=0.95):
    return {"box": (x0, 0, x0 + 200, 30), "text": text, "confidence": confidence}


def label(text_top: str, text_bottom: str) -> np.ndarray:
    img = np.full((200, 700, 3), 255, np.uint8)
    cv2.putText(img, text_top, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, text_bottom, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2, cv2.LINE_AA)
    return img


class CompareTests(unittest.TestCase):
    """The comparison logic on given readings (no OCR needed)."""

    def test_identical_readings_have_no_changes(self):
        r = textcheck.compare([line("Batch 4521")], [line("Batch 4521")], [line("Batch 4521")])
        self.assertEqual((r["changedCharacters"], r["numbersChanged"]), (0, []))

    def test_changed_digit_is_reported(self):
        r = textcheck.compare([line("Batch 4521")], [line("Batch 4527")], [line("Batch 4521")])
        self.assertEqual(r["changedCharacters"], 1)
        self.assertEqual(len(r["numbersChanged"]), 1)
        self.assertIn("[1]", r["numbersChanged"][0])

    def test_punctuation_and_unstable_characters_are_ignored(self):
        # "." vs "," is OCR noise; "0" was read as "o" in the second reading of the original, so it is not stable.
        r = textcheck.compare([line("2.0 g, 14.5 mmol")], [line("2.o g. 14,5 mmol")], [line("2.o g, 14.5 mmol")])
        self.assertEqual((r["changedCharacters"], r["numbersChanged"]), (0, []))

    def test_a_digit_read_only_afterwards_is_not_a_change(self):
        r = textcheck.compare([line("2.og")], [line("2.0g")], [line("2.og")])
        self.assertEqual(r["numbersChanged"], [])

    def test_missing_line_counts_as_changed(self):
        r = textcheck.compare([line("Assay 99.3"), line("Water 0.42", x0=400)], [line("Assay 99.3")],
                              [line("Assay 99.3"), line("Water 0.42", x0=400)])
        self.assertEqual(r["changedCharacters"], len("Water042"))


@unittest.skipUnless(textcheck.available(), "OCR (rapidocr-onnxruntime) not installed")
class OcrCheckTests(unittest.TestCase):
    def test_unchanged_text_passes_and_a_changed_value_is_flagged(self):
        original = label("Batch 4521: assay 99.3 %", "Water content 0.42 % w/w")
        self.assertFalse(textcheck.check(original, original.copy())["flag"])
        altered = original.copy()
        altered[50:95, 20:210] = 255
        cv2.putText(altered, "Batch 4527:", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2, cv2.LINE_AA)
        result = textcheck.check(original, altered)
        self.assertTrue(result["flag"])
        self.assertTrue(any("1" in n and "7" in n for n in result["numbersChanged"]))

    def test_a_symbol_misread_as_a_digit_does_not_flag_intact_text(self):
        # Blurry "Rs" before a price reads as a digit in the original and correctly afterwards: the strokes are all
        # still there, so it is reported as uncertain, not as a changed number.
        original = label("Price Rs 120.00", "Qty 10")
        blurred = cv2.GaussianBlur(original, (0, 0), 1.6)
        result = textcheck.check(blurred, original)
        self.assertFalse(result["flag"], result)
        self.assertEqual(result["numbersChanged"], [])

    def test_enhancement_reports_the_text_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "label.png", cv2.GaussianBlur(label("Batch 4521: assay 99.3 %", "Water 0.42 % w/w"), (0, 0), 1.2))
            result = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "out.png")}, None)
            self.assertEqual(result["status"], "completed")
            check = result["textCheck"]
            self.assertTrue(check["available"] and check["textFound"])
            self.assertFalse(check["flag"], check)
            self.assertNotIn("text_changed", result["reviewReasons"])
            # Photos are not read.
            photo = samples.write(Path(tmp) / "photo.png", samples.add_noise(samples.photo(), 12))
            result = enhance.enhance_cmd({"input": photo, "output": str(Path(tmp) / "p.png")}, None)
            self.assertTrue(result["textCheck"].get("skipped"))

    def test_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "scan.png", samples.faded_scan(samples.line_art()))
            result = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "o.png"),
                                          "settings": {"textCheck": False}}, None)
            self.assertTrue(result["textCheck"].get("skipped"))


if __name__ == "__main__":
    unittest.main()
