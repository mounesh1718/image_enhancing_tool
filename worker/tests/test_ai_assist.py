"""Optional AI enhancement after the normal result: suggestions, eligibility, checks and the ai_enhance command."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai_assist  # noqa: E402
import analyse  # noqa: E402
import enhance  # noqa: E402
import figure_classifier as fc  # noqa: E402
import restore_ai  # noqa: E402
import samples  # noqa: E402


def measured(**values) -> dict:
    """An analysis of a clean 800 x 600 picture, with the given measurements changed."""
    base = {"width": 800, "height": 600, "dtype": "uint8", "bilevel": False, "content": "text_or_diagram",
            "noise": 0.0, "noiseRatio": 0.5, "blur": 0.0, "sharpness": 5000.0}
    return base | values


def kinds(analysis, category=None, has_panels=False):
    with patch.object(ai_assist, "_installed", return_value=True):
        return [s["kind"] for s in ai_assist.suggestions(analysis, category, has_panels)]


class SuggestionTests(unittest.TestCase):
    def test_each_kind_of_damage_suggests_its_model(self):
        self.assertEqual(kinds(measured(noise=9.0, noiseRatio=0.4), fc.OTHER), ["grain"])
        self.assertEqual(kinds(measured(jpegQuality=25), fc.TABLE), ["jpeg"])
        self.assertEqual(kinds(measured(blur=2.4), fc.TABLE), ["deblur_text"])
        self.assertEqual(kinds(measured(content="photo", blur=2.0), fc.OTHER), ["photo"])

    def test_suggestions_never_promise_and_name_the_model(self):
        with patch.object(ai_assist, "_installed", return_value=True):
            found = ai_assist.suggestions(measured(noise=9.0, noiseRatio=0.4), fc.OTHER)
        self.assertEqual(found, [{"kind": "grain", "model": "SCUNet", "text": "Grain detected — SCUNet may help.", "installed": True}])
        for text in ai_assist.SUGGESTION.values():
            self.assertIn("may", text)
            self.assertNotIn("will", text)

    def test_the_photo_model_is_never_suggested_for_text_tables_graphs_or_chemistry(self):
        soft_photo = measured(content="photo", blur=2.0)
        for category in (fc.TABLE, fc.GRAPH, fc.CHEMICAL, None):
            self.assertNotIn("photo", kinds(soft_photo, category), category)
        self.assertNotIn("photo", kinds(measured(content="text_or_diagram", blur=2.0), fc.OTHER))
        # A faithful repair applies: the photo model, which invents texture, is not added.
        self.assertEqual(kinds(measured(content="photo", blur=2.0, noise=9.0, noiseRatio=0.4), fc.OTHER), ["grain"])

    def test_clean_bilevel_and_sheet_pictures_get_no_suggestion(self):
        self.assertEqual(kinds(measured(), fc.TABLE), [])
        self.assertEqual(kinds(measured(noise=9.0, noiseRatio=0.4, bilevel=True), fc.OTHER), [])
        self.assertEqual(kinds(measured(noise=9.0, noiseRatio=0.4), fc.OTHER, has_panels=True), [])
        self.assertEqual(kinds(measured(noise=9.0, noiseRatio=0.95), fc.OTHER), [])  # fine texture, not grain

    def test_missing_models_are_reported_not_tried(self):
        with patch.object(ai_assist, "_installed", return_value=False):
            result = ai_assist.try_ai(np.full((60, 80, 3), 200, np.uint8), measured(width=80, height=60, noise=9.0, noiseRatio=0.4),
                                      {}, {"category": fc.OTHER}, None)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("SCUNet is not installed", result["reasons"][0])
        self.assertNotIn("image", result)

    def test_clean_pictures_are_not_eligible(self):
        result = ai_assist.try_ai(np.full((60, 80, 3), 255, np.uint8), measured(width=80, height=60), {}, {"category": fc.TABLE}, None)
        self.assertEqual((result["status"], result["reasons"]), ("not_eligible", [ai_assist.NONE_SUITABLE]))

    def test_enhance_reports_suggestions_without_running_ai(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "noisy.png", samples.add_noise(samples.good_photo(), 12))
            with patch.object(restore_ai, "run", side_effect=AssertionError("AI ran during normal processing")):
                result = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "o.png"),
                                              "classification": {"category": fc.OTHER}}, None)
        self.assertIn("grain", [s["kind"] for s in result["aiSuggestions"]])
        self.assertFalse(any(o["op"].startswith("ai_") for o in result["operations"]))


class CheckTests(unittest.TestCase):
    """The AI version faces the same checks as any result; a failing one is rejected with the reason."""

    def setUp(self):
        self.img = samples.add_noise(samples.good_photo(), 12)
        self.analysis = analyse.analyse(self.img)

    def fake_candidates(self, image):
        return lambda img, analysis, settings, kinds: [{"name": "ai_clean", "ops": [{"op": "ai_grain"}], "image": image}]

    def run_with(self, image, category=fc.OTHER):
        with patch.object(ai_assist, "_installed", return_value=True), \
                patch.object(ai_assist, "_candidates", side_effect=self.fake_candidates(image)):
            return ai_assist.try_ai(self.img, self.analysis, {}, {"category": category}, None)

    def test_a_version_losing_detail_is_rejected(self):
        smeared = cv2.GaussianBlur(self.img, (0, 0), 6)
        result = self.run_with(smeared)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("SCUNet: more detail lost or added than with the filters", result["reasons"])
        self.assertIs(result["image"], smeared)  # kept, so it can be compared

    def test_a_faithful_version_is_accepted(self):
        result = self.run_with(cv2.fastNlMeansDenoisingColored(self.img, None, 9, 9, 7, 21))
        self.assertEqual(result["status"], "accepted", result["reasons"])
        self.assertEqual(result["models"], ["SCUNet"])
        self.assertIn("detail", result["checks"])

    def test_chemical_diagrams_must_read_as_the_same_molecule(self):
        good = cv2.fastNlMeansDenoisingColored(self.img, None, 9, 9, 7, 21)
        with patch.object(ai_assist, "_chemical_safety", return_value="the chemical structure reads differently after the AI change"):
            result = self.run_with(good, category=fc.CHEMICAL)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("reads differently", result["reasons"][0])

        def read(img):
            return {"status": "read", "inchiKey": "SAME"}

        with patch.object(fc, "chemical_check", side_effect=read):
            self.assertIsNone(ai_assist._chemical_safety(self.img, good))
        with patch.object(fc, "chemical_check", side_effect=[{"status": "read", "inchiKey": "A"}, {"status": "read", "inchiKey": "B"}]):
            self.assertIn("reads differently", ai_assist._chemical_safety(self.img, good))
        with patch.object(fc, "chemical_check", return_value={"status": "uncertain"}):
            self.assertIn("cannot be read reliably", ai_assist._chemical_safety(self.img, good))


@unittest.skipUnless(restore_ai.installed("grain"), "SCUNet is not installed (worker/setup_ai.py)")
class AiEnhanceCommandTests(unittest.TestCase):
    def test_the_command_writes_a_checked_ai_version_and_never_touches_the_opencv_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "noisy.png", samples.add_noise(samples.good_photo(), 12))
            normal = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "cv.png")}, None)
            self.assertEqual(normal["status"], "completed")
            before = hashlib.sha256((Path(tmp) / "cv.png").read_bytes()).hexdigest()
            result = enhance.ai_enhance_cmd({"input": src, "output": str(Path(tmp) / "ai.png"), "opencv": str(Path(tmp) / "cv.png"),
                                             "classification": {"category": fc.OTHER},
                                             "thumbnails": {"ai": str(Path(tmp) / "ai_thumb.jpg")}}, None)
            self.assertIn(result["status"], ("accepted", "rejected"))
            self.assertEqual(result["models"], ["SCUNet"])
            ai = cv2.imread(str(Path(tmp) / "ai.png"))
            self.assertEqual(ai.shape, cv2.imread(src).shape)  # same pixel size
            self.assertTrue((Path(tmp) / "ai_thumb.jpg").exists())
            self.assertEqual(hashlib.sha256((Path(tmp) / "cv.png").read_bytes()).hexdigest(), before)
            if result["status"] == "accepted":
                self.assertLess(analyse.estimate_noise(analyse.to_gray8(ai)), analyse.estimate_noise(analyse.to_gray8(cv2.imread(src))))


if __name__ == "__main__":
    unittest.main()
