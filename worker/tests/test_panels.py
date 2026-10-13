import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import enhance  # noqa: E402
import panels  # noqa: E402
import samples  # noqa: E402


def sheet() -> np.ndarray:
    """A 2x2 figure sheet: framed panels with titles above them (photo, scan, photo, diagram)."""
    page = np.full((900, 900, 3), 255, np.uint8)
    contents = [samples.add_noise(samples.photo(330, 400, seed=2), 10), samples.faded_scan(samples.line_art(330, 400)),
                samples.blur(samples.good_photo(330, 400), 2), samples.faded_scan(samples.line_art(330, 400))]
    for k, content in enumerate(contents):
        x, y = 20 + (k % 2) * 440, 60 + (k // 2) * 420
        cv2.putText(page, f"Image {k + 1}", (x + 140, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        page[y:y + 330, x:x + 400] = content
        cv2.rectangle(page, (x - 2, y - 2), (x + 401, y + 331), (90, 90, 90), 2)
    return page


class PanelDetectionTests(unittest.TestCase):
    def test_finds_framed_panels_in_reading_order(self):
        boxes = panels.detect(sheet())
        self.assertEqual(len(boxes), 4)
        self.assertLess(boxes[0][0], boxes[1][0])  # first row: left, then right
        self.assertLess(boxes[1][1], boxes[2][1])  # then the second row

    def test_ordinary_images_are_not_composites(self):
        for img in (samples.photo(), samples.poor_scan(), samples.line_art()):
            self.assertEqual(panels.detect(img), [])

    def test_each_panel_gets_its_own_treatment_and_titles_stay_unchanged(self):
        page = sheet()
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "sheet.png", page)
            result = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "out.png")}, None)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["panels"]), 4)
            kinds = [p["content"] for p in result["panels"]]
            self.assertEqual(kinds.count("photo"), 2)
            self.assertIn("clean_background", result["panels"][1]["operations"])  # the faded scan panel
            self.assertNotIn("clean_background", result["panels"][0]["operations"])  # the photo panel
            out = cv2.imread(str(Path(tmp) / "out.png"))
            self.assertEqual(out.shape, page.shape)
            self.assertTrue(np.array_equal(out[:50], page[:50]))  # the title strip is untouched


class SpeckRemovalTests(unittest.TestCase):
    def page(self):
        img = np.full((300, 500, 3), 250, np.uint8)
        cv2.putText(img, "pH 7.45", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.line(img, (300, 110), (320, 110), (20, 20, 20), 3)  # a short dash (bond or minus sign)
        return img

    def test_isolated_specks_go_and_text_stays(self):
        clean = self.page()
        dirty = clean.copy()
        rng = np.random.default_rng(1)
        for x, y in rng.integers([20, 180], [480, 290], size=(40, 2)):
            cv2.circle(dirty, (int(x), int(y)), 1, (30, 30, 30), -1)
        out = enhance._remove_specks(dirty, {})
        dark = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 128
        self.assertLess(dark(out)[170:, :].sum(), dark(dirty)[170:, :].sum() * 0.1)   # specks below the text removed
        self.assertTrue(np.array_equal(out[60:140, 30:340], dirty[60:140, 30:340]))      # text, decimal point, dash kept

    def test_clean_page_is_untouched(self):
        page = self.page()
        self.assertTrue(np.array_equal(enhance._remove_specks(page, {}), page))


if __name__ == "__main__":
    unittest.main()
