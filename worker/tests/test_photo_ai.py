import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyse  # noqa: E402
import enhance  # noqa: E402
import photo_ai  # noqa: E402
import samples  # noqa: E402


def degraded_photo():
    clean = samples.good_photo()
    blurred = samples.add_noise(cv2.GaussianBlur(clean, (0, 0), 1.0), 8)
    ok, buf = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return clean, cv2.imdecode(buf, cv2.IMREAD_COLOR)


@unittest.skipUnless(photo_ai.available(), "AI photo model not installed")
class PhotoAiTests(unittest.TestCase):
    def test_restores_a_degraded_photo_at_the_same_size(self):
        clean, degraded = degraded_photo()
        restored = photo_ai.restore(degraded)
        self.assertEqual(restored.shape, degraded.shape)
        gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self.assertGreater(analyse.ssim(gray(restored), gray(clean)), analyse.ssim(gray(degraded), gray(clean)) + 0.05)

    def test_used_for_photos_only_and_only_when_asked(self):
        _, degraded = degraded_photo()
        with tempfile.TemporaryDirectory() as tmp:
            photo = samples.write(Path(tmp) / "photo.png", degraded)
            plain = enhance.enhance_cmd({"input": photo, "output": str(Path(tmp) / "a.png")}, None)
            self.assertNotIn("ai_photo", [o["op"] for o in plain["operations"]])
            ai = enhance.enhance_cmd({"input": photo, "output": str(Path(tmp) / "b.png"), "settings": {"aiPhoto": True}}, None)
            self.assertIn("ai_photo", [o["op"] for o in ai["operations"]])
            self.assertIn("ai_restored", ai["reviewReasons"])
            self.assertEqual((ai["output"]["width"], ai["output"]["height"]), (640, 480))
            scan = samples.write(Path(tmp) / "scan.png", samples.faded_scan(samples.line_art()))
            text = enhance.enhance_cmd({"input": scan, "output": str(Path(tmp) / "c.png"), "settings": {"aiPhoto": True}}, None)
            self.assertNotIn("ai_photo", [o["op"] for o in text["operations"]])  # never on text or diagrams


if __name__ == "__main__":
    unittest.main()
