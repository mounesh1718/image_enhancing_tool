import io
import json
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
import image_io  # noqa: E402
import samples  # noqa: E402
from protocol import WorkerError, handle_line  # noqa: E402
from test_image_io import jpeg_with_orientation  # noqa: E402


class AnalyseTests(unittest.TestCase):
    def test_classifies_photos_and_diagrams(self):
        self.assertEqual(analyse.analyse(samples.photo())["content"], "photo")
        self.assertEqual(analyse.analyse(samples.low_contrast(samples.photo()))["content"], "photo")
        self.assertEqual(analyse.analyse(samples.line_art())["content"], "text_or_diagram")
        self.assertEqual(analyse.analyse(samples.faded_scan(samples.line_art()))["content"], "text_or_diagram")

    def test_noise_estimate_tracks_added_noise_and_ignores_texture(self):
        flat = np.full((300, 300), 128, np.uint8)
        self.assertLess(analyse.estimate_noise(flat), 0.5)
        self.assertAlmostEqual(analyse.estimate_noise(samples.add_noise(flat, 8)), 8, delta=1.0)
        self.assertLess(analyse.analyse(samples.good_photo())["noise"], 1.5)

    def test_blur_lowers_sharpness(self):
        img = samples.photo()
        self.assertLess(analyse.analyse(samples.blur(img, 2.5))["sharpness"], analyse.analyse(img)["sharpness"] / 5)

    def test_bilevel_detection(self):
        bw = (analyse.to_gray8(samples.line_art()) > 128).astype(np.uint8) * 255
        self.assertTrue(analyse.analyse(bw)["bilevel"])
        self.assertFalse(analyse.analyse(samples.line_art())["bilevel"])

    def test_tiled_denoise_matches_whole_image(self):
        noisy = samples.add_noise(samples.photo(300, 400), 12)
        op = {"op": "denoise", "h": 8}
        whole = enhance.apply(noisy, [op])
        original_tile = enhance.DENOISE_TILE
        enhance.DENOISE_TILE = 128
        try:
            tiled = enhance.apply(noisy, [op])
        finally:
            enhance.DENOISE_TILE = original_tile
        self.assertTrue(np.array_equal(whole, tiled))

    def test_compare_identical_and_denoised(self):
        img = samples.photo()
        same = analyse.compare(img, img)
        self.assertGreater(same["ssim"], 0.999)
        self.assertEqual(same["newEdgeRatio"], 0)
        # Removing noise is not a structural change.
        noisy = samples.add_noise(img, 12)
        result = enhance.apply(noisy, [{"op": "denoise", "h": 8}])
        self.assertGreater(analyse.compare(noisy, result, analyse.analyse(noisy)["noise"])["ssim"], 0.85)


class PlanTests(unittest.TestCase):
    settings = dict(enhance.DEFAULT_SETTINGS)

    def ops(self, img, **overrides):
        return [o["op"] for o in enhance.flatten(enhance.plan(analyse.analyse(img), self.settings | overrides))]

    def test_good_images_get_no_operations(self):
        self.assertEqual(self.ops(samples.good_photo(), strength="gentle"), [])
        self.assertEqual(self.ops(samples.line_art()), [])  # clean digital diagram: nothing to improve

    def test_standard_keeps_photos_faithful_and_strong_adds_clarity(self):
        # Local contrast changes how a photo looks (measured against the true picture it always costs accuracy):
        # only the stronger levels add it.
        self.assertNotIn("local_contrast", self.ops(samples.good_photo()))
        self.assertNotIn("local_contrast", self.ops(samples.good_photo(), strength="gentle"))
        self.assertIn("local_contrast", self.ops(samples.good_photo(), strength="strong"))

    def test_faded_photo_gets_its_colours_back_including_the_cast(self):
        photo = samples.photo(seed=4)
        cv2.rectangle(photo, (20, 20), (60, 60), (12, 10, 8), -1)        # a shadow ...
        cv2.rectangle(photo, (580, 20), (620, 60), (245, 247, 248), -1)  # ... and a highlight, as in real photos
        veil = np.array([60.0, 75.0, 72.0], np.float32)  # yellowed, raised blacks (an old faded print)
        faded = np.clip(photo.astype(np.float32) * np.array([0.52, 0.6, 0.62], np.float32) + veil, 0, 255).astype(np.uint8)
        ops = enhance.plan(analyse.analyse(faded), self.settings)
        tone = next(o for o in ops if o["op"] == "contrast")
        self.assertEqual(tone["method"], "levels")
        out = enhance.apply(faded, [tone])
        stretched = enhance.apply(faded, [tone | {"method": "stretch", "low": int(np.min(tone["low"])), "high": int(np.max(tone["high"]))}])

        def cast(img):  # how differently the three colour channels are off
            offset = img.astype(np.float32).mean((0, 1)) - photo.astype(np.float32).mean((0, 1))
            return float(offset.max() - offset.min())

        def error(img):
            return float(np.abs(img.astype(np.float32) - photo).mean())

        self.assertLess(cast(out), cast(faded) / 3)        # the colour cast is gone ...
        self.assertGreater(cast(stretched), cast(faded) / 2)  # ... which a lightness stretch alone does not do
        self.assertLess(error(out), error(stretched) / 2)
        # A colourful photo that is not faded keeps its own colours.
        self.assertFalse(any(o.get("method") == "levels" for o in enhance.plan(analyse.analyse(photo), self.settings)))

    def test_cleaning_a_chart_keeps_its_bars_grid_lines_and_colours(self):
        """Regression (bar charts): the paper cleaning washed wide bars out to white, darkened coloured lines and
        erased the faint grid lines."""
        chart = np.full((420, 640, 3), 228, np.uint8)  # grey scanned paper: cleaned
        for y in range(60, 360, 50):
            cv2.line(chart, (60, y), (600, y), (200, 200, 200), 1)  # faint grid lines
        cv2.rectangle(chart, (100, 110), (190, 359), (178, 114, 0), -1)  # wide dark blue bar
        cv2.rectangle(chart, (260, 160), (350, 359), (40, 40, 40), -1)  # wide black bar
        cv2.rectangle(chart, (420, 210), (510, 359), (0, 159, 230), -1)  # orange bar
        cv2.line(chart, (60, 360), (600, 360), (0, 0, 0), 2)
        for k, label in enumerate(["A-101", "B-201", "C-301"]):
            cv2.putText(chart, label, (110 + 160 * k, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        scan = samples.add_noise(chart, 3)
        ops = self.ops(scan)
        self.assertIn("clean_background", ops)
        out = enhance.apply(scan, enhance.plan(analyse.analyse(scan), self.settings))
        for (x0, y0, x1, y1) in ((120, 150, 170, 340), (440, 240, 490, 340)):
            before = chart[y0:y1, x0:x1].reshape(-1, 3).mean(0)
            after = out[y0:y1, x0:x1].reshape(-1, 3).mean(0)
            self.assertLess(float(np.abs(after - before).max()), 25, (before, after))  # coloured bars keep their colour
        self.assertLess(float(out[200:340, 280:330].mean()), 50)  # the black bar stays black (ink may get darker)
        gray = analyse.to_gray8(out)
        paper = gray[70:100, 380:410]
        self.assertGreater(float(np.median(paper)), 240)                            # the paper is cleaned ...
        self.assertLess(float(np.median(gray[109:112, 380:410].min(0))), 225)       # ... the grid line stays

    def test_clean_digital_pictures_keep_their_look_at_every_level(self):
        """Regression (user's QOS document, Level 4): grey boxes turned white, a blue gradient got white streaks and
        lines got bolder in pictures that were already clean. Clean pictures keep colours, shading, lines and text;
        only measured blur is reversed."""
        page = np.full((420, 640, 3), 255, np.uint8)
        cv2.rectangle(page, (20, 20), (300, 200), (221, 221, 221), -1)              # grey box ...
        cv2.rectangle(page, (20, 20), (300, 200), (19, 69, 139), 3)                 # ... with a brown frame
        cv2.putText(page, "Step 1: Identify CQAs", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 30, 120), 2, cv2.LINE_AA)
        for x in range(340, 620):                                                    # light blue gradient box
            shade = 255 - int((x - 340) / 280 * 70)
            cv2.line(page, (x, 30), (x, 190), (255, shade, shade - 30), 1)
        cv2.putText(page, "Granulation", (380, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.line(page, (20, 300), (620, 300), (0, 0, 0), 2)
        analysis = analyse.analyse(page)
        self.assertTrue(analysis["clean"], analysis)
        for level in ("gentle", "standard", "strong", "maximum"):
            self.assertEqual(self.ops(page, strength=level), [], level)
        blurred = enhance.plan(analyse.analyse(cv2.GaussianBlur(page, (0, 0), 1.6)), self.settings | {"strength": "maximum"})
        names = [o["op"] for o in enhance.flatten(blurred)]
        self.assertIn("deblur", names)                                               # blur is reversed ...
        self.assertFalse({"clean_background", "crisp_text", "bolden", "smooth_speckle", "remove_specks", "contrast"} & set(names))
        # A scan (grain, uneven paper) or a faded picture is not clean.
        self.assertFalse(analyse.analyse(samples.poor_scan())["clean"])
        self.assertFalse(analyse.analyse(samples.faded_scan(samples.line_art()))["clean"])
        self.assertFalse(analyse.analyse(samples.low_contrast(page, 0.5, 100))["clean"])

    def test_sharp_chart_with_black_text_is_left_as_it_is(self):
        chart = np.full((400, 600, 3), 255, np.uint8)
        pts = np.array([[60, 300], [160, 120], [260, 200], [360, 90], [460, 250]], np.int32)
        cv2.polylines(chart, [pts], False, (180, 119, 31), 2, cv2.LINE_AA)
        cv2.line(chart, (50, 340), (560, 340), (0, 0, 0), 2)
        for k in range(5):
            cv2.putText(chart, str(10 * k), (55 + 100 * k, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        self.assertNotIn("clean_background", self.ops(chart))

    def test_noisy_photo_is_denoised(self):
        self.assertIn("denoise", self.ops(samples.add_noise(samples.good_photo(), 12)))

    def test_blurry_photo_is_sharpened(self):
        ops = self.ops(samples.blur(samples.good_photo(), 2))
        self.assertTrue("sharpen" in ops or "deblur" in ops, ops)

    def test_faded_scan_gets_clean_background_never_local_contrast(self):
        names = self.ops(samples.faded_scan(samples.line_art()))
        self.assertIn("clean_background", names)
        self.assertNotIn("local_contrast", names)
        gentle = enhance.plan(analyse.analyse(samples.faded_scan(samples.line_art())), self.settings | {"strength": "gentle"})
        self.assertTrue(all(o.get("method", "stretch") == "stretch" for o in gentle if o["op"] == "contrast"))

    def test_grainy_grey_page_gets_an_even_white_background(self):
        # Colour grain (every pixel randomly tinted) must not count as colour: the paper turns evenly white
        # instead of into salt-and-pepper dots, and the text stays.
        page = np.full((300, 500, 3), 170, np.uint8)
        cv2.putText(page, "Report 2026", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (90, 90, 90), 3, cv2.LINE_AA)
        grainy = samples.add_noise(page, 10)
        out = analyse.to_gray8(enhance.apply(grainy, enhance.plan(analyse.analyse(grainy), self.settings)))
        paper = out[190:290, 40:460]
        self.assertGreater(float(paper.mean()), 240)
        self.assertLess(float(paper.std()), 8)
        self.assertLess(float(out[80:125, 30:330].min()), 120)  # the text is still there

    def test_very_grainy_photo_gets_no_local_contrast(self):
        names = self.ops(samples.add_noise(samples.good_photo(), 25))
        self.assertIn("denoise", names)
        self.assertNotIn("local_contrast", names)

    def test_clean_background_whitens_paper_and_keeps_ink_dark(self):
        scan = samples.faded_scan(samples.line_art())
        out = enhance.apply(scan, [{"op": "clean_background"}])
        gray_in, gray_out = analyse.to_gray8(scan), analyse.to_gray8(out)
        ink = analyse.to_gray8(samples.line_art()) < 128
        paper = cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0  # away from (blurred) strokes
        self.assertGreater(float(gray_out[paper].mean()), 245)              # paper is now white
        self.assertLess(float(gray_out[ink].mean()), float(gray_in[ink].mean()))  # ink is darker
        self.assertEqual(out.shape, scan.shape)

    def test_deblur_brings_blurred_text_closer_to_the_sharp_original(self):
        """Measured against a known sharp original: blurred table text must come back closer to it."""
        sharp = np.full((420, 780), 255, np.uint8)
        for r, text in enumerate(["Inventory Dashboard", "Item  Stock  Unit", "Reagent A  120  pcs", "Item C  16  kg"]):
            cv2.putText(sharp, text, (20, 60 + r * 95), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 80, 2, cv2.LINE_AA)
            cv2.line(sharp, (10, 80 + r * 95), (770, 80 + r * 95), 120, 2)
        blurred = np.clip(cv2.GaussianBlur(sharp.astype(np.float32), (0, 0), 2.0)
                          + np.random.default_rng(0).normal(0, 2, sharp.shape), 0, 255).astype(np.uint8)
        analysis = analyse.analyse(blurred)
        self.assertAlmostEqual(analysis["blur"], 2.0, delta=0.5)
        ops = enhance.plan(analysis | {"autoUpscale": 1.0}, dict(enhance.DEFAULT_SETTINGS))
        self.assertIn("deblur", [o["op"] for o in enhance.flatten(ops)])
        out = cv2.resize(analyse.to_gray8(enhance.apply(blurred, ops)), (780, 420), interpolation=cv2.INTER_AREA)
        self.assertGreater(analyse.ssim(out, sharp), analyse.ssim(blurred, sharp) + 0.03)

    def test_crisp_text_keeps_coloured_strokes_colour(self):
        """Regression (seen when zooming in the app): a red arrow came out dark maroon after crisp-text."""
        img = np.full((200, 300, 3), 255, np.uint8)
        cv2.putText(img, "Text 42", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.line(img, (10, 140), (290, 140), (30, 30, 200), 5, cv2.LINE_AA)  # red line (BGR)
        img = cv2.GaussianBlur(img, (0, 0), 1.0)
        out = enhance.apply(img, [{"op": "crisp_text"}])
        red_in, red_out = img[138:143, 50:250].reshape(-1, 3).mean(0), out[138:143, 50:250].reshape(-1, 3).mean(0)
        self.assertLess(np.abs(red_out - red_in).max(), 6)                            # colour unchanged
        self.assertLess(float(analyse.to_gray8(out)[40:70, 10:200].min()), float(analyse.to_gray8(img)[40:70, 10:200].min()) + 1)

    def test_document_pipeline_keeps_thin_coloured_line_colour(self):
        """Regression: speckle smoothing washed a faded red arrow grey, and later steps then darkened it."""
        img = samples.faded_scan(samples.line_art())
        ops = enhance.plan(analyse.analyse(img) | {"autoUpscale": 1.0}, dict(enhance.DEFAULT_SETTINGS))
        out = enhance.apply(img, ops)
        f = out.shape[1] / img.shape[1]
        def strongest_chroma(im, scale):
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
            seg = lab[int(239 * scale):int(242 * scale), int(360 * scale):int(560 * scale)]
            return float(np.hypot(seg[:, :, 1] - 128, seg[:, :, 2] - 128).max())
        self.assertGreater(strongest_chroma(out, f), 0.8 * strongest_chroma(img, 1))

    def test_sharp_images_are_not_deblurred(self):
        self.assertLess(analyse.analyse(samples.line_art())["blur"], enhance.MIN_DEBLUR)

    def test_poor_scan_becomes_clean_white_page_with_dark_text(self):
        """Regression: an unevenly lit, yellowed scan was treated as a photo and came out grainy and grey."""
        scan = samples.poor_scan()
        analysis = analyse.analyse(scan)
        self.assertEqual(analysis["content"], "text_or_diagram")
        ops = enhance.plan(analysis, dict(enhance.DEFAULT_SETTINGS))
        self.assertIn("clean_background", [o["op"] for o in ops])
        out = enhance.apply(scan, ops)
        gray = analyse.to_gray8(out)
        self.assertGreater(float(np.percentile(gray, 60)), 245)        # paper is white
        self.assertLess(float(np.percentile(gray, 2)), float(np.percentile(analyse.to_gray8(scan), 2)))  # ink darker
        b, g, r = out[gray > 245].reshape(-1, 3).mean(axis=0)
        self.assertLess(abs(float(r) - float(b)), 4)                     # no yellow cast left on the paper

    def test_mixed_page_cleans_document_area_and_leaves_photo_alone(self):
        """Regression (user's test sheet): one 'photo' treatment for a page mixing photos and documents
        amplified the speckle; documents are now cleaned region by region and photos left as photos."""
        photo = samples.photo(300, 400, seed=2)
        scan = samples.poor_scan()[:, :400]  # native resolution, as scanned
        label = np.full((40, 800, 3), (225, 200, 250), np.uint8)  # pastel pink label strip (as on the user sheet)
        page = np.vstack([label, np.hstack([photo, scan])])
        analysis = analyse.analyse(page)
        self.assertTrue(enhance.MIXED_MIN_DOCUMENT <= analysis["documentFraction"] < 0.9)
        ops = enhance.plan(analysis | {"autoUpscale": 1.0}, dict(enhance.DEFAULT_SETTINGS))
        self.assertEqual(ops[0]["op"], "regional")
        out = enhance.apply(page, ops)
        doc_out, doc_in = analyse.to_gray8(out[40:, 420:]), analyse.to_gray8(page[40:, 420:])
        self.assertGreater(float(np.percentile(doc_out, 60)), 240)     # document paper cleaned to white
        self.assertLess(float(np.percentile(doc_out, 2)), float(np.percentile(doc_in, 2)) + 5)  # text kept dark
        photo_only = enhance.apply(page, ops[0]["photo"])
        diff = np.abs(out[40:, :380].astype(int) - photo_only[40:, :380].astype(int)).mean()
        self.assertLess(diff, 3.0)                                        # photo area got photo treatment only (soft edge)
        self.assertLess(np.abs(out[:30, 600:].astype(int) - page[:30, 600:].astype(int)).mean(), 6)  # label colour kept

    def test_bright_textured_photo_is_never_treated_as_paper(self):
        """Regression (real photo of white crystals): the pile became a flat white blob."""
        rng = np.random.default_rng(3)
        crystals = np.full((480, 640), 40, np.uint8)
        for _ in range(9000):
            x, y = int(rng.normal(320, 110)), int(rng.normal(250, 90))
            cv2.ellipse(crystals, (x, y), (int(rng.integers(2, 6)), int(rng.integers(1, 3))), float(rng.uniform(0, 180)),
                        0, 360, int(rng.integers(170, 255)), -1)
        img = cv2.cvtColor(cv2.GaussianBlur(crystals, (0, 0), 0.6), cv2.COLOR_GRAY2BGR)
        self.assertLess(analyse.analyse(img)["documentFraction"], enhance.MIXED_MIN_DOCUMENT)

    def test_low_resolution_document_image_gets_more_pixels(self):
        small = samples.faded_scan(samples.line_art(240, 320))
        settings = dict(enhance.DEFAULT_SETTINGS)
        # 320 px shown 4.4 inches wide = 72 dpi: upscaled; the same image shown 1 inch wide = 320 dpi: not.
        self.assertEqual(enhance.auto_upscale(320, 240, 3, 320.0, settings, enhance.DEFAULT_LIMITS), 2.0)
        self.assertEqual(enhance.auto_upscale(320, 240, 3, 72.0, settings, enhance.DEFAULT_LIMITS), 1.0)
        self.assertEqual(enhance.auto_upscale(320, 240, 3, None, settings, enhance.DEFAULT_LIMITS), 1.0)  # image files
        # Standalone image files: small ones get more pixels; large ones and Gentle strength do not.
        self.assertEqual(enhance.auto_upscale(800, 600, 3, None, settings, enhance.DEFAULT_LIMITS, standalone=True), 2.0)
        self.assertEqual(enhance.auto_upscale(1400, 1000, 3, None, settings, enhance.DEFAULT_LIMITS, standalone=True), 1.5)
        self.assertEqual(enhance.auto_upscale(4000, 3000, 3, None, settings, enhance.DEFAULT_LIMITS, standalone=True), 1.0)
        self.assertEqual(enhance.auto_upscale(800, 600, 3, None, settings | {"strength": "gentle"}, enhance.DEFAULT_LIMITS, standalone=True), 1.0)
        analysis = analyse.analyse(small) | {"autoUpscale": 2.0}
        names = [o["op"] for o in enhance.plan(analysis, settings | {"keepPixelSize": False})]
        detail = names.index("deblur") if "deblur" in names else names.index("sharpen")
        self.assertLess(names.index("resize"), detail)
        # By default the size is kept: the detail steps run on a finer grid inside "fine_grid" instead.
        plan = enhance.plan(analysis, settings)
        self.assertNotIn("resize", [o["op"] for o in plan])
        fine = next(o for o in plan if o["op"] == "fine_grid")
        self.assertEqual(fine["factor"], 2.0)
        self.assertTrue({"deblur", "sharpen"} & {o["op"] for o in fine["ops"]})

    def test_colour_diagram_on_dark_background_keeps_its_colours(self):
        """Regression (real Wikipedia chemistry figure): blue/red atoms on black must not be brightened."""
        img = np.zeros((300, 400, 3), np.uint8)
        for k, colour in enumerate([(200, 40, 20), (30, 30, 200), (200, 40, 20), (30, 30, 200)]):
            cv2.circle(img, (60 + k * 90, 150), 22, colour, -1, cv2.LINE_AA)
            cv2.line(img, (60 + k * 90, 150), (150 + k * 90, 150), (90, 90, 90), 4, cv2.LINE_AA)
        a = analyse.analyse(img)
        self.assertEqual(a["content"], "text_or_diagram")
        self.assertNotIn("contrast", self.ops(img))

    def test_fine_texture_is_not_mistaken_for_noise(self):
        """Regression (real photo of crystals): dense fine texture must not be smoothed away."""
        rng = np.random.default_rng(5)
        texture = cv2.resize(rng.integers(60, 230, (240, 320), dtype=np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(cv2.GaussianBlur(texture, (0, 0), 0.7), cv2.COLOR_GRAY2BGR)
        self.assertNotIn("denoise", self.ops(img))
        self.assertIn("denoise", self.ops(samples.add_noise(samples.photo(), 12)))

    def test_switches_disable_operations(self):
        noisy = samples.add_noise(samples.blur(samples.good_photo(), 2), 12)
        self.assertEqual(self.ops(noisy, denoise=False, sharpen=False, contrast=False), [])

    def test_manual_mode_uses_exact_values(self):
        plan = enhance.plan(analyse.analyse(samples.good_photo()),
                            self.settings | {"mode": "manual", "denoiseStrength": 4, "contrastStrength": 2, "sharpenAmount": 0.5})
        self.assertEqual([o["op"] for o in plan], ["denoise", "contrast", "sharpen"])
        self.assertEqual(plan[0]["h"], 4)

    def test_resize_goes_before_sharpen(self):
        plan = enhance.plan(analyse.analyse(samples.blur(samples.good_photo(), 2)),
                            self.settings | {"resizeFactor": 2, "keepPixelSize": False})
        names = [o["op"] for o in plan]
        detail = names.index("deblur") if "deblur" in names else names.index("sharpen")
        self.assertLess(names.index("resize"), detail)



def _strokes():
    """Thin strokes on paper: a double bond (2-pixel gap), a ring with a hole, a thick block and a red line."""
    img = np.full((120, 200, 3), 250, np.uint8)
    cv2.line(img, (10, 40), (190, 40), (40, 40, 40), 1)
    cv2.line(img, (10, 43), (190, 43), (40, 40, 40), 1)
    cv2.circle(img, (120, 80), 6, (40, 40, 40), 1)
    cv2.rectangle(img, (10, 60), (60, 100), (30, 30, 30), -1)
    cv2.line(img, (150, 60), (190, 110), (0, 0, 200), 1)
    return img


class LevelTests(unittest.TestCase):
    settings = dict(enhance.DEFAULT_SETTINGS)

    def plan_at(self, img, level, standalone=True, **overrides):
        settings = self.settings | {"strength": level} | overrides
        a = analyse.analyse(img)
        a["autoUpscale"] = enhance.auto_upscale(img.shape[1], img.shape[0], 3, None, settings, enhance.DEFAULT_LIMITS,
                                                standalone=standalone)
        return enhance.flatten(enhance.plan(a, settings))

    def test_each_level_does_more_than_the_one_before(self):
        img = samples.poor_scan()
        names = {level: [o["op"] for o in self.plan_at(img, level)] for level in ("gentle", "standard", "strong", "maximum")}
        self.assertNotIn("bolden", names["standard"])
        self.assertIn("bolden", names["strong"])
        self.assertIn("bolden", names["maximum"])
        self.assertIn("crisp_text", names["maximum"])
        amounts = [next((o["amount"] for o in self.plan_at(img, level) if o["op"] == "bolden"), 0)
                   for level in ("standard", "strong", "maximum")]
        self.assertEqual(amounts, sorted(amounts))
        self.assertGreater(len(names["standard"]), len(names["gentle"]))

    def test_maximum_adds_pixels_to_larger_images_too(self):
        settings = self.settings | {"strength": "maximum"}
        self.assertEqual(enhance.auto_upscale(1800, 1200, 3, None, settings, enhance.DEFAULT_LIMITS, standalone=True), 2.0)
        self.assertEqual(enhance.auto_upscale(1800, 1200, 3, None, self.settings | {"strength": "strong"},
                                              enhance.DEFAULT_LIMITS, standalone=True), 1.5)

    def test_good_images_stay_untouched_at_every_level(self):
        for level in ("gentle", "standard", "strong", "maximum"):
            # (a small standalone file would still get more pixels; here it is a sharp image inside a document)
            self.assertEqual(self.plan_at(samples.line_art(), level, standalone=False), [], level)

    def test_bold_text_switch_overrides_the_level(self):
        img = samples.poor_scan()
        self.assertNotIn("bolden", [o["op"] for o in self.plan_at(img, "maximum", boldText=False)])
        self.assertIn("bolden", [o["op"] for o in self.plan_at(img, "standard", boldText=True)])
        # Photos never get bolder "text".
        self.assertNotIn("bolden", [o["op"] for o in self.plan_at(samples.add_noise(samples.photo(), 10), "maximum", boldText=True)])


class BoldenTests(unittest.TestCase):
    def setUp(self):
        self.img = _strokes()
        self.out = enhance.apply(self.img, [{"op": "bolden", "amount": 1.0, "thick": 2.5, "gap": 3}])
        self.before = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) < 150
        self.after = cv2.cvtColor(self.out, cv2.COLOR_BGR2GRAY) < 150

    def test_thin_strokes_get_thicker(self):
        thin_before = self.before.sum() - self.before[60:101, 10:61].sum()
        self.assertGreater(self.after.sum() - self.after[60:101, 10:61].sum(), thin_before * 1.5)
        self.assertGreater(self.after[:, 100].sum(), self.before[:, 100].sum())

    def test_gaps_and_holes_stay_open(self):
        self.assertFalse(self.after[41:43, 100].any())  # double bond: both lines stay separate
        self.assertFalse(self.after[77:84, 117:124].any())  # ring: the hole stays open
        self.assertEqual(cv2.connectedComponents(self.after.astype(np.uint8))[0],
                         cv2.connectedComponents(self.before.astype(np.uint8))[0])

    def test_thick_strokes_and_paper_are_unchanged(self):
        self.assertTrue(np.array_equal(self.out[62:99, 12:59], self.img[62:99, 12:59]))
        self.assertTrue(np.array_equal(self.out[5:30, 70:110], self.img[5:30, 70:110]))

    def test_coloured_strokes_keep_their_colour_and_nothing_gets_lighter(self):
        grown = self.after & ~self.before
        b, g, r = (self.out[grown][:, i].astype(int) for i in range(3))
        red = grown[60:111, 150:191].sum()
        self.assertGreater(red, 0)
        region = self.out[60:111, 150:191][grown[60:111, 150:191]].astype(int)
        self.assertTrue(np.all(region[:, 2] > region[:, 0] + 100))
        self.assertTrue(np.all(self.out.astype(int) <= self.img.astype(int)))


class EnhanceCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_enhance(self, input_path, output_name="out", **args):
        ext = Path(input_path).suffix
        request = {
            "input": str(input_path),
            "output": str(self.dir / (output_name + ext)),
            "thumbnails": {"original": str(self.dir / "t_orig.jpg"), "enhanced": str(self.dir / "t_enh.jpg"), "maxSide": 128},
        } | args
        return enhance.enhance_cmd(request, None)

    def test_noisy_photo_completes_same_size_and_less_noise(self):
        src = samples.write(self.dir / "noisy.png", samples.add_noise(samples.good_photo(), 12))
        result = self.run_enhance(src)

        self.assertEqual(result["status"], "completed")
        self.assertIn("denoise", [o["op"] for o in result["operations"]])
        out = image_io.decode(result["output"]["path"])
        self.assertEqual(out.shape, cv2.imread(src, cv2.IMREAD_UNCHANGED).shape)
        self.assertLess(analyse.analyse(out)["noise"], analyse.analyse(image_io.decode(src))["noise"] / 2)
        self.assertLessEqual(max(cv2.imread(str(self.dir / "t_enh.jpg")).shape[:2]), 128)
        self.assertGreater(result["memory"]["peakRssBytes"], 0)

    def test_result_reports_quality_before_and_after(self):
        src = samples.write(self.dir / "noisy.png", samples.add_noise(samples.good_photo(), 12))
        result = self.run_enhance(src)
        for key in ("sharpness", "noise", "blur", "dynamicRange"):
            self.assertIn(key, result["analysis"])
            self.assertIn(key, result["after"])
        self.assertLess(result["after"]["noise"], result["analysis"]["noise"])
        self.assertIn("ssim", result["metrics"])

    def test_original_file_is_never_modified(self):
        src = self.dir / "noisy.jpg"
        samples.write(src, samples.add_noise(samples.photo(), 12))
        before = src.read_bytes()
        self.run_enhance(src)
        self.assertEqual(src.read_bytes(), before)

    def test_already_good_image_is_skipped_without_output(self):
        src = samples.write(self.dir / "good.png", samples.line_art())
        result = self.run_enhance(src)
        self.assertEqual((result["status"], result["reason"]), ("skipped", "already_good"))
        self.assertFalse((self.dir / "out.png").exists())
        self.assertTrue((self.dir / "t_orig.jpg").exists())  # gallery still gets a preview

    def test_diagram_changes_always_need_review(self):
        src = samples.write(self.dir / "scan.png", samples.faded_scan(samples.line_art()))
        result = self.run_enhance(src)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["needsReview"])
        self.assertIn("text_or_diagram", result["reviewReasons"])

    def test_alpha_channel_is_preserved_exactly(self):
        alpha = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
        rgba = np.dstack([samples.add_noise(samples.good_photo(), 12), alpha])
        src = samples.write(self.dir / "alpha.png", rgba)
        result = self.run_enhance(src)
        out = image_io.decode(result["output"]["path"])
        self.assertEqual(out.shape, (480, 640, 4))
        self.assertTrue(np.array_equal(out[:, :, 3], alpha))

    def test_resize_factor_scales_output_only_when_allowed(self):
        src = samples.write(self.dir / "small.png", samples.blur(samples.photo(100, 150), 2))
        result = self.run_enhance(src, settings={"resizeFactor": 2, "keepPixelSize": False})
        self.assertEqual((result["output"]["width"], result["output"]["height"]), (300, 200))
        result = self.run_enhance(src, settings={"resizeFactor": 2})
        self.assertEqual((result["output"]["width"], result["output"]["height"]), (150, 100))

    def test_every_level_keeps_the_exact_pixel_size(self):
        """Low-resolution, blurred text at every level: same width and height out, and the fine grid is used."""
        scan = cv2.resize(samples.poor_scan(), (333, 201), interpolation=cv2.INTER_AREA)
        src = samples.write(self.dir / "small_scan.png", scan)
        for level in ("gentle", "standard", "strong", "maximum"):
            result = self.run_enhance(src, settings={"strength": level, "boldText": True}, standalone=True)
            if result["status"] == "completed":
                self.assertEqual((result["output"]["width"], result["output"]["height"]), (333, 201), level)
                self.assertNotIn("resize", [o["op"] for o in result["operations"]], level)
        self.assertIn("fine_grid", [o["op"] for o in result["operations"]])

    def test_jpeg_output_keeps_exif_orientation(self):
        src = self.dir / "rotated.jpg"
        src.write_bytes(jpeg_with_orientation(samples.add_noise(samples.photo(120, 200), 12), 6))
        result = self.run_enhance(src)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(image_io.jpeg_orientation(Path(result["output"]["path"]).read_bytes()), 6)
        thumb = cv2.imread(str(self.dir / "t_enh.jpg"))
        self.assertGreater(thumb.shape[0], thumb.shape[1])  # preview shows the image upright

    def test_gif_is_written_as_png(self):
        src = samples.write(self.dir / "anim.gif", samples.add_noise(samples.photo(120, 160), 12))
        result = enhance.enhance_cmd({"input": src, "output": str(self.dir / "out.png")}, None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["output"]["format"], "png")
        self.assertEqual(image_io.probe(result["output"]["path"]).format, "png")
        self.assertIn("saved as PNG instead of GIF", result["notes"])

    def test_multipage_tiff_frame_enhance_and_reassemble(self):
        pages = [samples.add_noise(samples.photo(100, 120, seed=s), 12) for s in (1, 2, 3)]
        ok, buf = cv2.imencodemulti(".tif", pages)
        src = self.dir / "multi.tif"
        buf.tofile(str(src))

        result = self.run_enhance(src, output_name="page1", frame=1)
        self.assertEqual(result["status"], "completed")
        assembled = self.dir / "assembled.tif"
        enhance.assemble_tiff_cmd({
            "frames": [{"path": str(src), "frame": 0}, {"path": result["output"]["path"]}, {"path": str(src), "frame": 2}],
            "output": str(assembled),
            "source": str(src),
        }, type("Ctx", (), {"progress": lambda self, **k: None})())
        self.assertEqual(image_io.probe(str(assembled)).frames, 3)
        self.assertTrue(np.array_equal(image_io.decode(str(assembled), 0), pages[0]))  # untouched page unchanged

    def test_skips(self):
        cases = {
            "too_large": (samples.write(self.dir / "big.png", samples.photo(200, 300)), {"limits": {"maxPixels": 10_000}}),
            "over_memory": (samples.write(self.dir / "big2.png", samples.photo(200, 300)), {"limits": {"maxMemoryBytes": 100_000}}),
            "bilevel": (samples.write(self.dir / "bw.png", ((analyse.to_gray8(samples.line_art()) > 128) * 255).astype(np.uint8)), {}),
            "high_bit_depth": (samples.write(self.dir / "deep.png", (samples.photo().astype(np.uint16) * 257)), {}),
            "disabled": (samples.write(self.dir / "off.png", samples.photo()), {"settings": {"mode": "off"}}),
        }
        expected_reason = {"over_memory": "too_large"}
        for name, (path, args) in cases.items():
            with self.subTest(name):
                result = self.run_enhance(path, output_name=name, **args)
                self.assertEqual(result["status"], "skipped")
                self.assertEqual(result["reason"], expected_reason.get(name, name))
                self.assertTrue(result["message"])

    def test_vector_and_heic_skipped(self):
        (self.dir / "d.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        (self.dir / "p.heic").write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 16)
        self.assertEqual(self.run_enhance(self.dir / "d.svg")["reason"], "vector")
        self.assertEqual(self.run_enhance(self.dir / "p.heic")["reason"], "unsupported_format")

    def test_corrupt_image_is_an_error(self):
        ok, buf = cv2.imencode(".jpg", samples.photo())
        (self.dir / "bad.jpg").write_bytes(buf.tobytes()[:400])
        with self.assertRaises(WorkerError) as ctx:
            self.run_enhance(self.dir / "bad.jpg")
        self.assertEqual(ctx.exception.code, "decode_failed")

    def test_probe_reports_memory_estimate(self):
        src = samples.write(self.dir / "p.png", samples.photo(100, 200))
        result = enhance.probe_cmd({"path": src}, None)
        self.assertEqual((result["width"], result["height"], result["outputFormat"]), (200, 100, "png"))
        self.assertGreater(result["estimatedMemoryBytes"], 100 * 200 * 3)

    def test_thumbnail_command_for_oversized_jpeg(self):
        src = samples.write(self.dir / "big.jpg", samples.photo(400, 800))
        out = self.dir / "thumb.jpg"
        result = enhance.thumbnail_cmd({"input": src, "output": str(out), "limits": {"maxPixels": 1000}}, None)
        self.assertTrue(out.exists())
        self.assertLessEqual(max(result["width"], result["height"]), 256)

    def test_errors_go_through_protocol_without_killing_worker(self):
        out = io.StringIO()
        handle_line(json.dumps({"id": "1", "cmd": "enhance", "args": {"input": str(self.dir / "missing.png"), "output": str(self.dir / "o.png")}}), out)
        message = json.loads(out.getvalue())
        self.assertFalse(message["ok"])
        self.assertEqual(message["error"]["code"], "file_not_found")


if __name__ == "__main__":
    unittest.main()
