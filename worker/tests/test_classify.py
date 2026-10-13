"""Picture classification (DocumentFigureClassifier) and MolScribe's second check for chemical diagrams."""
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

import enhance  # noqa: E402
import figure_classifier as fc  # noqa: E402
import samples  # noqa: E402

CLASSIFIER = fc.available()


def probs(**values) -> np.ndarray:
    """A probability row with the given labels (the rest spread evenly)."""
    row = np.zeros(len(fc.LABELS))
    for label, value in values.items():
        row[fc.LABELS.index(label)] = value
    rest = 1.0 - row.sum()
    empty = row == 0
    row[empty] = rest / empty.sum()
    return row


def line_chart() -> np.ndarray:
    img = np.full((420, 640, 3), 255, np.uint8)
    cv2.line(img, (60, 360), (600, 360), (0, 0, 0), 2)
    cv2.line(img, (60, 360), (60, 40), (0, 0, 0), 2)
    pts = np.array([[60, 330], [150, 250], [240, 270], [330, 160], [420, 190], [510, 90], [600, 120]], np.int32)
    cv2.polylines(img, [pts], False, (180, 90, 20), 3, cv2.LINE_AA)
    for k, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x), int(y)), 5, (180, 90, 20), -1)
        cv2.putText(img, str(k * 2), (int(x) - 6, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for v in range(0, 60, 10):
        cv2.putText(img, str(v), (25, 360 - v * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Dissolution (%) vs time (min)", (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def framed(img: np.ndarray, size=(380, 300)) -> np.ndarray:
    tile = np.full((size[1], size[0], 3), 255, np.uint8)
    small = cv2.resize(img, (size[0] - 24, size[1] - 24), interpolation=cv2.INTER_AREA)
    tile[12:-12, 12:-12] = small
    cv2.rectangle(tile, (2, 2), (size[0] - 3, size[1] - 3), (60, 60, 60), 3)
    return tile


class SuggestionTests(unittest.TestCase):
    """How the 26 figure types become the app's four groups (no model needed)."""

    def test_figure_types_map_to_the_four_groups(self):
        self.assertEqual(fc.suggest(probs(chemistry_structure=0.97))["category"], fc.CHEMICAL)
        self.assertEqual(fc.suggest(probs(table=0.95))["category"], fc.TABLE)
        for label in ("flow_chart", "line_chart", "bar_chart", "pie_chart", "scatter_plot", "box_plot"):
            self.assertEqual(fc.suggest(probs(**{label: 0.9}))["category"], fc.GRAPH, label)
        for label in ("photograph", "logo", "screenshot_from_computer", "signature", "geographical_map"):
            self.assertEqual(fc.suggest(probs(**{label: 0.9}))["category"], fc.OTHER, label)

    def test_a_group_counts_all_its_figure_types(self):
        # 0.45 line chart + 0.40 bar chart is a graph at 0.85, although "table" (0.15) is no single type's rival.
        s = fc.suggest(probs(line_chart=0.45, bar_chart=0.40, table=0.15))
        self.assertEqual(s["category"], fc.GRAPH)
        self.assertAlmostEqual(s["confidence"], 0.85, places=2)
        self.assertEqual(s["label"], "line_chart")
        self.assertFalse(s["uncertain"])

    def test_a_picture_that_looks_chemical_stays_with_the_chemical_diagrams(self):
        s = fc.suggest(probs(chemistry_structure=0.38, photograph=0.30, logo=0.30))
        self.assertEqual(s["category"], fc.CHEMICAL)
        self.assertTrue(s["uncertain"])  # 38 % sure: the user should check it

    def test_uncertain_suggestions_are_marked(self):
        self.assertTrue(fc.suggest(probs(table=0.5, photograph=0.5))["uncertain"])
        self.assertFalse(fc.suggest(probs(table=0.9))["uncertain"])

    def test_groups_and_panel_categories(self):
        result = {"category": fc.OTHER, "panels": [{"box": [0, 0, 10, 10], "category": fc.CHEMICAL},
                                                   {"box": [10, 0, 20, 10], "category": fc.TABLE}]}
        self.assertEqual(fc.groups(result), [fc.OTHER, fc.CHEMICAL, fc.TABLE])
        self.assertEqual(fc.panel_category(result, (0, 0, 10, 10)), fc.CHEMICAL)
        self.assertEqual(fc.panel_category(result, (50, 50, 60, 60)), fc.OTHER)  # not listed: the picture's own
        self.assertIsNone(fc.panel_category(None, (0, 0, 10, 10)))


class ClassifyLogicTests(unittest.TestCase):
    """Panels, and where MolScribe runs (model replaced by fixed probabilities)."""

    def setUp(self):
        self.img = np.full((300, 600, 3), 255, np.uint8)
        self.boxes = [(10, 10, 290, 290), (310, 10, 590, 290)]

    def test_confident_single_figure_is_not_split_into_panels(self):
        rows = np.array([probs(table=0.97), probs(chemistry_structure=0.9), probs(photograph=0.9)])
        with patch.object(fc, "probabilities", return_value=rows), patch.object(fc, "chemical_check") as check:
            result = fc.classify(self.img, self.boxes)
        self.assertEqual(result["category"], fc.TABLE)
        self.assertEqual(result["panels"], [])
        self.assertIn("cells of a table", result["panelsNote"])
        check.assert_not_called()  # a table cell is never read by MolScribe

    def test_sheet_of_figures_has_each_panel_classified_and_only_chemical_ones_checked(self):
        rows = np.array([probs(table=0.45, line_chart=0.35), probs(chemistry_structure=0.92), probs(photograph=0.95)])
        with patch.object(fc, "probabilities", return_value=rows), \
                patch.object(fc, "chemical_check", return_value={"status": "read"}) as check:
            result = fc.classify(self.img, self.boxes)
        self.assertEqual([p["category"] for p in result["panels"]], [fc.CHEMICAL, fc.OTHER])
        self.assertEqual(check.call_count, 1)                       # the structure panel only
        self.assertEqual(check.call_args[0][0].shape[:2], (280, 280))
        self.assertEqual(result["panels"][0]["chemicalCheck"], {"status": "read"})
        self.assertNotIn("chemicalCheck", result["panels"][1])
        self.assertNotIn("chemicalCheck", result)                   # the sheet itself is not chemical
        self.assertEqual(fc.groups(result), [fc.TABLE, fc.CHEMICAL, fc.OTHER])

    def test_panels_of_the_same_kind_as_the_picture_are_one_figure(self):
        # An unsure whole picture whose framed areas are all tables: the cells of one table, not a sheet.
        rows = np.array([probs(table=0.65, calendar=0.3), probs(table=0.95), probs(table=0.9)])
        with patch.object(fc, "probabilities", return_value=rows):
            result = fc.classify(self.img, self.boxes)
        self.assertEqual(result["panels"], [])
        self.assertIn("parts of one picture", result["panelsNote"])
        self.assertEqual(fc.groups(result), [fc.TABLE])

    def test_molscribe_never_runs_on_tables_graphs_or_photos(self):
        for row in (probs(table=0.95), probs(flow_chart=0.95), probs(photograph=0.95), probs(logo=0.9)):
            with patch.object(fc, "probabilities", return_value=np.array([row])), \
                    patch.object(fc, "chemical_check") as check:
                result = fc.classify(self.img)
            check.assert_not_called()
            self.assertNotIn("chemicalCheck", result)

    def test_chemical_picture_gets_the_second_check(self):
        with patch.object(fc, "probabilities", return_value=np.array([probs(chemistry_structure=0.99)])), \
                patch.object(fc, "chemical_check", return_value={"status": "uncertain", "reasons": ["x"]}) as check:
            result = fc.classify(self.img)
        check.assert_called_once()
        self.assertEqual(result["category"], fc.CHEMICAL)  # an unreadable chemical picture stays chemical
        self.assertEqual(result["chemicalCheck"]["status"], "uncertain")


class ChemicalCheckTests(unittest.TestCase):
    """MolScribe's reading is a second check only: unreliable readings are reported, never hidden or 'proven'."""

    def reading(self, confidence=0.9, atom_confidence=0.9, bonds=True):
        atoms = [{"symbol": "C", "x": 50.0 + 60 * k, "y": 100.0, "confidence": atom_confidence} for k in range(3)]
        reading = {"atoms": atoms, "bonds": [{"atoms": (0, 1), "type": "single"}, {"atoms": (1, 2), "type": "single"}] if bonds else [],
                   "confidence": confidence, "smiles": "CCC"}
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles("CCC")
            AllChem.Compute2DCoords(mol)
            reading["molfile"] = Chem.MolToMolBlock(mol)  # what MolScribe returns with its reading
        except ImportError:
            pass
        return reading

    def check_with(self, reading):
        img = np.full((200, 400, 3), 255, np.uint8)
        cv2.line(img, (50, 100), (110, 100), (0, 0, 0), 3)
        cv2.line(img, (110, 100), (170, 100), (0, 0, 0), 3)
        from redraw import molscribe_reader, pipeline
        with patch.object(molscribe_reader, "available", return_value=True), \
                patch.object(molscribe_reader, "read", return_value=reading), \
                patch.object(pipeline, "read_labels", return_value=[]):
            return fc.chemical_check(img)

    def test_unsure_reading_is_uncertain_with_the_reason(self):
        result = self.check_with(self.reading(confidence=0.55))
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("55 %", result["reasons"][0])
        self.assertNotIn("formula", result)

    def test_reading_without_bonds_is_uncertain(self):
        result = self.check_with(self.reading(bonds=False))
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("no atoms and bonds", result["reasons"][0])

    def test_confident_reading_of_drawn_bonds_is_read(self):
        reading = self.reading()
        if "molfile" not in reading:
            self.skipTest("RDKit not installed")
        result = self.check_with(reading)
        self.assertEqual(result["status"], "read", result)
        self.assertEqual((result["atoms"], result["bonds"]), (3, 2))
        self.assertEqual(result["formula"], "C3H8")

    def test_missing_molscribe_is_reported_not_guessed(self):
        from redraw import molscribe_reader
        with patch.object(molscribe_reader, "available", return_value=False):
            result = fc.chemical_check(np.full((50, 50, 3), 255, np.uint8))
        self.assertEqual(result["status"], "unavailable")


@unittest.skipUnless(CLASSIFIER, "the picture classifier is not installed (worker/setup_ai.py)")
class ModelTests(unittest.TestCase):
    """The real DocumentFigureClassifier on drawn examples of each group."""

    def test_each_kind_of_picture_lands_in_its_group(self):
        from test_redraw import ordinary_table
        cases = [(ordinary_table(), fc.TABLE), (line_chart(), fc.GRAPH), (samples.photo(), fc.OTHER)]
        try:
            from test_redraw import drawing
            cases.append((drawing("CC(C)(S)[C@@H](N)C(=O)O"), fc.CHEMICAL))
        except Exception:  # RDKit missing: the chemical example is skipped
            pass
        for img, expected in cases:
            with self.subTest(expected=expected):
                result = fc.classify(img, check_chemistry=False)
                self.assertEqual(result["category"], expected, result["top"])
                self.assertGreater(result["confidence"], 0.5)

    def test_classify_command_reads_the_extracted_file_and_never_changes_it(self):
        from test_redraw import ordinary_table
        with tempfile.TemporaryDirectory() as tmp:
            path = samples.write(Path(tmp) / "table.png", ordinary_table())
            before = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            result = enhance.classify_cmd({"input": path}, None)
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), before)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["table.png"])  # nothing written
        self.assertTrue(result["available"])
        self.assertEqual(result["category"], fc.TABLE)
        self.assertEqual(result["model"], fc.MODEL_NAME)
        self.assertNotIn("chemicalCheck", result)

    def test_vector_pictures_are_not_classified_but_listed_as_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drawing.svg"
            path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>', encoding="utf-8")
            result = enhance.classify_cmd({"input": str(path)}, None)
        self.assertEqual(result["category"], fc.OTHER)
        self.assertIsNone(result["confidence"])
        self.assertIn("vector", result["note"])

    def test_sheet_with_a_structure_and_a_photo_is_classified_panel_by_panel(self):
        try:
            from test_redraw import drawing
            structure = drawing("CC(C)(S)[C@@H](N)C(=O)O")
        except Exception:
            self.skipTest("RDKit not installed")
        from test_redraw import ordinary_table
        sheet = np.full((720, 1240, 3), 255, np.uint8)
        sheet[20:320, 20:400] = framed(structure)
        sheet[20:320, 430:810] = framed(samples.photo())
        sheet[20:320, 840:1220] = framed(line_chart())
        sheet[380:680, 20:400] = framed(ordinary_table())
        import panels
        boxes = panels.detect(sheet)
        self.assertEqual(len(boxes), 4)
        result = fc.classify(sheet, boxes, check_chemistry=False)
        self.assertEqual([p["category"] for p in result["panels"]], [fc.CHEMICAL, fc.OTHER, fc.GRAPH, fc.TABLE],
                         [(p["label"], p["confidence"]) for p in result["panels"]])


class EnhanceGateTests(unittest.TestCase):
    """Redraw (and so MolScribe) only runs on pictures the app classified as chemical."""

    def test_non_chemical_pictures_never_reach_molscribe(self):
        try:
            from test_redraw import drawing
            img = drawing("CC(=O)Oc1ccccc1C(=O)O")
        except Exception:
            self.skipTest("RDKit not installed")
        from redraw import molscribe_reader
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "s.png", img)
            for category in (fc.TABLE, fc.GRAPH, fc.OTHER, "unclassified"):
                with self.subTest(category=category), \
                        patch.object(molscribe_reader, "read", side_effect=AssertionError("MolScribe ran")):
                    result = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "o.png"),
                                                  "settings": {"redraw": True, "redrawAll": True},
                                                  "classification": {"category": category}}, None)
                    self.assertNotIn("redraw", [o["op"] for o in result.get("operations", [])])
                    self.assertFalse(result.get("redraw"))

    def test_panels_follow_their_own_category(self):
        img = np.full((300, 600, 3), 255, np.uint8)
        calls = []

        def fake_redraw(crop, analysis, settings, chemical=None):
            calls.append(chemical)
            return {}

        classification = {"category": fc.OTHER, "panels": [{"box": [10, 10, 290, 290], "category": fc.CHEMICAL},
                                                          {"box": [310, 10, 590, 290], "category": fc.TABLE}]}
        with patch.object(enhance, "_try_redraw", side_effect=fake_redraw):
            enhance._enhance_panels(img, [(10, 10, 290, 290), (310, 10, 590, 290)], dict(enhance.DEFAULT_SETTINGS) | {"redraw": True},
                                    dict(enhance.DEFAULT_LIMITS), {}, classification)
        self.assertEqual(calls, [True, False])


if __name__ == "__main__":
    unittest.main()
