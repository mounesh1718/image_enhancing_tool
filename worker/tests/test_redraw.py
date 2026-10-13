import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import enhance  # noqa: E402
import samples  # noqa: E402
from redraw import pipeline  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
AVAILABLE = pipeline.available()


def drawing(smiles: str, size=(520, 400)) -> np.ndarray:
    """A structure drawing made by RDKit (black and white), so the right answer is known."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    canvas = rdMolDraw2D.MolDraw2DCairo(*size)
    canvas.drawOptions().useBWAtomPalette()
    canvas.drawOptions().bondLineWidth = 2
    canvas.DrawMolecule(rdMolDraw2D.PrepareMolForDrawing(Chem.MolFromSmiles(smiles)))
    canvas.FinishDrawing()
    return cv2.imdecode(np.frombuffer(canvas.GetDrawingText(), np.uint8), cv2.IMREAD_COLOR)


def inchikey(smiles: str) -> str:
    from rdkit import Chem
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def ordinary_table() -> np.ndarray:
    image = np.full((500, 820, 3), 245, np.uint8)
    xs, ys = [15, 245, 435, 615, 805], [15, 110, 230, 350, 485]
    for x in xs:
        cv2.line(image, (x, ys[0]), (x, ys[-1]), (25, 25, 25), 2)
    for y in ys:
        cv2.line(image, (xs[0], y), (xs[-1], y), (25, 25, 25), 2)
    cells = [["Control", "Specification", "Batch 01", "Proposed"],
             ["Sieve size", "40", "08", "70"], ["Particle size", "20", "66", "30"],
             ["Pass through", "NMT 60%", "In", "Cl"]]
    for row, values in enumerate(cells):
        for column, word in enumerate(values):
            cv2.putText(image, word, (xs[column] + 12, ys[row] + 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (35, 35, 35), 1, cv2.LINE_AA)
    return image


class ChemicalCandidateTests(unittest.TestCase):
    def test_text_cell_does_not_reach_structure_reader_even_if_it_would_hallucinate_atoms(self):
        for value in ("Control", "70", "In", "Cl", "NH2"):
            image = np.full((100, 260, 3), 255, np.uint8)
            cv2.putText(image, value, (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            labels = [{"text": value, "box": (20, 30, 200, 70), "confidence": 0.95}]
            with self.subTest(value=value), patch.object(pipeline.molscribe_reader, "read") as reader:
                result = pipeline.redraw(image, mode="all", labels=labels)
                reader.assert_not_called()
                self.assertFalse(result["isChemical"])
                self.assertEqual(result["findings"], [])
                self.assertEqual(result["reasons"], [])

    def test_bond_between_labels_is_kept_as_a_chemical_candidate(self):
        image = np.full((120, 300, 3), 255, np.uint8)
        cv2.line(image, (75, 60), (210, 60), (0, 0, 0), 2)
        labels = [{"text": "Cl", "box": (20, 40, 70, 80)}, {"text": "Cl", "box": (215, 40, 270, 80)}]
        self.assertTrue(pipeline.has_structure_strokes(image, labels))

    def test_low_confidence_nonstructure_is_a_quiet_skip_in_both_modes(self):
        image = np.full((120, 300, 3), 255, np.uint8)
        cv2.line(image, (20, 40), (220, 80), (0, 0, 0), 2)
        prediction = {"atoms": [{"confidence": 0.1}, {"confidence": 0.1}], "bonds": [object()], "confidence": 0.1}
        with patch.object(pipeline.molscribe_reader, "read", return_value=prediction):
            for mode in ("proven", "all"):
                result = pipeline.redraw(image, mode=mode, labels=[])
                self.assertFalse(result["isChemical"])
                self.assertEqual(result["findings"], [])


    def test_scattered_atoms_or_words_with_an_unsure_reading_are_not_structures(self):
        """Regression (flowchart box "Manufacturing Process Steps"): the reader made 6 atoms with 2 bonds of it."""
        image = np.full((200, 400, 3), 255, np.uint8)
        cv2.rectangle(image, (10, 10), (390, 190), (0, 0, 0), 2)
        atoms = [{"symbol": "C", "x": 40.0 + 60 * k, "y": 100.0, "confidence": 0.6} for k in range(6)]
        scattered = {"atoms": atoms, "bonds": [{"atoms": (0, 1), "type": "single"}, {"atoms": (2, 3), "type": "single"}],
                     "confidence": 0.61, "smiles": "CC.CC.C.C"}
        words = [{"text": "Manufacturing", "box": (40, 40, 300, 80), "confidence": 0.93},
                 {"text": "Process Steps", "box": (40, 110, 300, 150), "confidence": 0.87}]
        with patch.object(pipeline.molscribe_reader, "read", return_value=scattered):
            for mode in ("proven", "all"):
                result = pipeline.redraw(image, mode=mode, labels=[])
                self.assertFalse(result["isChemical"], result)
                self.assertEqual(result["findings"], [])
        joined = scattered | {"bonds": [{"atoms": (k, k + 1), "type": "single"} for k in range(5)]}
        with patch.object(pipeline.molscribe_reader, "read", return_value=joined),                 patch.object(pipeline, "has_structure_strokes", return_value=True):
            self.assertFalse(pipeline.redraw(image, mode="all", labels=words)["isChemical"])  # words, unsure reading

    def test_bonds_that_are_not_drawn_lines_mean_no_structure(self):
        """Regression (table cell "0.9999" redrawn as a square ring by Redraw all): the bonds read must be drawn."""
        image = np.full((124, 1333, 3), 255, np.uint8)
        cv2.putText(image, "0.9999", (560, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        # What the reader made of the user's cell: two O atoms joined by one bond, drawn nowhere as a line.
        atoms = [{"symbol": "[O]", "x": 600.0, "y": 62.0, "confidence": 0.56}, {"symbol": "O", "x": 648.0, "y": 62.0, "confidence": 0.53}]
        reading = {"atoms": atoms, "bonds": [{"atoms": (0, 1), "type": "double"}], "confidence": 0.56, "smiles": "O=O"}
        with patch.object(pipeline.molscribe_reader, "read", return_value=reading):
            for mode in ("proven", "all"):
                result = pipeline.redraw(image, mode=mode, labels=[])
                self.assertFalse(result["redrawn"])
                self.assertFalse(result["isChemical"], result)
                self.assertEqual(result["findings"], [])

    def test_atoms_read_at_one_spot_do_not_fail(self):
        ink = np.zeros((100, 100), np.uint8)
        ink[50, 10:90] = 1
        atoms = [{"symbol": "H", "x": 5.0, "y": 5.0, "confidence": 0.5} for _ in range(4)]
        points, residual = pipeline.register(ink, atoms, [{"atoms": (0, 1)}, {"atoms": (2, 3)}])
        self.assertEqual(residual, 1.0)


@unittest.skipUnless(AVAILABLE, "redraw and OCR models not installed")
class NonchemicalRedrawTests(unittest.TestCase):
    def test_table_and_each_cell_are_not_chemical_structures(self):
        import panels
        image = ordinary_table()
        boxes = panels.detect(image)
        self.assertGreater(len(boxes), 2)  # reproduces the table-cells-as-panels case in the review screen
        for view in [image] + [image[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]:
            result = pipeline.redraw(view, mode="all")
            self.assertFalse(result["redrawn"], result)
            self.assertFalse(result["isChemical"], result)
            self.assertEqual(result["findings"], [])

    def test_redraw_all_enhances_table_without_chemical_warnings_or_changing_cells_to_molecules(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "table.png", ordinary_table())
            settings = {"redraw": True, "redrawAll": True, "strength": "maximum"}
            result = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "out.png"), "settings": settings}, None)
            # A clean digital table has nothing to clean: kept as it is (or enhanced), never turned into molecules.
            self.assertIn(result["status"], ("completed", "skipped"))
            self.assertNotIn("redraw", [op["op"] for op in enhance.flatten(result.get("operations", []))])
            self.assertNotIn("redrawn_structure", result.get("reviewReasons", []))
            self.assertNotIn("unproven_redraw", result.get("reviewReasons", []))
            self.assertTrue(all(not panel["redraw"].get("findings") for panel in result.get("panels", [])))


@unittest.skipUnless(AVAILABLE, "redraw (MolScribe model, PyTorch, RDKit) not installed")
class RedrawTests(unittest.TestCase):
    D_PENICILLAMINE = "CC(C)(S)[C@@H](N)C(=O)O"
    L_PENICILLAMINE = "CC(C)(S)[C@H](N)C(=O)O"

    def check_redrawn(self, smiles: str):
        img = drawing(smiles)
        result = pipeline.redraw(img)
        self.assertTrue(result["redrawn"], result["reasons"])
        evidence = result["evidence"]
        self.assertEqual(evidence["molecule"]["inchiKey"], inchikey(smiles))  # including stereochemistry
        self.assertEqual(evidence["secondReading"], "identical")
        self.assertEqual(evidence["redrawReadBack"], "identical")
        self.assertEqual(result["image"].shape, img.shape)  # same pixel size
        return result

    def test_both_enantiomers_keep_their_configuration(self):
        d = self.check_redrawn(self.D_PENICILLAMINE)
        l = self.check_redrawn(self.L_PENICILLAMINE)
        self.assertNotEqual(d["evidence"]["molecule"]["inchiKey"], l["evidence"]["molecule"]["inchiKey"])
        self.assertIn("S", d["evidence"]["molecule"]["stereocentres"][0])
        self.assertIn("R", l["evidence"]["molecule"]["stereocentres"][0])

    def test_ring_structures(self):
        self.check_redrawn("CC(=O)Oc1ccccc1C(=O)O")  # aspirin

    def test_pictures_that_are_not_structures_are_not_redrawn(self):
        for img in (samples.photo(), samples.poor_scan(), samples.line_art()):
            self.assertFalse(pipeline.redraw(img)["redrawn"])

    def test_condensed_labels_are_expanded_with_stereo_kept(self):
        # MolScribe reading with a condensed COOH label and a solid wedge to N at the stereocentre (positions as
        # in a drawing): the molecule must come out as D-penicillamine (S).
        reading = {
            "molfile": "\n     RDKit          2D\n\n  7  6  0  0  0  0  0  0  0  0999 V2000\n"
                       "    3.2190   -6.4603    0.0000 C   0  0\n    5.9782   -4.7143    0.0000 C   0  0\n"
                       "    5.9782   -8.2063    0.0000 S   0  0\n    5.9782   -1.5397    0.0000 C   0  0\n"
                       "    8.7374    0.2063    0.0000 R   0  0\n    3.2190    0.2063    0.0000 N   0  0\n"
                       "    8.7374   -6.4603    0.0000 C   0  0\n"
                       "  1  2  1  0\n  2  3  1  0\n  2  4  1  0\n  2  7  1  0\n  4  5  1  0\n  4  6  1  1\n"
                       "A    5\nCOOH\nM  END\n",
            "atoms": [{"x": 179, "y": 267}, {"x": 245, "y": 227}, {"x": 245, "y": 308}, {"x": 245, "y": 154},
                      {"x": 312, "y": 113}, {"x": 179, "y": 113}, {"x": 312, "y": 267}],
        }
        mol, problems = pipeline.build_molecule(reading)
        self.assertEqual(problems, [])
        self.assertEqual(pipeline.identity(mol)["inchiKey"], inchikey(self.D_PENICILLAMINE))

    def test_unknown_labels_are_refused(self):
        reading = {"molfile": "\n     RDKit          2D\n\n  2  1  0  0  0  0  0  0  0  0999 V2000\n"
                              "    0.0000    0.0000    0.0000 C   0  0\n    1.0000    0.0000    0.0000 R   0  0\n"
                              "  1  2  1  0\nA    2\nXyz9\nM  END\n",
                   "atoms": [{"x": 0, "y": 0}, {"x": 50, "y": 0}]}
        mol, problems = pipeline.build_molecule(reading)
        self.assertIsNone(mol)
        self.assertIn("Xyz9", problems[0])

    def test_hashed_wedges_are_counted_in_the_picture(self):
        img = drawing(self.L_PENICILLAMINE)
        marks = [pipeline.stereo_marks(img, length) for length in (60, 80)]
        self.assertTrue(any(m["hashed"] + m["solid"] >= 1 for m in marks))
        self.assertEqual(pipeline.stereo_marks(drawing("CC(C)(S)C(N)C(=O)O"), 70)["hashed"], 0)

    def test_thick_scanned_junctions_are_not_solid_wedges(self):
        # Scanned drawings have thick lines; where four of them meet (a quaternary carbon) the ink is thick too.
        # A scan look: ink bleeds (blur, then dark threshold), which makes thick round blobs at junctions. The old rule
        # counted 2 solid wedges in this wedge-free drawing.
        scanned = lambda img: np.where(cv2.GaussianBlur(img, (0, 0), 2.0) < 150, 0, 255).astype(np.uint8)
        self.assertEqual(pipeline.stereo_marks(scanned(drawing("CC(C)(SSC(C)(C)C(N)C(=O)O)C(N)C(=O)O")), 70)["solid"], 0)
        # A real solid wedge (RDKit draws the L-form with one) is still found with the same scan look.
        self.assertGreaterEqual(pipeline.stereo_marks(scanned(drawing(self.L_PENICILLAMINE)), 70)["solid"], 1)

    def test_a_bond_line_joined_to_a_label_by_ocr_still_reads_as_the_label(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(self.D_PENICILLAMINE)
        result = pipeline.check_labels([{"text": "_COOH", "box": (0, 0, 10, 10)}, {"text": "H2N", "box": (0, 0, 10, 10)}], mol)
        self.assertTrue(result["ok"], result)

    def test_refusals_say_exactly_what_does_not_match_and_where(self):
        img = cv2.copyMakeBorder(drawing("CC(=O)Oc1ccccc1C(=O)O"), 0, 70, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(img, "Figure 3: Aspirin tablets", (60, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        result = pipeline.redraw(img)
        self.assertFalse(result["redrawn"])
        caption = [f for f in result["findings"] if "not part of the structure" in f["message"]]
        self.assertTrue(caption, result["findings"])
        x0, y0, x1, y1 = caption[0]["box"]
        self.assertGreater(y1, 400)  # the box is on the caption line, below the drawing
        # Not a structure: a plain sentence, no place.
        photo = pipeline.redraw(samples.photo())
        self.assertTrue(all("message" in f for f in photo["findings"]))

    def test_enhance_command_redraws_only_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "structure.png", drawing(self.D_PENICILLAMINE))
            plain = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "a.png")}, None)
            self.assertNotIn("redraw", [o["op"] for o in plain.get("operations", [])])
            result = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "b.png"),
                                          "settings": {"redraw": True}}, None)
            self.assertEqual(result["status"], "completed")
            self.assertEqual([o["op"] for o in result["operations"]], ["redraw"])
            self.assertEqual(result["reviewReasons"], ["redrawn_structure"])
            self.assertTrue(result["redraw"]["redrawn"])
            self.assertEqual((result["output"]["width"], result["output"]["height"]), (520, 400))
            # Not a structure: enhanced as usual, and reported as not chemical (or with the reason it was not redrawn).
            scan = samples.write(Path(tmp) / "scan.png", samples.faded_scan(samples.line_art()))
            result = enhance.enhance_cmd({"input": scan, "output": str(Path(tmp) / "c.png"),
                                          "settings": {"redraw": True}}, None)
            self.assertFalse(result["redraw"]["redrawn"])
            self.assertTrue(result["redraw"].get("isChemical") is False or result["redraw"]["reasons"], result["redraw"])
            self.assertNotIn("redraw", [o["op"] for o in result["operations"]])


@unittest.skipUnless(AVAILABLE, "redraw (MolScribe model, PyTorch, RDKit) not installed")
class RedrawAllTests(unittest.TestCase):
    """"Redraw all": uncertain structures are redrawn and marked not proven; nothing else is redrawn."""

    ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

    def test_an_uncertain_structure_is_redrawn_and_marked_not_proven(self):
        img = drawing(self.ASPIRIN)
        strict = pipeline.MIN_ATOM_CONFIDENCE
        pipeline.MIN_ATOM_CONFIDENCE = 0.9999  # every reading now counts as uncertain
        try:
            self.assertFalse(pipeline.redraw(img)["redrawn"])
            result = pipeline.redraw(img, mode="all")
        finally:
            pipeline.MIN_ATOM_CONFIDENCE = strict
        self.assertTrue(result["redrawn"], result["reasons"])
        self.assertFalse(result["proven"])
        self.assertIn("confidence", result["unproven"][0])
        self.assertEqual(result["evidence"]["molecule"]["inchiKey"], inchikey(self.ASPIRIN))
        self.assertEqual(result["image"].shape, img.shape)

    def test_proven_structures_stay_proven(self):
        result = pipeline.redraw(drawing(self.ASPIRIN), mode="all")
        self.assertTrue(result["redrawn"] and result["proven"], result)
        self.assertEqual(result["unproven"], [])

    def test_pictures_that_are_not_structures_are_never_redrawn(self):
        for img in (samples.photo(), samples.poor_scan(), samples.line_art()):
            self.assertFalse(pipeline.redraw(img, mode="all")["redrawn"])

    def test_a_structure_with_a_caption_is_not_redrawn(self):
        img = cv2.copyMakeBorder(drawing(self.ASPIRIN), 0, 70, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(img, "Figure 3: Aspirin tablets", (60, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        result = pipeline.redraw(img, mode="all")
        self.assertFalse(result["redrawn"])

    def test_the_enhance_command_flags_unproven_redraws(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = samples.write(Path(tmp) / "structure.png", drawing(self.ASPIRIN))
            strict = pipeline.MIN_ATOM_CONFIDENCE
            pipeline.MIN_ATOM_CONFIDENCE = 0.9999
            try:
                proven_only = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "a.png"),
                                                   "settings": {"redraw": True}}, None)
                every = enhance.enhance_cmd({"input": src, "output": str(Path(tmp) / "b.png"),
                                             "settings": {"redraw": True, "redrawAll": True}}, None)
            finally:
                pipeline.MIN_ATOM_CONFIDENCE = strict
            self.assertFalse(proven_only["redraw"]["redrawn"])
            self.assertEqual([o["op"] for o in every["operations"]], ["redraw"])
            self.assertEqual(every["reviewReasons"], ["redrawn_structure", "unproven_redraw"])
            self.assertFalse(every["redraw"]["proven"])


if __name__ == "__main__":
    unittest.main()
