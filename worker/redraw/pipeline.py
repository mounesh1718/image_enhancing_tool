"""Redraws a chemical structure image only when the reading is proven, otherwise leaves it to normal enhancement.

Steps (every one must pass, or the image is not redrawn and the reason is reported):
1. Read      MolScribe reads the drawing: atoms, bonds, wedges and their positions in the image.
2. Build     The molecule is built here with RDKit from that drawing: condensed labels (COOH, NH2, OMe, ...) are
             expanded from a fixed table, and stereochemistry is taken from the wedges and the drawn geometry, the
             standard way a structure drawing is interpreted. (MolScribe's own SMILES can flip a stereocentre when it
             expands a label; it is only used to cross-check the connectivity.)
3. Confirm   - MolScribe's own SMILES has the same connectivity;
             - the picture really shows each bond the reading claims: a line where a single bond is read, two or three
               lines for double/triple bonds, a widening solid wedge or a dashed hashed wedge where stereo is read,
               and nothing drawn that the reading does not explain;
             - the labels OCR reads in the picture (COOH, NH2, SH, OH, ...) exist in the molecule, and the picture
               holds no other text (a caption or logo would be lost by a redraw);
             - a second, differently scaled reading of the picture gives the identical molecule.
4. Redraw    RDKit draws the molecule in the original layout (same atom positions, labels condensed as in the
             original), at the original pixel size, line width and label size.
5. Prove     The redrawn picture is read again and must give the identical molecule (same InChIKey, including
             stereochemistry).

A redrawn image always needs a person's approval (a chemist should compare it with the original).
"""
from __future__ import annotations

import math
import re

import cv2
import numpy as np

from . import molscribe_reader

# Condensed labels the redraw understands: SMILES with the attachment atom first. Anything else is not redrawn.
ABBREVIATIONS = {
    "COOH": "C(=O)O", "HOOC": "C(=O)O", "CO2H": "C(=O)O", "HO2C": "C(=O)O",
    "NH2": "N", "H2N": "N", "NH": "N", "HN": "N",
    "OH": "O", "HO": "O", "SH": "S", "HS": "S",
    "CH3": "C", "H3C": "C", "Me": "C",
    "OMe": "OC", "MeO": "OC", "OCH3": "OC", "H3CO": "OC",
    "Et": "CC", "C2H5": "CC", "OEt": "OCC", "EtO": "OCC",
    "CN": "C#N", "NC": "C#N", "NO2": "[N+](=O)[O-]", "O2N": "[N+](=O)[O-]",
    "CF3": "C(F)(F)F", "F3C": "C(F)(F)F", "CHO": "C=O", "OHC": "C=O",
    "COOMe": "C(=O)OC", "CO2Me": "C(=O)OC", "MeO2C": "C(=O)OC", "MeOOC": "C(=O)OC",
    "COOEt": "C(=O)OCC", "CO2Et": "C(=O)OCC", "EtO2C": "C(=O)OCC", "EtOOC": "C(=O)OCC",
    "CONH2": "C(N)=O", "H2NOC": "C(N)=O", "Ac": "C(C)=O", "OAc": "OC(C)=O", "AcO": "OC(C)=O",
    "NHAc": "NC(C)=O", "AcHN": "NC(C)=O", "NHMe": "NC", "MeHN": "NC", "NMe2": "N(C)C", "Me2N": "N(C)C",
    "SMe": "SC", "MeS": "SC", "SO3H": "S(=O)(=O)O", "HO3S": "S(=O)(=O)O",
    "Ph": "c1ccccc1", "tBu": "C(C)(C)C", "t-Bu": "C(C)(C)C", "iPr": "C(C)C", "i-Pr": "C(C)C",
}
MIN_CONFIDENCE = 0.70       # MolScribe's overall score; real structures read 0.73-0.90, a logo 0.15
MIN_ATOM_CONFIDENCE = 0.70
MIN_COVERAGE = 0.93         # share of the drawing's ink explained by the read bonds and labels
MAX_LAYOUT_RESIDUAL = 0.12  # after alignment, read bonds must lie within this many bond lengths of the drawn lines
MIN_BONDS_VERIFIED = 0.80   # share of bonds long enough to check in the picture
SECOND_VIEW_SCALE = 0.8
# "Redraw all" mode: below these a picture is not treated as a chemical structure at all (a logo read 0.15, pages of
# text and charts about 0.0), or the reading explains too little of it to stand in for it.
STRUCTURE_FLOOR = 0.30
CLEAN_STYLE = 0.6  # lines and labels this much of the measured size, when a redraw from a blurred original is unclear
MIN_COVERAGE_ALL = 0.60

_LABEL_GROUPS = {  # OCR label -> SMARTS of the group that must exist in the molecule
    "COOH": "C(=O)[OX2H1,OX1-]", "HOOC": "C(=O)[OX2H1,OX1-]", "CO2H": "C(=O)[OX2H1,OX1-]",
    "NH2": "[NX3;H2]", "H2N": "[NX3;H2]", "SH": "[SX2;H1]", "HS": "[SX2;H1]",
    "OH": "[OX2;H1]", "HO": "[OX2;H1]",
}
# Short tokens OCR produces from parts of a drawing (ring letters, hash marks); they are not "other text".
_ATOM_TOKEN = re.compile(r"^(?:[A-Z][a-z]?\d*){1,4}$")


def available() -> bool:
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return False
    return molscribe_reader.available()


def unavailable_reason() -> str | None:
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return "RDKit is not installed"
    return molscribe_reader.unavailable_reason()


# --------------------------------------------------------------------------- build the molecule

def build_molecule(reading: dict):
    """The molecule from MolScribe's drawing (molfile with image positions and wedges), labels expanded here.
    Returns (mol, problems)."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Geometry import Point3D

    problems: list[str] = []
    raw = Chem.MolFromMolBlock(reading["molfile"], sanitize=False, removeHs=False) if reading.get("molfile") else None
    if raw is None or raw.GetNumAtoms() == 0:
        return None, ["the drawing could not be turned into a molecule"]
    # Positions as drawn in the image (y down), so wedges are interpreted on the real layout.
    conf = raw.GetConformer()
    for k, atom in enumerate(reading["atoms"][:raw.GetNumAtoms()]):
        conf.SetAtomPosition(k, Point3D(atom["x"], -atom["y"], 0.0))
    # Wedges from the molfile (1 = solid wedge, 6 = hashed wedge, starting at the bond's first atom).
    for bond in raw.GetBonds():
        stereo = bond.GetIntProp("_MolFileBondStereo") if bond.HasProp("_MolFileBondStereo") else 0
        if stereo in (1, 6):
            bond.SetBondDir(Chem.BondDir.BEGINWEDGE if stereo == 1 else Chem.BondDir.BEGINDASH)
    rw = Chem.RWMol(raw)
    expansions = []
    for atom in raw.GetAtoms():
        if atom.GetAtomicNum() == 0 or atom.HasProp("molFileAlias"):
            label = atom.GetProp("molFileAlias") if atom.HasProp("molFileAlias") else atom.GetSymbol()
            smiles = ABBREVIATIONS.get(label.strip())
            if smiles is None:
                problems.append(f"label '{label}' is not a known group")
                continue
            expansions.append((atom.GetIdx(), smiles, label.strip()))
    if problems:
        return None, problems
    drawn = rw.GetNumAtoms()
    for idx, smiles, label in expansions:
        # The label atom becomes the group's first atom in place: same index, position, bonds and wedges.
        fragment = Chem.MolFromSmiles(smiles)
        label_atom = rw.GetAtomWithIdx(idx)
        first = fragment.GetAtomWithIdx(0)
        label_atom.SetAtomicNum(first.GetAtomicNum())
        label_atom.SetFormalCharge(first.GetFormalCharge())
        label_atom.SetNoImplicit(False)
        label_atom.SetNumExplicitHs(0)
        if label_atom.HasProp("molFileAlias"):
            label_atom.ClearProp("molFileAlias")
        # Remembered so the redraw shows the group condensed exactly as the original did.
        label_atom.SetProp("condensedLabel", label)
        mapping = {0: idx}
        for atom in list(fragment.GetAtoms())[1:]:
            new = Chem.Atom(atom.GetAtomicNum())
            new.SetBoolProp("condensedPart", True)
            new.SetFormalCharge(atom.GetFormalCharge())
            new.SetIsAromatic(atom.GetIsAromatic())
            mapping[atom.GetIdx()] = rw.AddAtom(new)
        for bond in fragment.GetBonds():
            rw.AddBond(mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()], bond.GetBondType())
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return None, [f"the read structure is not chemically valid ({exc.__class__.__name__})"]
    if mol.GetNumAtoms() > drawn:
        # The added group atoms get proper 2D positions; every drawn atom keeps its position from the picture.
        from rdkit.Geometry import Point2D
        conf = mol.GetConformer()
        keep = {k: conf.GetAtomPosition(k) for k in range(drawn)}
        bond_length = _bond_length_from(mol, keep) or 50.0
        scale = 1.5 / bond_length
        coord_map = {k: Point2D(p.x * scale, p.y * scale) for k, p in keep.items()}
        directions = {b.GetIdx(): b.GetBondDir() for b in mol.GetBonds()}
        rdDepictor.Compute2DCoords(mol, coordMap=coord_map, clearConfs=True)
        conf = mol.GetConformer()
        for k in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(k)
            conf.SetAtomPosition(k, Point3D(p.x / scale, p.y / scale, 0.0))
        for b in mol.GetBonds():
            b.SetBondDir(directions.get(b.GetIdx(), b.GetBondDir()))
    # Remember which bonds the drawing wedged (stereo perception below may clear the markers), so the redraw
    # puts the wedges where the original had them.
    for b in mol.GetBonds():
        if b.GetBondDir() in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
            b.SetIntProp("drawnWedge", int(b.GetBondDir()))
            b.SetIntProp("drawnWedgeFrom", b.GetBeginAtomIdx())
    Chem.AssignChiralTypesFromBondDirs(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol, []


def _bond_length_from(mol, positions) -> float | None:
    lengths = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in positions and b in positions:
            pa, pb = positions[a], positions[b]
            d = math.hypot(pa.x - pb.x, pa.y - pb.y)
            if d > 1:
                lengths.append(d)
    return float(np.median(lengths)) if lengths else None


def identity(mol) -> dict:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    return {
        "smiles": Chem.MolToSmiles(mol),
        "inchiKey": Chem.MolToInchiKey(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecularWeight": round(Descriptors.MolWt(mol), 2),
        "stereocentres": [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i + 1} {c}"
                          for i, c in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)],
    }


# --------------------------------------------------------------------------- confirm against the picture

def _ink(img: np.ndarray) -> np.ndarray:
    gray = img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (gray <= min(threshold, 200)).astype(np.uint8)


def _stroke_width(ink: np.ndarray) -> float:
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    ridge = (dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    values = dist[ridge]
    return float(max(1.0, 2.0 * np.median(values))) if values.size else 2.0


def _labelled(atoms) -> list[bool]:
    return [re.sub(r"[\[\]@]", "", a["symbol"]) not in ("C", "CH", "CH2", "CH3") for a in atoms]


def register(ink: np.ndarray, atoms: list[dict], bonds: list[dict], iterations: int = 40) -> tuple[list[np.ndarray], float]:
    """Aligns the read skeleton with the drawn lines (trimmed iterative closest point, affine).

    MolScribe's atom positions are approximate (a coarse grid, and sometimes a stretched layout when it expands a
    label), so the bond checks would look beside the lines. Points along the read bonds are matched to the nearest
    ink and an affine correction is fitted. Returns the corrected positions and the remaining mean distance in
    bond lengths (large = the reading's layout does not match the drawing)."""
    points = np.array([[a["x"], a["y"]] for a in atoms], float)
    labelled = _labelled(atoms)
    lengths = [np.linalg.norm(points[i] - points[j]) for i, j in (b["atoms"] for b in bonds)]
    bond_length = float(np.median(lengths)) if lengths else 1.0
    if bond_length < 1e-6:
        return [p for p in points], 1.0  # every atom read at one spot: nothing to align (and no bond to check)
    # Distance to the nearest ink pixel, and which pixel that is.
    dist, labels = cv2.distanceTransformWithLabels((1 - ink).astype(np.uint8), cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(ink)
    lookup = np.zeros((labels.max() + 1, 2), float)
    lookup[labels[ys, xs]] = np.stack([xs, ys], axis=1)
    samples, owners = [], []
    for b in bonds:
        i, j = b["atoms"]
        for t in np.linspace(0.3 if labelled[i] else 0.1, 0.7 if labelled[j] else 0.9, 7):
            samples.append(points[i] + (points[j] - points[i]) * t)
            owners.append((i, j, t))
    if len(samples) < 6:
        return [p for p in points], 1.0
    source = np.array(samples)
    h, w = ink.shape
    matrix = np.eye(3)
    residual = 1.0
    for _ in range(iterations):
        moved = source @ matrix[:2, :2].T + matrix[:2, 2]
        xi = np.clip(moved[:, 0].round().astype(int), 0, w - 1)
        yi = np.clip(moved[:, 1].round().astype(int), 0, h - 1)
        target = lookup[labels[yi, xi]]
        error = np.linalg.norm(target - moved, axis=1)
        keep = error <= np.percentile(error, 80)
        a = np.hstack([source[keep], np.ones((keep.sum(), 1))])
        solution, *_ = np.linalg.lstsq(a, target[keep], rcond=None)
        new = np.eye(3)
        new[:2, :2], new[:2, 2] = solution[:2].T, solution[2]
        if np.abs(new - matrix).max() < 1e-4:
            matrix = new
            break
        matrix = new
        residual = float(error[keep].mean()) / bond_length
    corrected = points @ matrix[:2, :2].T + matrix[:2, 2]
    return [p for p in corrected], residual


def check_drawing(img: np.ndarray, reading: dict, label_boxes: list[tuple] | None = None) -> dict:
    """Does the picture show what was read? Per-bond line checks and how much of the drawing is explained."""
    ink = _ink(img)
    stroke = _stroke_width(ink)
    atoms, bonds = reading["atoms"], reading["bonds"]
    points, residual = register(ink, atoms, bonds)
    lengths = [float(np.linalg.norm(points[i] - points[j])) for i, j in (b["atoms"] for b in bonds)]
    bond_length = float(np.median(lengths)) if lengths else 0.0
    if bond_length < 4 * stroke:
        return {"ok": False, "problems": ["the drawing is too small to check"],
                "findings": [finding("The drawing is too small to check its bonds.")], "stroke": stroke,
                "bondLength": bond_length}
    labelled = _labelled(atoms)
    # A drawing that shows hydrogens on carbons writes its carbons out as letters too ("H-C" style).
    symbols = [re.sub(r"[\[\]@]", "", a["symbol"]) for a in atoms]
    if any(symbols[i] == "H" and symbols[j].startswith("C") or symbols[j] == "H" and symbols[i].startswith("C")
           for i, j in (b["atoms"] for b in bonds)):
        labelled = [True] * len(atoms)
    explicit_carbons = all(labelled)
    problems: list[str] = []
    findings: list[dict] = []
    if residual > MAX_LAYOUT_RESIDUAL:
        problems.append(f"the read layout does not match the drawing (off by {residual:.2f} bond lengths)")
        findings.append(finding(f"The atom positions the AI read do not match the drawing (off by {residual:.2f} "
                                f"bond lengths on average)."))
    # Text labels (H2N, COOH, ...): atoms inside a label, and the extra atoms MolScribe adds when it expands one
    # (the O atoms of COOH, placed beside the text), belong to the label; bonds inside a label are not lines.
    pad = 0.12 * bond_length
    boxes = [(x0 - pad, y0 - pad, x1 + pad, y1 + pad) for x0, y0, x1, y1 in label_boxes or []]

    def in_label(point) -> bool:
        return any(x0 <= point[0] <= x1 and y0 <= point[1] <= y1 for x0, y0, x1, y1 in boxes)

    neighbours = {k: [] for k in range(len(atoms))}
    for b in bonds:
        neighbours[b["atoms"][0]].append(b["atoms"][1])
        neighbours[b["atoms"][1]].append(b["atoms"][0])
    plain = [re.sub(r"[\[\]@H\d]", "", a["symbol"]) for a in atoms]
    # Which text label each atom belongs to. The reader places the atoms of a written group (HOOC, NH2, OH) inside
    # or just beside its text: the carbon of an expanded COOH often lands just outside the box.
    raw_boxes = [tuple(map(float, box)) for box in label_boxes or []]
    near = LABEL_REACH * bond_length
    box_of: list[int | None] = []
    for point in points:
        distances = [_box_distance(point, box) for box in raw_boxes]
        best = int(np.argmin(distances)) if distances else None
        box_of.append(best if best is not None and distances[best] <= near else None)
    label_atom = [in_label(point) for point in points]
    for k in range(len(atoms)):
        if label_atom[k] or box_of[k] is None:
            continue
        if plain[k] not in ("C", ""):
            label_atom[k] = True  # a heteroatom written next to its text (the O of "HO", the N of "NH2")
        else:
            # The carbon of a written group (COOH): apart from where it is attached, its neighbours are the
            # group's own end atoms (O, N), which the reader may place anywhere around the text.
            group = [n for n in neighbours[k] if plain[n] not in ("C", "") and len(neighbours[n]) == 1]
            if group and len(neighbours[k]) - len(group) <= 1:
                label_atom[k] = True
    for k in range(len(atoms)):
        if not label_atom[k] and plain[k] not in ("C", "") and len(neighbours[k]) == 1 and label_atom[neighbours[k][0]]:
            label_atom[k] = True  # e.g. an O of an expanded COOH placed away from the text
            box_of[k] = box_of[k] if box_of[k] is not None else box_of[neighbours[k][0]]
    checked = checkable = 0
    h, w = ink.shape
    for b, length in zip(bonds, lengths):
        i, j = b["atoms"]
        p, q = points[i], points[j]
        if length < 1 or (label_atom[i] and label_atom[j]):
            continue  # inside a text label
        # A bond into a text label is drawn to the letter it attaches to, which need not be where the reader put
        # the atom: the drawn line is looked for between the atom and the label's edge, close to the read direction.
        for skeleton, labelled_end in ((i, j), (j, i)):
            if label_atom[labelled_end] and not label_atom[skeleton] and box_of[labelled_end] is not None:
                start_point = points[skeleton]
                end_point = _line_to_label(ink, start_point, points[labelled_end], raw_boxes[box_of[labelled_end]],
                                           boxes, bond_length, stroke)
                p, q = (start_point, end_point) if skeleton == i else (end_point, start_point)
                length = float(np.linalg.norm(q - p))
                break
        if length < 1:
            continue
        checkable += 1
        direction = (q - p) / length
        normal = np.array([-direction[1], direction[0]])
        start = (0.38 if labelled[i] else 0.12) * bond_length
        end = length - (0.38 if labelled[j] else 0.12) * bond_length
        # Only the part of the bond outside any text label is a drawn line.
        ts = [t for t in np.linspace(start, end, 21) if not in_label(p + direction * t)]
        if len(ts) < 8 or ts[-1] - ts[0] < 0.25 * bond_length:
            if "H" in (symbols[i], symbols[j]) or label_atom[i] or label_atom[j]:
                checkable -= 1  # a short stub into a text label: the label itself is what shows the bond
            continue  # too short between labels to judge
        checked += 1
        # The reading can sit a little beside the drawn line (a slightly shifted or turned part of the drawing):
        # the straight line that fits the ink best within a narrow band around the read bond is checked.
        side = max(int(stroke) + 1, int(0.08 * bond_length))  # sideways tolerance of the continuity test below
        p, q = _snap_to_line(ink, p, q, ts[0], ts[-1], LINE_SNAP * bond_length, side)
        direction = (q - p) / max(float(np.linalg.norm(q - p)), 1e-6)
        normal = np.array([-direction[1], direction[0]])
        ts = np.linspace(ts[0], ts[-1], 15)
        half = 0.22 * bond_length
        offsets = np.arange(-half, half + 1)
        runs, widths, centre_hits = [], [], []
        for t in ts:
            centre = p + direction * t
            xs = np.clip((centre[0] + offsets * normal[0]).round().astype(int), 0, w - 1)
            ys = np.clip((centre[1] + offsets * normal[1]).round().astype(int), 0, h - 1)
            profile = ink[ys, xs]
            edges = np.diff(np.concatenate([[0], profile, [0]]))
            starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
            runs.append(len(starts))
            # The width of this line only (the stroke nearest the centre), not of a neighbouring bond in the window.
            if len(starts):
                centre_index = len(profile) / 2
                nearest = int(np.argmin([0 if a <= centre_index < b else min(abs(a - centre_index), abs(b - centre_index))
                                         for a, b in zip(starts, ends)]))
                widths.append(int(ends[nearest] - starts[nearest]))
            else:
                widths.append(0)
            # Thick scanned lines sit a little off the read centre line: a small sideways tolerance (the gaps of a
            # hashed wedge are along the line, so they are still found).
            middle = profile[len(profile) // 2 - side: len(profile) // 2 + side + 1]
            centre_hits.append(bool(middle.any()))
        kind = b["type"]
        central = runs[4:11]
        continuity = float(np.mean(centre_hits))
        # Width near a quarter and three quarters of the bond: a solid wedge widens all along, while at the very ends
        # a plain line meets other bonds (a crowded junction looked like a wide end).
        first, last = np.median(widths[3:6]), np.median(widths[9:12])
        widening = max(first, last) / max(1.0, min(first, last))
        # (Continuity mainly guards against a hashed wedge read as a single bond; hashed wedges are also counted
        # in the picture itself, see stereo_marks.)
        expected = None
        if kind == "single" and (continuity < 0.6 or np.median(central) != 1 or widening > 2.2):
            expected = "single bond"
        elif kind == "double" and np.median(central) < 2:
            expected = "double bond"
        elif kind == "triple" and np.median(central) < 3:
            expected = "triple bond"
        elif kind == "solid wedge" and (widening < 1.5 or continuity < 0.8):
            expected = "solid wedge"
        elif kind == "dashed wedge" and continuity > 0.85:
            expected = "hashed wedge"
        if expected:
            problems.append(_bond_problem(atoms, i, j, expected, continuity, central, widening))
            findings.append(finding(_bond_message(atoms, i, j, expected, continuity, central, widening),
                                    _box_around([p, q], 0.18 * bond_length, w, h)))
    verified = checked / checkable if checkable else 0.0
    if verified < MIN_BONDS_VERIFIED:
        problems.append(f"only {checked} of {checkable} drawn bonds could be checked in the picture")
        findings.append(finding(f"Only {checked} of the {checkable} bonds the AI read could be found in the drawing "
                                f"(too short or hidden by labels), too few to confirm the reading."))
    # Everything drawn should be explained by a bond or a label.
    explained = np.zeros_like(ink)
    corridor = max(2, int(round(0.2 * bond_length)))
    for i, j in (b["atoms"] for b in bonds):
        cv2.line(explained, tuple(points[i].round().astype(int)), tuple(points[j].round().astype(int)), 1, 2 * corridor)
    for point, has_label in zip(points, labelled):
        radius = int(round((0.4 if has_label else 0.2) * bond_length))
        cv2.circle(explained, (int(point[0]), int(point[1])), radius, 1, -1)
    # Labels OCR found (H2N, COOH, ...) are explained by their own boxes, when they sit at a labelled atom.
    for x0, y0, x1, y1 in label_boxes or []:
        centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        if any(has_label and np.linalg.norm(point - centre) < 0.6 * bond_length + (x1 - x0) / 2
               for point, has_label in zip(points, labelled)):
            pad = int(round(0.1 * bond_length))
            cv2.rectangle(explained, (int(x0) - pad, int(y0) - pad), (int(x1) + pad, int(y1) + pad), 1, -1)
    total = int(ink.sum())
    coverage = float((ink & explained).sum()) / total if total else 0.0
    if coverage < MIN_COVERAGE:
        problems.append(f"{(1 - coverage) * 100:.0f} % of the drawing is not explained by the reading")
        # Where: the biggest pieces of the drawing that nothing read accounts for.
        missing = (ink & (1 - explained)).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(missing, connectivity=8)
        pieces = sorted(range(1, count), key=lambda k: -stats[k, cv2.CC_STAT_AREA])[:3]
        boxes = [[float(stats[k, 0]), float(stats[k, 1]), float(stats[k, 0] + stats[k, 2]), float(stats[k, 1] + stats[k, 3])]
                 for k in pieces if stats[k, cv2.CC_STAT_AREA] >= 0.02 * total]
        message = (f"{(1 - coverage) * 100:.0f} % of the drawing is not explained by what the AI read: part of the "
                   f"structure (or other marks) was not read.")
        findings.extend(finding(message, box) for box in boxes) if boxes else findings.append(finding(message))
    return {"ok": not problems, "problems": problems, "findings": findings, "coverage": round(coverage, 3), "bondsChecked": checked,
            "bonds": checkable, "stroke": round(stroke, 2), "bondLength": round(bond_length, 1),
            "layoutResidual": round(residual, 3), "explicitCarbons": explicit_carbons}


# Atoms this close to a text label (in bond lengths) can belong to it; directions searched for a bond into a label.
LABEL_REACH = 0.5
LABEL_SEARCH_DEGREES = 35


def _box_distance(point, box) -> float:
    x0, y0, x1, y1 = box
    dx = max(x0 - point[0], 0.0, point[0] - x1)
    dy = max(y0 - point[1], 0.0, point[1] - y1)
    return float(np.hypot(dx, dy))


def _line_to_label(ink: np.ndarray, start, read_end, box, padded_boxes, bond_length: float, stroke: float) -> np.ndarray:
    """Where the drawn line from `start` into a text label ends: of the points on the label's edge within
    LABEL_SEARCH_DEGREES of the read direction, the one whose straight path from `start` lies most on ink (only
    the part outside every label counts). The read position when nothing is closer to a drawn line."""
    h, w = ink.shape
    x0, y0, x1, y1 = box
    edge = [np.array([x, y]) for x in np.linspace(x0, x1, 13) for y in (y0, y1)] + \
           [np.array([x, y]) for y in np.linspace(y0, y1, 9) for x in (x0, x1)]
    read_vector = np.asarray(read_end, float) - np.asarray(start, float)
    read_angle = np.arctan2(read_vector[1], read_vector[0])
    reach = max(2, int(stroke) + 1)

    def on_ink(end) -> float:
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 1:
            return 0.0
        direction = vector / length
        hits = total = 0
        for t in np.linspace(0.12 * bond_length, length, 25):
            point = start + direction * t
            if any(bx0 <= point[0] <= bx1 and by0 <= point[1] <= by1 for bx0, by0, bx1, by1 in padded_boxes):
                continue
            px, py = int(round(point[0])), int(round(point[1]))
            total += 1
            hits += bool(ink[max(0, py - reach):min(h, py + reach + 1), max(0, px - reach):min(w, px + reach + 1)].any())
        return hits / total if total >= 6 else 0.0

    best, best_score = np.asarray(read_end, float), on_ink(np.asarray(read_end, float))
    for point in edge:
        vector = point - start
        angle = np.arctan2(vector[1], vector[0])
        turn = abs((angle - read_angle + np.pi) % (2 * np.pi) - np.pi)
        if np.degrees(turn) > LABEL_SEARCH_DEGREES:
            continue
        score = on_ink(point)
        if score > best_score + 0.05:
            best, best_score = point, score
    return best


SOLID_WEDGE_LENGTH = 0.2  # bond lengths: a thick shape at least this long is a solid wedge, not a junction
LINE_SNAP = 0.15  # in bond lengths: how far beside the read bond a drawn line is looked for
SNAP_ONLY_BELOW = 0.6  # only bonds whose read line lies less on ink than this are fitted to the drawing


def _snap_to_line(ink: np.ndarray, p, q, start: float, end: float, max_shift: float, side: int):
    """The bond's two ends moved sideways (each by up to `max_shift`) so the straight line between them lies most
    on ink (within `side` pixels, as the continuity test counts it), sampled between `start` and `end` along the read
    bond. Unchanged when the read line already passes, or when nothing is clearly better."""
    h, w = ink.shape
    length = float(np.linalg.norm(q - p))
    if length < 1 or max_shift < 1:
        return p, q
    direction = (q - p) / length
    normal = np.array([-direction[1], direction[0]])
    fractions = np.linspace(start, end, 15) / length
    reach = side
    shifts = np.arange(-int(max_shift), int(max_shift) + 1, 2)

    def score(a, b) -> float:
        hits = 0
        for f in fractions:
            x, y = (1 - f) * (p + normal * a) + f * (q + normal * b)
            xi, yi = int(round(x)), int(round(y))
            hits += bool(ink[max(0, yi - reach):min(h, yi + reach + 1), max(0, xi - reach):min(w, xi + reach + 1)].any())
        return hits / len(fractions)

    best, best_score = (0, 0), score(0, 0)
    if best_score >= SNAP_ONLY_BELOW:
        return p, q  # the read line already lies on ink: moving it could only drift towards a neighbouring line
    for a in shifts:
        for b in shifts:
            value = score(a, b)
            if value > best_score + 0.1:
                best, best_score = (a, b), value
    return p + normal * best[0], q + normal * best[1]


def finding(message: str, box=None) -> dict:
    """A check result for the app: a plain sentence and, when known, where in the picture (x0, y0, x1, y1)."""
    return {"message": message, "box": [round(float(v), 1) for v in box] if box is not None else None}


def _box_around(points, pad: float, w: int, h: int) -> list[float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [max(0.0, min(xs) - pad), max(0.0, min(ys) - pad), min(float(w), max(xs) + pad), min(float(h), max(ys) + pad)]


def _bond_message(atoms, i, j, expected, continuity, central, widening) -> str:
    """What does not match at one bond, in plain words."""
    a, b = atoms[i]["symbol"].strip("[]"), atoms[j]["symbol"].strip("[]")
    lines = int(np.median(central))
    if expected == "single bond":
        if lines == 0 or continuity < 0.15:
            return f"Bond {a}–{b}: the AI read a bond here, but no line is drawn at this place."
        if widening > 2.2:
            return f"Bond {a}–{b}: the line gets wider like a bold wedge (3D bond), but the AI read a plain bond."
        if lines >= 2:
            return f"Bond {a}–{b}: {lines} lines are drawn here (a double bond?), but the AI read a single bond."
        return (f"Bond {a}–{b}: the AI read a plain line, but the drawn line is broken or dashed here "
                f"(only {continuity * 100:.0f} % of it is drawn), as in a dashed wedge.")
    if expected == "double bond":
        return f"Bond {a}={b}: the AI read a double bond, but only one line is drawn."
    if expected == "triple bond":
        return f"Bond {a}≡{b}: the AI read a triple bond, but fewer lines are drawn."
    if expected == "solid wedge":
        return f"Bond {a}–{b}: the AI read a bold wedge (3D bond), but the drawn line does not widen."
    return f"Bond {a}–{b}: the AI read a dashed wedge (3D bond), but a solid line is drawn."


def _bond_problem(atoms, i, j, expected, continuity, central, widening) -> str:
    return (f"bond {atoms[i]['symbol'].strip('[]')}–{atoms[j]['symbol'].strip('[]')} read as {expected} does not "
            f"look like one (line continuity {continuity:.2f}, lines {int(np.median(central))}, widening {widening:.1f})")


def read_labels(img: np.ndarray) -> list[dict]:
    import textcheck
    return textcheck.read(img) if textcheck.available() else []


def has_structure_strokes(img: np.ndarray, lines: list[dict]) -> bool:
    """Exclude text-only images/cells before treating OCR letters and numbers as chemical atoms.

    A chemical drawing needs strokes outside its text labels. Remove confidently read text and table/frame
    rules, then look for a remaining bond-sized line. This is only a candidate check: passing it does not prove
    the image is chemical. Without OCR evidence, leave the decision to the structure reader.
    """
    ink = _ink(img)
    if not ink.any():
        return False
    text = [line for line in lines if line.get("confidence", 1.0) >= 0.65]
    if not text:
        return True
    h, w = ink.shape
    heights = [max(1, line["box"][3] - line["box"][1]) for line in text]
    for line, height in zip(text, heights):
        x0, y0, x1, y1 = line["box"]
        pad = max(2, round(height * 0.12))
        cv2.rectangle(ink, (int(x0) - pad, int(y0) - pad), (int(x1) + pad, int(y1) + pad), 0, -1)
    # Residual frames at the edge of a cropped table cell are not chemical bonds.
    border = max(2, round(min(h, w) * 0.02))
    ink[:border] = ink[-border:] = 0
    ink[:, :border] = ink[:, -border:] = 0
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(12, w // 3)), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(12, h // 3), 1), np.uint8))
    # Only remove long rules when they form a grid/frame, so a lone horizontal molecular bond is kept.
    if cv2.connectedComponents(horizontal)[0] >= 3 and cv2.connectedComponents(vertical)[0] >= 3:
        rules = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8))
        ink[rules > 0] = 0
    length = max(10, round(float(np.median(heights)) * 1.2))
    segments = cv2.HoughLinesP(ink, 1, np.pi / 180, threshold=max(8, length // 2),
                               minLineLength=length, maxLineGap=2)
    return segments is not None


# A structure is one molecule, or a salt of two: more separate pieces means the reader is guessing at text.
MAX_PIECES = 2
# A structure shows at least this many bonds as drawn lines, and at least half of the bonds read.
MIN_DRAWN_BONDS = 2
_WORD = re.compile(r"[A-Za-z]{5,}")


def _pieces(reading: dict) -> int:
    """How many separate pieces (molecules) the reading has."""
    parent = list(range(len(reading["atoms"])))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for b in reading["bonds"]:
        i, j = b["atoms"]
        parent[root(i)] = root(j)
    return len({root(k) for k in range(len(parent))})


def _prose(lines: list[dict]) -> list[str]:
    """Ordinary words the OCR read confidently ("Manufacturing", "Process"): not chemical labels such as COOH."""
    words = [w for line in lines if line.get("confidence", 1.0) >= 0.8 for w in _WORD.findall(line.get("text", ""))]
    return [w for w in words if sum(c.islower() for c in w) >= 3]


def nonchemical() -> dict:
    """A normal image is not a failed chemical redraw and has no chemical warnings to show."""
    return {"redrawn": False, "isChemical": False, "reasons": [], "findings": [], "evidence": {}}


def check_labels(lines: list[dict], mol) -> dict:
    """OCR (a second, independent reader) reads the labels: each must exist in the molecule, and no other text."""
    import textcheck
    from rdkit import Chem
    if not textcheck.available():
        return {"ok": True, "available": False, "labels": []}
    labels, other = [], []
    label_boxes: dict[str, list] = {}
    other_boxes: list = []
    for line in lines:
        for token in re.split(r"[\s,;:()]+", line["text"]):
            token = token.strip(".'\"`_-~|")  # OCR joins the end of a bond line to a label ("_COOH")
            if not token:
                continue
            if token in _LABEL_GROUPS or _ATOM_TOKEN.match(token):
                labels.append(token)
                label_boxes.setdefault(_LABEL_GROUPS.get(token, token), []).append(line["box"])
            elif len(token) <= 2 and token.islower():
                continue  # fragments of hash marks or wedges
            else:
                other.append(token)
                if other_boxes and other_boxes[-1][1] is line["box"]:
                    other_boxes[-1] = (other_boxes[-1][0] + " " + token, line["box"])  # one finding per line of text
                else:
                    other_boxes.append((token, line["box"]))
    problems = []
    findings = []
    if other:
        problems.append("the picture holds text other than atom labels: " + ", ".join(other[:4]))
        findings.extend(finding(f"Text “{token}” in the picture is not part of the structure (a caption, or a "
                                f"label the AI did not read).", box) for token, box in other_boxes[:6])
    counts: dict[str, int] = {}
    for token in labels:
        if token in _LABEL_GROUPS:
            counts[_LABEL_GROUPS[token]] = counts.get(_LABEL_GROUPS[token], 0) + 1
    for smarts, seen in counts.items():
        present = len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
        if present < seen:
            problems.append(f"OCR reads {seen} label(s) for {smarts} but the molecule has {present}")
            name = next((t for t, g in _LABEL_GROUPS.items() if g == smarts), smarts)
            findings.extend(finding(f"The text reader sees {seen} “{name}” group(s), but the molecule the AI read "
                                    f"has only {present}.", box) for box in label_boxes.get(smarts, [])[:4]
                            ) if label_boxes.get(smarts) else findings.append(finding(
                f"The text reader sees {seen} “{name}” group(s), but the molecule the AI read has only {present}."))
    return {"ok": not problems, "available": True, "labels": labels, "problems": problems, "findings": findings,
            "other": other}


# --------------------------------------------------------------------------- redraw

def render(mol, size: tuple[int, int], box: tuple[int, int, int, int], stroke: float, label_height: float,
           monochrome: bool, explicit_carbons: bool = False) -> np.ndarray:
    """Draws the molecule in its own layout into the drawing's box of an image of the original size."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    width, height = size
    x0, y0, x1, y1 = box
    bw, bh = int(max(16, x1 - x0)), int(max(16, y1 - y0))
    # Groups the original showed condensed (COOH, H2N, ...) are drawn condensed with the original's own text;
    # everything the original drew as atoms and bonds is drawn as atoms and bonds.
    drawn = Chem.RWMol(mol)
    if explicit_carbons:  # the original writes its carbons out as "C"
        for atom in drawn.GetAtoms():
            if atom.GetAtomicNum() == 6 and not atom.HasProp("condensedLabel"):
                atom.SetProp("_displayLabel", "C")
    for atom in drawn.GetAtoms():
        if atom.HasProp("condensedLabel"):
            text = re.sub(r"(\d+)", r"<sub></sub>", atom.GetProp("condensedLabel"))
            atom.SetProp("_displayLabel", text)
            atom.SetProp("_displayLabelW", text)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
    for idx in sorted((a.GetIdx() for a in drawn.GetAtoms() if a.HasProp("condensedPart")), reverse=True):
        drawn.RemoveAtom(idx)
    drawn = drawn.GetMol()
    canvas = rdMolDraw2D.MolDraw2DCairo(bw, bh)
    options = canvas.drawOptions()
    options.padding = 0.10  # room for labels at the edges (the fixed label size is not part of RDKit's fit)
    options.bondLineWidth = max(1.0, round(stroke, 1))
    options.scaleBondWidth = False
    options.fixedFontSize = max(10, int(round(label_height)))
    options.additionalAtomLabelPadding = 0.05
    if monochrome:
        options.useBWAtomPalette()
    drawn.UpdatePropertyCache(strict=False)
    # The wedges stay on the bonds the original drew them on (RDKit would otherwise choose its own bond).
    has_wedges = False
    for b in drawn.GetBonds():
        b.SetBondDir(Chem.BondDir.NONE)
    for b in drawn.GetBonds():
        if b.HasProp("drawnWedge"):
            if b.GetBeginAtomIdx() != b.GetIntProp("drawnWedgeFrom"):
                continue  # direction lost when atoms were removed; let RDKit place this wedge
            b.SetBondDir(Chem.BondDir.values[b.GetIntProp("drawnWedge")])
            has_wedges = True
    prepared = rdMolDraw2D.PrepareMolForDrawing(drawn, kekulize=False, addChiralHs=False, wedgeBonds=not has_wedges)
    canvas.DrawMolecule(prepared)
    canvas.FinishDrawing()
    picture = cv2.imdecode(np.frombuffer(canvas.GetDrawingText(), np.uint8), cv2.IMREAD_COLOR)
    out = np.full((height, width, 3), 255, np.uint8)
    out[y0:y0 + picture.shape[0], x0:x0 + picture.shape[1]] = picture[:height - y0, :width - x0]
    return out


def fresh_layout(mol):
    """The molecule in RDKit's standard 2D layout, turned and scaled onto the read positions (rotation only, never
    mirrored). The original's wedges are dropped: their meaning depends on the layout, so RDKit places wedges again
    from the molecule's stereochemistry. None when it cannot be done."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Geometry import Point3D
    try:
        old = mol.GetConformer()
        before = np.array([[old.GetAtomPosition(i).x, old.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
        fresh = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(fresh, clearConfs=True)
        conf = fresh.GetConformer()
        after = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(fresh.GetNumAtoms())])
        a, b = after - after.mean(axis=0), before - before.mean(axis=0)
        u, sv, vt = np.linalg.svd(a.T @ b)
        if np.linalg.det(u @ vt) < 0:  # a mirror image would change what the wedges mean
            u[:, -1] *= -1
            sv[-1] *= -1
        rotation = u @ vt
        scale = float(sv.sum() / max((a ** 2).sum(), 1e-9))
        placed = a @ rotation * scale + before.mean(axis=0)
        for i, (x, y) in enumerate(placed):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        for bond in fresh.GetBonds():
            if bond.HasProp("drawnWedge"):
                bond.ClearProp("drawnWedge")
        return fresh
    except Exception:
        return None


def condense_like_picture(mol, lines: list[dict]) -> None:
    """Groups the original shows as text (OCR read "COOH" there) but the reader expanded into atoms are marked to
    be drawn condensed again, so the redraw looks like the original."""
    from rdkit import Chem
    conf = mol.GetConformer()
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    for line in lines:
        text = line["text"].strip()
        if text not in ("COOH", "HOOC", "CO2H", "HO2C"):
            continue
        x0, y0, x1, y1 = line["box"]
        # The reader's position for an expanded group is approximate: within about one bond length of the text.
        pad = _bond_length_from(mol, {k: conf.GetAtomPosition(k) for k in range(mol.GetNumAtoms())}) or (y1 - y0)
        for carbon, oxo, hydroxy in mol.GetSubstructMatches(pattern):
            atom = mol.GetAtomWithIdx(carbon)
            if atom.HasProp("condensedLabel"):
                continue
            point = conf.GetAtomPosition(carbon)
            if x0 - pad <= point.x <= x1 + pad and y0 - pad <= -point.y <= y1 + pad:
                atom.SetProp("condensedLabel", text)
                mol.GetAtomWithIdx(oxo).SetBoolProp("condensedPart", True)
                mol.GetAtomWithIdx(hydroxy).SetBoolProp("condensedPart", True)
                break


def _drawing_box(img: np.ndarray, margin: float = 0.02) -> tuple[int, int, int, int]:
    ink = _ink(img)
    ys, xs = np.nonzero(ink)
    h, w = ink.shape
    if not len(xs):
        return 0, 0, w, h
    pad_x, pad_y = int(w * margin), int(h * margin)
    return (int(max(0, xs.min() - pad_x)), int(max(0, ys.min() - pad_y)),
            int(min(w, xs.max() + 1 + pad_x)), int(min(h, ys.max() + 1 + pad_y)))


def _label_height(lines: list[dict], bond_length: float) -> float:
    heights = [l["box"][3] - l["box"][1] for l in lines]
    return float(np.median(heights)) * 0.85 if heights else 0.4 * bond_length


def _second_view(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(32, int(w * SECOND_VIEW_SCALE)), max(32, int(h * SECOND_VIEW_SCALE))),
                       interpolation=cv2.INTER_AREA)
    pad = max(8, int(0.08 * max(small.shape[:2])))
    return cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def redraw(img: np.ndarray, original: np.ndarray | None = None, mode: str = "proven",
           labels: list[dict] | None = None) -> dict:
    """{"redrawn": bool, "proven": bool, "image": array when redrawn, "reasons": [...] when not, "evidence": {...}}.

    `original`: when `img` is a cleaned copy of a noisy picture (the reader reads clean drawings far better), the
    original read directly must agree on the molecule's connectivity, so nothing lost in cleaning goes unnoticed.

    mode "proven": redrawn only when every check passes. mode "all": a check that only makes the reading less certain
    (confidence, stereo marks, labels, a second reading, the read-back) no longer refuses: the structure is redrawn
    from the reading and marked not proven, with the checks it failed in "unproven". Still refused: pictures that are
    not chemical structures, readings that give no molecule or explain too little of the drawing, and pictures with
    other text (it would be lost)."""
    from rdkit import Chem
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = np.ascontiguousarray(img[:, :, :3])
    lines = read_labels(img) if labels is None else labels
    if not has_structure_strokes(img, lines):
        return nonchemical()
    evidence: dict = {}
    unproven: list[str] = []
    found: list[dict] = []  # what does not match, in plain words and where (for the app)
    lenient = mode == "all"
    height_px, width_px = img.shape[:2]

    def refuse(*reasons, findings=None):
        reasons = [r for r in reasons if r]
        items = findings if findings else [finding(r[:1].upper() + r[1:] + ".") for r in reasons]
        return {"redrawn": False, "isChemical": True, "reasons": reasons, "findings": found + items, "evidence": evidence}

    def doubt(*reasons, findings=None):
        """A failed check: refuses in "proven" mode, is recorded in "all" mode (returns None then)."""
        reasons = [r for r in reasons if r]
        items = findings if findings else [finding(r[:1].upper() + r[1:] + ".") for r in reasons]
        if not lenient:
            return refuse(*reasons, findings=items)
        unproven.extend(reasons)
        found.extend(items)
        return None

    reading = molscribe_reader.read(img)
    low_atom = min((a["confidence"] for a in reading["atoms"]), default=0.0)
    evidence["confidence"] = round(reading["confidence"], 3)
    if len(reading["atoms"]) < 2 or not reading["bonds"] or reading["confidence"] < STRUCTURE_FLOOR:
        return nonchemical()
    if _pieces(reading) > MAX_PIECES or (reading["confidence"] < MIN_CONFIDENCE and len(_prose(lines)) >= 2):
        # Scattered "atoms" (a reader guessing at a text box), or ordinary words with an unsure reading: a table
        # cell, a flowchart box or a label, never a structure to redraw or to report atoms for.
        return nonchemical()
    drawing = check_drawing(img, reading, [l["box"] for l in lines])
    if "bonds" in drawing and (drawing["bonds"] < MIN_DRAWN_BONDS or drawing["bondsChecked"] < 0.5 * drawing["bonds"]):
        # The bonds the reader saw are not drawn lines in the picture (a number, a word or a box read as atoms).
        return nonchemical()
    if reading["confidence"] < MIN_CONFIDENCE or low_atom < MIN_ATOM_CONFIDENCE:
        weak = [a for a in reading["atoms"] if a["confidence"] < MIN_ATOM_CONFIDENCE]
        size = 0.05 * max(width_px, height_px)
        items = [finding(f"The AI is not sure about this atom ({a['symbol'].strip('[]')}, {a['confidence'] * 100:.0f} % "
                         f"sure).", _box_around([(a["x"], a["y"])], size, width_px, height_px)) for a in weak[:5]]
        if reading["confidence"] < MIN_CONFIDENCE:
            items.insert(0, finding(f"The AI is only {reading['confidence'] * 100:.0f} % sure of its reading of the "
                                    f"whole structure (at least {MIN_CONFIDENCE * 100:.0f} % is needed)."))
        if (r := doubt(f"not recognised as a chemical structure with enough confidence (overall "
                       f"{reading['confidence']:.2f}, weakest atom {low_atom:.2f})", findings=items)):
            return r
    mol, problems = build_molecule(reading)
    if mol is None:
        return refuse(*problems)
    built = identity(mol)
    if original is not None:
        direct, _ = build_molecule(molscribe_reader.read(original))
        if direct is None or Chem.MolToInchiKey(direct)[:14] != built["inchiKey"][:14]:
            if (r := doubt("read from the cleaned picture, but the original read directly gives a different molecule",
                           findings=[finding("Reading the cleaned picture and reading the original give different "
                                             "molecules, so cleaning may have changed something.")])):
                return r
        else:
            evidence["readFrom"] = "cleaned picture (the original, read directly, agrees)"
    evidence["molecule"] = built
    model = Chem.MolFromSmiles(reading["smiles"]) if reading.get("smiles") else None
    if model is None or Chem.MolToInchiKey(model)[:14] != built["inchiKey"][:14]:
        if (r := doubt("the two ways of building the molecule from the reading disagree on its connectivity",
                       findings=[finding("The AI's two descriptions of the molecule (its atom list and its formula "
                                         "string) disagree on which atoms are connected.")])):
            return r
    if model is not None and Chem.MolToInchiKey(model) != built["inchiKey"]:
        evidence["note"] = "stereochemistry taken from the drawn wedges (the model's own SMILES differed)"
    read_hashed = sum(b["type"] == "dashed wedge" for b in reading["bonds"])
    read_solid = sum(b["type"] == "solid wedge" for b in reading["bonds"])
    marks = {"hashed": read_hashed, "solid": read_solid}
    if drawing.get("bondLength"):
        # Stereo marks counted in the picture itself: a hashed wedge the reader missed (or invented) would turn the
        # molecule into a different stereoisomer. Solid wedges are verified bond by bond (they must widen); the
        # count here can only add a refusal.
        marks = stereo_marks(img, drawing["bondLength"], [l["box"] for l in lines])
        evidence["stereoMarks"] = {"pictureHashed": marks["hashed"], "readHashed": read_hashed,
                                   "pictureSolid": marks["solid"], "readSolid": read_solid}
    quick_reasons: list[str] = []
    quick_findings: list[dict] = []
    if drawing.get("bondLength") and (marks["hashed"] != read_hashed or marks["solid"] > read_solid):
        quick_reasons.append(f"stereo marks in the picture ({marks['hashed']} hashed, {marks['solid']} solid wedge) "
                             f"do not match the reading ({read_hashed} hashed, {read_solid} solid)")
        if marks["hashed"] != read_hashed:
            message = (f"The drawing has {marks['hashed']} dashed wedge(s) (3D bonds), but the AI read {read_hashed}. "
                       f"A redraw would change the 3D form (for example D/L).")
            boxes = marks.get("hashedBoxes") or [None]
            quick_findings.extend(finding(message, box) for box in boxes)
        if marks["solid"] > read_solid:
            message = (f"The drawing has {marks['solid']} bold wedge(s) (3D bonds), but the AI read {read_solid}. "
                       f"A redraw would change the 3D form (for example D/L).")
            boxes = marks.get("solidBoxes") or [None]
            quick_findings.extend(finding(message, box) for box in boxes)
    evidence["drawing"] = {k: v for k, v in drawing.items() if k not in ("problems", "findings")}
    if not drawing["ok"]:
        if lenient and (drawing.get("coverage") or 0.0) < MIN_COVERAGE_ALL:
            return refuse(f"the reading explains only {(drawing.get('coverage') or 0.0) * 100:.0f} % of the drawing",
                          *drawing["problems"][:1], findings=[finding(
                              f"What the AI read explains only {(drawing.get('coverage') or 0.0) * 100:.0f} % of the "
                              f"drawing, too little to stand in for it.")] + drawing.get("findings", [])[:5])
        quick_reasons.extend(drawing["problems"])
        quick_findings.extend(drawing.get("findings", []))
    labels = check_labels(lines, mol)
    evidence["labels"] = labels.get("labels", [])
    words = [t for t in labels.get("other", []) if len(t) >= 4 and t.isalpha()]
    if lenient and words:
        return refuse("the picture holds text besides the structure (" + ", ".join(words[:4])
                      + "); redrawing would remove it", findings=[f for f in labels.get("findings", [])
                                                                  if "not part of the structure" in f["message"]])
    if not labels["ok"]:
        quick_reasons.extend(labels["problems"])
        quick_findings.extend(labels.get("findings", []))
    if quick_reasons and (r := doubt(*quick_reasons, findings=quick_findings)):
        return r
    second, _ = build_molecule(molscribe_reader.read(_second_view(img)))
    if second is None or Chem.MolToInchiKey(second) != built["inchiKey"]:
        if (r := doubt("a second reading of the picture at another scale gives a different molecule",
                       findings=[finding("Reading the drawing a second time, at another size, gives a different "
                                         "molecule: the reading is not stable.")])):
            return r
    else:
        evidence["secondReading"] = "identical"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    chroma = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(np.int16) - 128
    monochrome = float(np.percentile(np.abs(chroma).max(axis=2)[gray < 200], 95)) < 20 if (gray < 200).any() else True
    condense_like_picture(mol, lines)
    height, width = img.shape[:2]
    box = _drawing_box(img, margin=0.05)
    tight = _drawing_box(img, margin=0.0)
    near = 0.02
    if tight[0] <= width * near or tight[1] <= height * near or tight[2] >= width * (1 - near) \
            or tight[3] >= height * (1 - near):
        box = (0, 0, width, height)  # a drawing that reaches the edges gets the whole picture, so labels fit
    if not drawing.get("stroke") or not drawing.get("bondLength"):
        return refuse("the drawing is too small to redraw",
                      findings=[finding("The drawing is too small to redraw clearly.")])
    label_height = _label_height(lines, drawing["bondLength"])

    def draw(molecule, style: float = 1.0):
        for shrink in (1.0, 0.85, 0.72):  # smaller labels until nothing is cut at the edges
            picture = render(molecule, (width, height), box, drawing["stroke"] * style, label_height * shrink * style,
                             monochrome, drawing.get("explicitCarbons", False))
            edge = _ink(picture)
            if not (edge[0].any() or edge[-1].any() or edge[:, 0].any() or edge[:, -1].any()):
                return picture
        return None

    def reads_back(picture):
        proof, proof_problems = build_molecule(molscribe_reader.read(picture))
        if proof is not None and Chem.MolToInchiKey(proof) == built["inchiKey"]:
            return True, ""
        return False, Chem.MolToSmiles(proof) if proof is not None else "; ".join(proof_problems)

    picture = draw(mol)
    if picture is None:
        # Labels measured on a blurred or scanned original look bigger than they were: smaller labels and thinner
        # lines, then a fresh layout, before giving up.
        fresh = fresh_layout(mol)
        for molecule, style, note in ((mol, CLEAN_STYLE, "drawn with thinner lines and smaller labels so it fits"),
                                      (fresh, 1.0, "laid out afresh so it fits"),
                                      (fresh, CLEAN_STYLE, "laid out afresh with thinner lines and smaller labels so it fits")):
            if molecule is not None and (picture := draw(molecule, style)) is not None:
                evidence["layout"] = note
                break
    if picture is None:
        return refuse("the redrawn structure does not fit in the picture without cutting a label",
                      findings=[finding("The clean redraw does not fit in the picture without cutting off a label.")])
    same, got = reads_back(picture)
    if not same:
        # A blurred or noisy original gives poor atom positions (a bond folded back onto its neighbour) and looks
        # thicker than it was (blur widens lines and letters). Drawn again in a cleaner style (thinner lines, smaller
        # labels), then also laid out afresh and turned to the original's orientation. Kept only if it then reads
        # back as the identical molecule, stereochemistry included.
        fresh = fresh_layout(mol)
        attempts = [(mol, CLEAN_STYLE, "drawn with thinner lines and smaller labels than the blurred original")]
        if fresh is not None:
            attempts += [(fresh, 1.0, "laid out afresh: the original's atom positions drew the molecule unclearly"),
                         (fresh, CLEAN_STYLE, "laid out afresh with thinner lines and smaller labels")]
        for molecule, style, note in attempts:
            candidate = draw(molecule, style)
            if candidate is not None and reads_back(candidate)[0]:
                picture, same = candidate, True
                evidence["layout"] = note
                break
    if not same:
        if (r := doubt(f"the redrawn picture does not read back as the same molecule (read back: {got})",
                       findings=[finding(f"The clean redraw does not read back as the same molecule (it reads as {got}), "
                                         f"so it cannot be proven to show the right structure.")])):
            return r
    else:
        evidence["redrawReadBack"] = "identical"
    return {"redrawn": True, "isChemical": True, "proven": not unproven, "unproven": unproven, "image": picture, "reasons": [],
            "findings": found, "evidence": evidence}


def stereo_marks(img: np.ndarray, bond_length: float, label_boxes: list | None = None) -> dict:
    """Counts the stereo marks drawn in the picture, independently of any reader.

    Hashed wedge: three or more short, parallel dashes in a row. Solid wedge: a filled shape several line widths
    thick (bold label letters stay below that)."""
    ink = _ink(img)
    stroke = _stroke_width(ink)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    dashes = []
    for k in range(1, count):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 3 or area > 0.05 * bond_length ** 2:
            continue
        ys, xs = np.nonzero(labels[stats[k, cv2.CC_STAT_TOP]:stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT],
                                  stats[k, cv2.CC_STAT_LEFT]:stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH]] == k)
        if len(xs) < 3:
            continue
        (_, _), (a, b), angle = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
        long_side, short_side = max(a, b), max(1.0, min(a, b))
        if 0.06 * bond_length <= long_side <= 0.5 * bond_length and short_side <= 2.5 * stroke and long_side / short_side >= 2.0:
            orientation = angle if a >= b else angle + 90.0
            dashes.append((centroids[k], orientation % 180.0, long_side))
    # Group dashes that are close, parallel and in a row.
    groups, seen = [], set()
    for start in range(len(dashes)):
        if start in seen:
            continue
        group, queue = [], [start]
        seen.add(start)
        while queue:
            k = queue.pop()
            group.append(k)
            for m in range(len(dashes)):
                if m in seen:
                    continue
                gap = np.linalg.norm(dashes[k][0] - dashes[m][0])
                turn = abs(dashes[k][1] - dashes[m][1])
                turn = min(turn, 180 - turn)
                if gap <= 0.22 * bond_length and turn <= 25:
                    seen.add(m)
                    queue.append(m)
        if len(group) >= 3:
            groups.append(group)
    hashed_boxes = []
    for group in groups:
        centres = np.array([dashes[k][0] for k in group])
        reach = max(dashes[k][2] for k in group) / 2 + stroke
        hashed_boxes.append([float(centres[:, 0].min() - reach), float(centres[:, 1].min() - reach),
                             float(centres[:, 0].max() + reach), float(centres[:, 1].max() + reach)])
    # Solid wedges: long shapes where the drawing is much thicker than its lines. A point where several thick
    # scanned lines meet is thick too, but small and round, so a thick shape must run along a bond for a while.
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    thick = (dist >= max(1.5, 1.4 * stroke)).astype(np.uint8)
    n, thick_labels, thick_stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    solid = 0
    solid_boxes = []
    pad = 0.05 * bond_length
    for k in range(1, n):
        if thick_stats[k, cv2.CC_STAT_AREA] < max(4, 0.6 * stroke ** 2):
            continue
        ys, xs = np.nonzero(thick_labels == k)
        cx, cy = float(xs.mean()), float(ys.mean())
        if any(x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad for x0, y0, x1, y1 in label_boxes or []):
            continue  # bold-looking letters of a label: a wedge is a bond, never part of the text
        (_, _), (a, b), _ = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
        if max(a, b) + 2 * 1.4 * stroke >= SOLID_WEDGE_LENGTH * bond_length:
            solid += 1
            grow = 2 * stroke
            solid_boxes.append([float(xs.min() - grow), float(ys.min() - grow), float(xs.max() + grow), float(ys.max() + grow)])
    return {"hashed": len(groups), "solid": solid, "stroke": round(stroke, 2), "hashedBoxes": hashed_boxes,
            "solidBoxes": solid_boxes}
