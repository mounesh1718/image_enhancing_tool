"""Finds where images are drawn on a PDF page by reading its content stream.

PyMuPDF's get_image_info()/get_image_rects() identify images by decoding and hashing them: slow on large
documents and ambiguous when two different image objects have identical pixels. Instead we tokenize the
content stream, track the transformation matrix through q/Q/cm, and resolve each `Do` to an image object
via the page's resource names (recursing into Form XObjects). No image is decoded.
"""

from __future__ import annotations

import re

import pymupdf

MAX_FORM_DEPTH = 12

_TOKEN = re.compile(
    rb"[\x00\t\n\x0c\r ]+"  # whitespace
    rb"|%[^\r\n]*"  # comment
    rb"|/[^\x00\t\n\x0c\r /<>\[\]()\{\}%]*"  # name
    rb"|<<|>>"
    rb"|<[0-9A-Fa-f\x00\t\n\x0c\r ]*>"  # hex string
    rb"|[\[\]{}]"
    rb"|\("  # literal string start, scanned by hand (nesting/escapes)
    rb"|[^\x00\t\n\x0c\r /<>\[\]()\{\}%]+"  # number, keyword or operator
)
_WHITESPACE = b"\x00\t\n\x0c\r "
_NUMBER = re.compile(rb"[+-]?(\d+\.?\d*|\.\d+)\Z")
_NAME_ESCAPE = re.compile(rb"#([0-9A-Fa-f]{2})")
_WHITESPACE_BYTES = {bytes([c]) for c in _WHITESPACE}

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def multiply(m1, m2):
    """m1 x m2 in PDF's row-vector convention (apply m1, then m2)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def _skip_string(data: bytes, pos: int) -> int:
    """pos is just after '('; returns the index after the matching ')'."""
    depth = 1
    n = len(data)
    while pos < n and depth:
        ch = data[pos]
        if ch == 0x5C:  # backslash escape
            pos += 2
            continue
        if ch == 0x28:
            depth += 1
        elif ch == 0x29:
            depth -= 1
        pos += 1
    return pos


def _skip_inline_image(data: bytes, pos: int) -> int:
    """pos is just after the ID operator; returns the index after EI."""
    pos += 1  # single whitespace after ID
    n = len(data)
    while True:
        i = data.find(b"EI", pos)
        if i < 0:
            return n
        # EI only ends the data when it stands alone; the binary data itself may contain "EI".
        before_ok = i == 0 or data[i - 1] in _WHITESPACE
        after_ok = i + 2 >= n or data[i + 2] in _WHITESPACE
        if before_ok and after_ok:
            return i + 2
        pos = i + 2


def tokens(data: bytes):
    """Yields ('op', bytes) for operators, ('name', str), ('num', float), ('other', None), ('inline', None)."""
    pos = 0
    n = len(data)
    while pos < n:
        m = _TOKEN.match(data, pos)
        if m is None:
            pos += 1
            continue
        tok = m.group()
        pos = m.end()
        first = tok[:1]
        if first in _WHITESPACE_BYTES or first == b"%":
            continue
        if first == b"(":
            pos = _skip_string(data, pos)
            yield "other", None
        elif first == b"/":
            yield "name", _NAME_ESCAPE.sub(lambda e: bytes([int(e.group(1), 16)]), tok[1:]).decode("latin-1")
        elif first in b"<>[]{}":
            yield "other", None
        elif _NUMBER.match(tok):
            yield "num", float(tok)
        elif tok == b"ID":
            pos = _skip_inline_image(data, pos)
            yield "inline", None
        elif tok in (b"true", b"false", b"null"):
            yield "other", None
        else:
            yield "op", tok


def _form_matrix(doc: pymupdf.Document, xref: int):
    kind, value = doc.xref_get_key(xref, "Matrix")
    if kind != "array":
        return IDENTITY
    try:
        values = tuple(float(v) for v in value.strip("[] ").split())
        return values if len(values) == 6 else IDENTITY
    except ValueError:
        return IDENTITY


def find_placements(doc: pymupdf.Document, page: pymupdf.Page) -> list[dict]:
    """Every image drawing operation on the page, in drawing order.

    Each item: {"xref": int (0 for inline images), "matrix": PDF-space CTM, "form": xref of the containing
    Form XObject or 0}. The image occupies the unit square mapped by "matrix".
    """
    images = {(item[9], item[7]): item[0] for item in page.get_images(full=True)}
    forms = {(item[2], item[1]): item[0] for item in page.get_xobjects()}
    placements: list[dict] = []
    contents = page.read_contents()
    if not images and not forms and b"ID" not in contents:
        return placements  # nothing that could draw an image

    def walk(data: bytes, context: int, ctm, active_forms: tuple):
        stack = []
        operands: list = []
        for kind, value in tokens(data):
            if kind != "op" and kind != "inline":
                operands.append((kind, value))
                continue
            if kind == "inline":
                placements.append({"xref": 0, "matrix": ctm, "form": context})
            elif value == b"q":
                stack.append(ctm)
            elif value == b"Q":
                if stack:
                    ctm = stack.pop()
            elif value == b"cm":
                nums = [v for k, v in operands[-6:] if k == "num"]
                if len(nums) == 6:
                    ctm = multiply(tuple(nums), ctm)
            elif value == b"Do" and operands and operands[-1][0] == "name":
                name = operands[-1][1]
                if (context, name) in images:
                    placements.append({"xref": images[(context, name)], "matrix": ctm, "form": context})
                elif (context, name) in forms:
                    form_xref = forms[(context, name)]
                    if form_xref not in active_forms and len(active_forms) < MAX_FORM_DEPTH:
                        walk(doc.xref_stream(form_xref) or b"", form_xref,
                             multiply(_form_matrix(doc, form_xref), ctm), active_forms + (form_xref,))
            operands = []

    walk(contents, 0, IDENTITY, ())
    return placements


def placement_rect(page: pymupdf.Page, matrix) -> pymupdf.Rect:
    """Bounding box of a placement in PyMuPDF page coordinates (top-left origin, page rotation applied)."""
    corners = [pymupdf.Point(x, y) * pymupdf.Matrix(*matrix) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1))]
    rect = pymupdf.Rect(
        min(p.x for p in corners), min(p.y for p in corners), max(p.x for p in corners), max(p.y for p in corners)
    )
    return rect * page.transformation_matrix
