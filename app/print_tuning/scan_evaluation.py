"""Read a scanned trial sheet, line it up with its manifest and measure how every tile printed.

trial_sheet.py writes a trial PDF, a manifest (every region in mm from the paper's top-left
corner) and a cells .npz (the intended gray of every data cell). After the sheet is printed and
scanned, evaluate_scan():
  1. reads the scan (PDF, PNG or JPG; one file with all pages or one image per page) in colour,
  2. finds the 4 corner markers on each page (in gray) and maps the page onto the manifest's mm grid,
  3. identifies each page by its page code,
  4. measures the paper and black patch (gray and RGB) and the gray patches: how this printer
     really prints gray on this paper; no brightness is stretched, so tinted paper shows as it is,
  5. for every candidate (one experiment printed with one mapping, candidate_id = variation_id)
     cuts out its tile and panels and compares the printed cell grays with the intended ones,
  6. saves the result (contract version 1) as JSON plus PNG crops of every tile and panel.
It only measures; choosing settings is left to the agents. Grays run from 0 (black) to 1 (white).
A null measurement always comes with a reason in the record's `notes`, and `measurement_docs`
gives the units and meaning of every number. A ScanError carries a `code`; `needs_rescan` tells
whether a new scan would help.

Command line:
    python -m app.print_tuning.scan_evaluation SCAN_FILE [MORE_PAGES ...] MANIFEST [--out DIR]

Offline diagnostic overlays (no session writes):
    diagnose_scan_alignment(scan_paths, manifest_path, diagnostic_directory)

Detection tolerates halftone/bilevel markers using local contrast and square
geometry. Bilevel source pages are explicitly flagged in scan-quality issues:
successful alignment cannot recover tone information discarded by a scanner.
"""
import argparse
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont, ImageOps

SCHEMA_VERSION = "scan_eval/2"
CONTRACT_VERSION = 1
PAGE_W, PAGE_H = 210, 297
MM_PER_INCH = 25.4
PDF_DPI = 300                      # PDF scans are rendered at this resolution
PX_PER_MM = 10                     # every page is resampled onto this grid
MAX_FIT_ERROR_MM = 1.0
DARK = 0.5                         # a pixel below this gray counts as ink
CORNER_SEARCH_MM = 45              # markers are searched this far from each image corner
COARSE_PX_PER_MM = 2               # resolution of the first, rough marker search
MIN_MARKER_CONTRAST = 0.15         # marker versus its local paper moat, not absolute black
MARKER_SCALE_RANGE = (0.65, 1.35)  # apparent marker side / printed side, allowing skew and scaling
NEAR_PAPER, NEAR_BLACK = 0.04, 0.04  # a cell this close to the measured paper / black counts as such
SAME_GRAY = 0.02                   # printed grays closer than this count as one level
LEVEL_BINS = 16                    # intended grays are grouped into this many levels
TILE_INSET_MM = 0.3                # skip the plot's border when cutting it out
MARK_LINE_MM = 0.35                # half width of the box mark's double outline, plus a little
LOW_CONTRAST = 0.5                 # paper gray minus black gray below this is a page issue
MANY_FULL_BLACK = 0.3              # candidate warnings: share of cells printed as full black,
MANY_PAPER_WHITE = 0.3             # share of cells printed as paper white,
MIN_CORRELATION = 0.9              # and the lowest acceptable intended/printed correlation

IMAGE_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
UPSIDE_DOWN = {"top_left": "bottom_right", "top_right": "bottom_left",
               "bottom_left": "top_right", "bottom_right": "top_left"}

Box = list[float]


SCAN_PROBLEMS = ("scan_unreadable", "low_dpi", "markers_not_found", "page_mismatch", "page_missing")
OPERATIONAL_PROBLEMS = ("manifest_missing", "cells_missing")

MEASUREMENT_DOCS = {
    # pages
    "page_number": {"units": "page number, from 1", "meaning": "page of the trial sheet (as in the manifest)"},
    "scan_index": {"units": "index, from 0", "meaning": "position of the page in the given scan"},
    "page_code_read": {"units": "6 bits (0 or 1)", "meaning": "page code squares read from the scan (1 = filled)"},
    "page_code_expected": {"units": "6 bits (0 or 1)", "meaning": "page code the manifest gives this page"},
    "dpi": {"units": "dots per inch", "meaning": "resolution of the scanned page"},
    "orientation": {"units": "degrees", "meaning": "clockwise quarter-turn of the page in the scan (0, 90, 180, 270)"},
    "fit_error_mm": {"units": "mm", "meaning": "largest miss between a found corner marker and where the "
                                               "other three markers put it"},
    "paper.gray": {"units": "gray (0 black to 1 white)", "meaning": "mean gray of the blank paper areas"},
    "paper.rgb": {"units": "[R, G, B], each 0 to 1", "meaning": "mean colour of the blank paper areas, unstretched"},
    "black.gray": {"units": "gray (0 black to 1 white)", "meaning": "mean gray of the solid black patch"},
    "black.rgb": {"units": "[R, G, B], each 0 to 1", "meaning": "mean colour of the solid black patch, unstretched"},
    "paper_to_black_contrast": {"units": "gray (0 black to 1 white)", "meaning": "paper gray minus black gray: the "
                                                                                  "full printable range"},
    "gray_patches.level": {"units": "gray (0 black to 1 white)", "meaning": "gray the patch was meant to print"},
    "gray_patches.measured": {"units": "gray (0 black to 1 white)", "meaning": "mean printed gray of the patch"},
    "gray_patches.std": {"units": "gray (0 black to 1 white)", "meaning": "standard deviation of the printed "
                                                                         "gray within the patch (noise, banding)"},
    # panels
    "cells_used": {"units": "cells", "meaning": "data cells measured (cells under a box mark's outline are left out)"},
    "paper_white_share": {"units": "fraction 0 to 1", "meaning": "share of cells printed within "
                                                                f"{NEAR_PAPER} of the paper gray"},
    "full_black_share": {"units": "fraction 0 to 1", "meaning": "share of cells printed within "
                                                               f"{NEAR_BLACK} of the black patch gray"},
    "intended_levels": {"units": "levels", "meaning": f"intended grays grouped into {LEVEL_BINS} levels: "
                                                      "how many of those levels the panel uses"},
    "printed_levels": {"units": "levels", "meaning": "how many intended levels stay apart on paper (neighbours "
                                                     f"closer than {SAME_GRAY} gray count as one)"},
    "collapsed_levels": {"units": "levels", "meaning": "intended levels minus printed levels: gray levels that "
                                                       "merge on paper (summed over panels for a candidate)"},
    "correlation": {"units": "Pearson r, -1 to 1", "meaning": "correlation of printed with intended cell grays "
                                                             "(mean over panels for a candidate)"},
    "mean_abs_difference": {"units": "gray (0 black to 1 white)", "meaning": "mean absolute difference between "
                                                                             "printed and intended cell grays"},
    "contrast": {"units": "gray (0 black to 1 white)", "meaning": "95th minus 5th percentile of the printed cell "
                                                                  "grays (mean over panels for a candidate)"},
    "intended_contrast": {"units": "gray (0 black to 1 white)", "meaning": "95th minus 5th percentile of the "
                                                                           "intended cell grays"},
    "focus.inside_mean": {"units": "gray (0 black to 1 white)", "meaning": "mean printed gray of the cells inside "
                                                                           "the focus box"},
    "focus.rest_mean": {"units": "gray (0 black to 1 white)", "meaning": "mean printed gray of the other cells"},
    "focus.difference": {"units": "gray (0 black to 1 white)", "meaning": "focus inside_mean minus rest_mean on paper"},
    "focus.intended_difference": {"units": "gray (0 black to 1 white)", "meaning": "the same difference in the "
                                                                                   "intended grays"},
    # candidates
    "mapping.gamma": {"units": "exponent", "meaning": "gamma of the data-to-gray mapping (1 = linear)"},
    "mapping.dark_level": {"units": "gray (0 black to 1 white)", "meaning": "gray the mapping gives the lowest data"},
    "mapping.light_level": {"units": "gray (0 black to 1 white)", "meaning": "gray the mapping gives the highest data"},
    "mapping.steps": {"units": "levels", "meaning": "number of gray steps (below 2 = smooth)"},
    "focus_difference": {"units": "gray (0 black to 1 white)", "meaning": "panel name -> focus.difference of "
                                                                          "that panel"},
}


class ScanError(ValueError):
    """The scan cannot be read, lined up or matched to this trial sheet.

    code is one of SCAN_PROBLEMS (a new scan may fix it) or OPERATIONAL_PROBLEMS (the trial's files).
    """

    def __init__(self, code: str, message: str):
        if code not in SCAN_PROBLEMS + OPERATIONAL_PROBLEMS:
            raise ValueError(f"unknown ScanError code {code!r}")
        super().__init__(message)
        self.code = code

    @property
    def needs_rescan(self) -> bool:
        return self.code in SCAN_PROBLEMS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def _local(saved: str, folder: Path) -> Path:
    """The saved path if it exists, else the same file name in folder (runs copied between machines)."""
    path = Path(saved)
    return path if path.exists() else folder / PureWindowsPath(saved).name


# ---------------------------------------------------------------- 1. read the scan

def _pdf_page_dpi(page) -> float:
    """The real resolution of a scanned PDF page: that of an image covering most of it, else PDF_DPI."""
    page_area = page.rect.width * page.rect.height
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        if (x1 - x0) * (y1 - y0) > 0.5 * page_area and info.get("width"):
            return info["width"] / ((x1 - x0) / 72)
    return PDF_DPI


def _estimate_dpi(gray: np.ndarray) -> float:
    """dpi from the short/long image sides; quarter-turns are resolved by page code."""
    height, width = gray.shape
    short, long = sorted((height, width))
    return min(short / (PAGE_W / MM_PER_INCH), long / (PAGE_H / MM_PER_INCH))


def _gray_and_rgb(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """(grays 0..1 as float32, RGB as uint8 height x width x 3) of one page image, unstretched."""
    rgb = image.convert("RGB")
    return np.asarray(rgb.convert("L"), dtype=np.float32) / 255, np.asarray(rgb, dtype=np.uint8)


def _read_scan(scan_path, accepted: list[str]) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], list[dict]]:
    """Every scanned page as (grays 0..1, RGB uint8, dpi), and the scan files with their sha256."""
    paths = [Path(p) for p in scan_path] if isinstance(scan_path, (list, tuple)) else [Path(scan_path)]
    if not paths:
        raise ScanError("scan_unreadable", "no scan files given")
    pages, files = [], []
    for path in paths:
        suffix = path.suffix.lower().lstrip(".")
        if not path.is_file():
            raise ScanError("scan_unreadable", f"scan file not found: {path}")
        if suffix not in accepted:
            raise ScanError("scan_unreadable", f"{path.name}: scans must be one of {accepted}")
        files.append({"path": str(path.resolve()), "sha256": _sha256(path)})
        try:
            if suffix == "pdf":
                with pymupdf.open(path) as document:
                    for page in document:
                        if any(info.get("bpc") == 1 and pymupdf.Rect(info["bbox"]).get_area() > .5 * page.rect.get_area()
                               for info in page.get_image_info()):
                            files[-1].setdefault("bilevel_scan_indices", []).append(len(pages))
                        pixmap = page.get_pixmap(dpi=PDF_DPI, colorspace=pymupdf.csRGB, alpha=False)
                        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples,
                                                "raw", "RGB", pixmap.stride)
                        gray, rgb = _gray_and_rgb(image)
                        pages.append((gray, rgb, min(_pdf_page_dpi(page), _estimate_dpi(gray))))
            else:
                with Image.open(path) as image:
                    if image.info.get("scan_provenance"):
                        files[-1]["scan_provenance"] = image.info["scan_provenance"]
                    if image.mode == "1":
                        files[-1].setdefault("bilevel_scan_indices", []).append(len(pages))
                    gray, rgb = _gray_and_rgb(ImageOps.exif_transpose(image))
                pages.append((gray, rgb, _estimate_dpi(gray)))
        except ScanError:
            raise
        except Exception as error:
            raise ScanError("scan_unreadable", f"cannot read {path.name}: {type(error).__name__}: {error}") from error
    return pages, files


# ---------------------------------------------------------------- 2. line up each page

def _box_sum(values: np.ndarray, half: int) -> np.ndarray:
    """Sum over the (2 half + 1)^2 square around every pixel; outside counts as 0."""
    padded = np.pad(values, half + 1)
    total = padded.cumsum(0).cumsum(1)
    size = 2 * half + 1
    return (total[size:, size:] - total[:-size, size:] - total[size:, :-size] + total[:-size, :-size])[
        :values.shape[0], :values.shape[1]]


def _marker_candidates(gray: np.ndarray, corner: str, px_per_mm: float, side_mm: float = 8.0) -> list[dict]:
    """Find isolated filled squares from image evidence in a corner search area.

    Area averaging is for geometry detection only: it joins scanner halftone dots
    without changing measured pixels. Connected components must have the expected
    size, square aspect, filled interior, unclipped boundary and a light paper moat.
    The final centre is the ink-weighted centroid of the original pixels. Absolute
    marker gray is deliberately not required to be near black: real bilevel scans
    can render a printed black marker as a 50% dotted square.
    """
    height, width = gray.shape
    size = int(CORNER_SEARCH_MM * px_per_mm)
    y0 = 0 if corner.startswith("top") else max(0, height - size)
    x0 = 0 if corner.endswith("left") else max(0, width - size)
    region = gray[y0:y0 + size, x0:x0 + size]

    step = max(1, int(round(px_per_mm / COARSE_PX_PER_MM)))
    rows, cols = region.shape[0] // step, region.shape[1] // step
    if not rows or not cols:
        return []
    coarse = region[:rows * step, :cols * step].reshape(rows, step, cols, step).mean(axis=(1, 3))
    paper = float(np.quantile(coarse, .85))
    contrast = paper - float(np.quantile(coarse, .02))
    mask = paper - coarse > max(.08, .35 * contrast)
    seen = np.zeros_like(mask)
    candidates = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        yy, xx = [], []
        while stack:
            y, x = stack.pop()
            yy.append(y)
            xx.append(x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(xx) < 4:  # isolated detection-grid specks cannot be an 8 mm marker
            continue
        left, right, top, bottom = min(xx), max(xx) + 1, min(yy), max(yy) + 1
        w, h = (right - left) * step / px_per_mm, (bottom - top) * step / px_per_mm
        fill = len(xx) / ((right - left) * (bottom - top))
        reason = "accepted square"
        if not all(side_mm * MARKER_SCALE_RANGE[0] <= v <= side_mm * MARKER_SCALE_RANGE[1] for v in (w, h)):
            reason = "rejected: size differs from marker"
        elif not .75 <= w / h <= 1.33:
            reason = "rejected: not square"
        elif fill < .8:
            reason = "rejected: interior is not filled"
        elif left == 0 or top == 0 or right == cols or bottom == rows:
            reason = "rejected: boundary clipped"
        pad = max(1, round(px_per_mm / step))
        outer = coarse[max(0, top - pad):min(rows, bottom + pad), max(0, left - pad):min(cols, right + pad)]
        border = np.ones((bottom - top + 2 * pad, right - left + 2 * pad), bool)
        border[pad:-pad, pad:-pad] = False
        ring = float(np.median(outer[border])) if outer.shape == border.shape else paper
        ink_contrast = ring - float(coarse[top:bottom, left:right].mean())
        if reason == "accepted square" and ink_contrast < MIN_MARKER_CONTRAST:
            reason = "rejected: insufficient contrast with paper moat"
        px, py = x0 + left * step, y0 + top * step
        window = gray[py:y0 + bottom * step, px:x0 + right * step]
        weights = np.maximum(ring - window, 0)
        iy, ix = np.indices(window.shape)
        total = float(weights.sum())
        cx = px + float(((ix + .5) * weights).sum() / total) if total else px + window.shape[1] / 2
        cy = py + float(((iy + .5) * weights).sum() / total) if total else py + window.shape[0] / 2
        distance = float(np.hypot(cx if corner.endswith("left") else width - cx,
                                  cy if corner.startswith("top") else height - cy))
        candidates.append({"box_px": [px, py, window.shape[1], window.shape[0]], "centre_px": [cx, cy],
                           "width_mm": w, "height_mm": h, "fill": fill, "contrast": ink_contrast,
                           "reason": reason, "distance_px": distance})
    return sorted(candidates, key=lambda candidate: candidate["distance_px"])


def _find_marker(gray: np.ndarray, corner: str, px_per_mm: float, side_mm: float = 8.0,
                 diagnostics: list | None = None) -> tuple[float, float]:
    candidates = _marker_candidates(gray, corner, px_per_mm, side_mm)
    accepted = [c for c in candidates if c["reason"] == "accepted square"]
    if diagnostics is not None:
        diagnostics.extend(candidates)
    if not accepted:
        raise ScanError("markers_not_found", f"no isolated, filled corner marker found near the {corner.replace('_', ' ')} corner")
    if len(accepted) > 1 and accepted[1]["distance_px"] - accepted[0]["distance_px"] < 2 * px_per_mm:
        raise ScanError("markers_not_found", f"ambiguous square markers near the {corner.replace('_', ' ')} corner")
    accepted[0]["selected"] = True
    return tuple(accepted[0]["centre_px"])


def _homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The 3 x 3 perspective transform taking source points (n x 2) to target points (n >= 4)."""
    rows, values = [], []
    for (x, y), (u, v) in zip(source, target):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        values += [u, v]
    h = np.linalg.lstsq(np.array(rows, float), np.array(values, float), rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)


def _fit_error_mm(mm_points: np.ndarray, px_points: np.ndarray) -> float:
    """Largest distance in mm between a found marker and where the other three put it.

    A homography through 4 points always fits exactly, so it cannot show a misplaced marker.
    A scanner's page is affine (shift, rotation, scale, shear), and 3 markers fix an affine map
    exactly: so each marker is predicted from the other three, and the worst miss is the error.
    """
    errors = []
    for k in range(len(mm_points)):
        others = np.arange(len(mm_points)) != k
        affine = np.linalg.lstsq(np.column_stack([mm_points[others], np.ones(others.sum())]),
                                 px_points[others], rcond=None)[0]
        miss = np.linalg.norm(np.append(mm_points[k], 1) @ affine - px_points[k])
        errors.append(miss / np.sqrt(abs(np.linalg.det(affine[:2]))))
    return float(max(errors))


def _warp(gray: np.ndarray, mm_to_px: np.ndarray) -> np.ndarray:
    """The page resampled onto the manifest's mm grid at PX_PER_MM (bilinear; outside the scan is white).

    gray is grays (height x width) or colours (height x width x channels), 0..1 or uint8.
    """
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255
    channels = gray.shape[2:]
    out = np.empty((PAGE_H * PX_PER_MM, PAGE_W * PX_PER_MM) + channels, dtype=np.float32)
    x_mm = (np.arange(out.shape[1]) + 0.5) / PX_PER_MM
    height, width = gray.shape[:2]
    padded = np.pad(gray, ((1, 1), (1, 1)) + ((0, 0),) * len(channels), constant_values=1.0)
    for start in range(0, out.shape[0], 256):                   # in bands, to keep memory small
        y_mm = (np.arange(start, min(start + 256, out.shape[0])) + 0.5) / PX_PER_MM
        xx, yy = np.meshgrid(x_mm, y_mm)
        u, v, w = (mm_to_px @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)]))
        px = np.clip(u / w - 0.5 + 1, 0, width)                 # +1 for the padding, -0.5 for pixel centres
        py = np.clip(v / w - 0.5 + 1, 0, height)
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = (px - x0).reshape((-1,) + (1,) * len(channels)), (py - y0).reshape((-1,) + (1,) * len(channels))
        band = (padded[y0, x0] * (1 - fx) * (1 - fy) + padded[y0, x0 + 1] * fx * (1 - fy)
                + padded[y0 + 1, x0] * (1 - fx) * fy + padded[y0 + 1, x0 + 1] * fx * fy)
        out[start:start + len(y_mm)] = band.reshape((len(y_mm), -1) + channels)
    return out


def _region(page: np.ndarray, box: Box, inset: float) -> np.ndarray:
    """Pixels of a lined-up page inside an mm box, shrunk by inset mm on every side."""
    x, y, w, h = box
    return page[int(round((y + inset) * PX_PER_MM)):int(round((y + h - inset) * PX_PER_MM)),
                int(round((x + inset) * PX_PER_MM)):int(round((x + w - inset) * PX_PER_MM))]


def _centre(box: Box) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


# ---------------------------------------------------------------- 3. identify the page

def _read_page_code(page: np.ndarray, page_code_regions: list[dict]) -> list[int]:
    """The 6 page code bits: a filled (black) square is 1."""
    return [int(_region(page, square["box"], 0.7).mean() < DARK) for square in page_code_regions]


def _align_page(gray: np.ndarray, rgb: np.ndarray, dpi: float, manifest_pages: list[dict], index: int,
                diagnostics: dict | None = None) -> dict:
    """Line up one scanned page; resolve quarter-turn orientation from its page code.

    The page code holds only the page number and the round, not the trial id: a scan of
    another trial sheet of the same round would also match. Keep scans with their manifest.
    """
    px_per_mm = dpi / MM_PER_INCH
    regions = manifest_pages[0]["regions"]                       # markers and code squares sit alike on every page
    marker_mm = {marker["corner"]: _centre(marker["box"]) for marker in regions["markers"]}
    side_mm = float(np.mean([marker["box"][2] for marker in regions["markers"]]))
    found = {}
    for corner in IMAGE_CORNERS:
        choices = [] if diagnostics is not None else None
        if diagnostics is not None:
            diagnostics.setdefault("candidates", {})[corner] = choices
        found[corner] = _find_marker(gray, corner, px_per_mm, side_mm, choices)
    tried = []
    orientations = ((0, {c: c for c in IMAGE_CORNERS}), (180, UPSIDE_DOWN),
                    (90, dict(zip(IMAGE_CORNERS, ("top_right", "bottom_right", "top_left", "bottom_left")))),
                    (270, dict(zip(IMAGE_CORNERS, ("bottom_left", "top_left", "bottom_right", "top_right")))))
    for orientation, corner_in_image in orientations:
        mm_points = np.array([marker_mm[corner] for corner in IMAGE_CORNERS])
        px_points = np.array([found[corner_in_image[corner]] for corner in IMAGE_CORNERS])
        mm_to_px = _homography(mm_points, px_points)
        page = _warp(gray, mm_to_px)
        code = _read_page_code(page, regions["page_code"])
        tried.append(code)
        matches = [p for p in manifest_pages if p["page_code"] == code]
        if matches:
            break
    else:
        raise ScanError("page_mismatch",
                        f"scan page {index + 1}: page codes {tried} at quarter-turn orientations match no page "
                        f"of this trial {[p['page_code'] for p in manifest_pages]}; is it another round's sheet?")

    error = _fit_error_mm(mm_points, px_points)
    if error > MAX_FIT_ERROR_MM:
        raise ScanError("markers_not_found",
                        f"scan page {index + 1}: the corner markers fit only within {error:.2f} mm "
                        f"(limit {MAX_FIT_ERROR_MM} mm); rescan the page flat and straight")
    basis = np.array([(px_points[1] - px_points[0]) / np.linalg.norm(mm_points[1] - mm_points[0]),
                      (px_points[2] - px_points[0]) / np.linalg.norm(mm_points[2] - mm_points[0])])
    scales = np.linalg.norm(basis, axis=1) / px_per_mm
    if (np.linalg.det(basis) <= 0 or not all(MARKER_SCALE_RANGE[0] <= scale <= MARKER_SCALE_RANGE[1] for scale in scales)
            or abs(np.dot(*basis) / np.prod(np.linalg.norm(basis, axis=1))) > .2):
        raise ScanError("markers_not_found", f"scan page {index + 1}: implausible marker scale, reflection or shear")
    marker_contrasts = []
    for marker in matches[0]["regions"]["markers"]:
        x, y, w, h = marker["box"]
        inner = _region(page, marker["box"], 1.0)
        outer = _region(page, [x - 1, y - 1, w + 2, h + 2], 0)
        border = np.ones(outer.shape, bool)
        border[PX_PER_MM:-PX_PER_MM, PX_PER_MM:-PX_PER_MM] = False
        contrast = float(np.median(outer[border]) - inner.mean())
        marker_contrasts.append(contrast)
        if contrast < MIN_MARKER_CONTRAST:
            raise ScanError("markers_not_found",
                            f"scan page {index + 1}: aligned {marker['corner']} marker lacks contrast with its paper moat")
    # Do not silently fill clipped measurement regions with _warp's white padding.
    for group in matches[0]["regions"].values():
        for region in group if isinstance(group, list) else [group] if isinstance(group, dict) else []:
            if "box" not in region:
                continue
            x, y, w, h = region["box"]
            corners = mm_to_px @ np.array([[x, x + w, x + w, x], [y, y, y + h, y + h], [1, 1, 1, 1]])
            corners = corners[:2] / corners[2]
            if (corners.min() < 0 or corners[0].max() > gray.shape[1] or corners[1].max() > gray.shape[0]):
                raise ScanError("markers_not_found", f"scan page {index + 1}: a required manifest region is clipped")
    if diagnostics is not None:
        diagnostics.update(mm_to_px=mm_to_px.tolist(), page_number=matches[0]["page_number"],
                           orientation=orientation, fit_error_mm=error, scale=scales.tolist(),
                           marker_contrasts=marker_contrasts, page_code_read=code)
    return {"image": page, "rgb": _warp(rgb, mm_to_px), "manifest_page": matches[0], "scan_index": index,
            "dpi": dpi, "orientation": orientation, "fit_error_mm": error, "page_code_read": code}


# ---------------------------------------------------------------- 4. printer response

def _colour(grays: list[float], rgbs: list[np.ndarray]) -> dict | None:
    """{gray, rgb}: the mean of measured areas, as read (nothing stretched); None without areas."""
    if not grays:
        return None
    return {"gray": _round(np.mean(grays)), "rgb": [_round(v) for v in np.mean(rgbs, axis=0)]}


def _printer_response(page: dict, regions: dict) -> dict:
    """Measured paper and black patch (gray and RGB) and every gray patch on one lined-up page."""
    gray, rgb = page["image"], page["rgb"]
    paper_boxes = [area["box"] for area in regions["paper_refs"]]
    paper = _colour([float(_region(gray, box, 1.5).mean()) for box in paper_boxes],
                    [_region(rgb, box, 1.5).reshape(-1, 3).mean(axis=0) for box in paper_boxes])
    black = None
    if regions["black_patch"]:
        box = regions["black_patch"]["box"]
        black = _colour([float(_region(gray, box, 1.5).mean())], [_region(rgb, box, 1.5).reshape(-1, 3).mean(axis=0)])
    patches = []
    for patch in regions["gray_patches"]:
        pixels = _region(gray, patch["box"], 2.0)
        patches.append({"level": patch["level"], "measured": _round(pixels.mean()), "std": _round(pixels.std())})
    return {"paper": paper, "black": black, "gray_patches": patches}


def _page_record(page: dict, response: dict) -> dict:
    """The page's identity, alignment and printer response, with its issues and reasons for nulls."""
    manifest_page = page["manifest_page"]
    paper, black = response["paper"], response["black"]
    contrast = _round(paper["gray"] - black["gray"]) if paper and black else None
    notes, issues = {}, []
    if paper is None:
        notes["paper"] = "this page has no blank paper area"
    if black is None:
        notes["black"] = "this page has no black patch"
    if contrast is None:
        notes["paper_to_black_contrast"] = "needs both the paper and the black patch"
    elif contrast < LOW_CONTRAST:
        issues.append("low paper-to-black contrast")
    return {"page_number": manifest_page["page_number"], "scan_index": page["scan_index"],
            "page_code_read": page["page_code_read"], "page_code_expected": manifest_page["page_code"],
            "code_matches": page["page_code_read"] == manifest_page["page_code"],
            "aligned": True, "fit_error_mm": _round(page["fit_error_mm"], 3), "dpi": _round(page["dpi"], 1),
            "orientation": page["orientation"],
            "paper": paper, "black": black, "paper_to_black_contrast": contrast,
            "gray_patches": response["gray_patches"], "issues": issues, "notes": notes}


def _combined_response(pages: list[dict]) -> dict:
    """The printer response averaged over all pages: paper, black and intended level -> measured gray."""
    levels = [patch["level"] for patch in pages[0]["gray_patches"]]
    measured = np.array([[patch["measured"] for patch in page["gray_patches"]] for page in pages])
    combined, notes = {}, {}
    for key in ("paper", "black"):
        found = [page[key] for page in pages if page[key] is not None]
        combined[key] = ({"gray": _round(np.mean([v["gray"] for v in found])),
                          "rgb": [_round(v) for v in np.mean([v["rgb"] for v in found], axis=0)]} if found else None)
        if not found:
            notes[key] = f"no page has a {'blank paper area' if key == 'paper' else 'black patch'}"
    return combined | {"gray_patches": [{"level": level, "measured": _round(value)}
                                        for level, value in zip(levels, measured.mean(axis=0))], "notes": notes}


# ---------------------------------------------------------------- 5. tiles

def _cell_grays(page: np.ndarray, box: Box, shape: tuple[int, int]) -> np.ndarray:
    """Measured gray at the centre of every data cell: the mean of the middle half of the cell."""
    rows, cols = shape
    x, y, w, h = box
    grays = np.empty(shape)
    for r in range(rows):
        for c in range(cols):
            cell = [x + (c + 0.25) * w / cols, y + (r + 0.25) * h / rows, 0.5 * w / cols, 0.5 * h / rows]
            pixels = _region(page, cell, 0.0)
            grays[r, c] = pixels.mean() if pixels.size else np.nan
    return grays


def _cell_masks(box: Box, shape: tuple[int, int], marks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(usable, inside): cells not crossed by a box mark's outline, and usable cells inside a mark."""
    rows, cols = shape
    x, y, w, h = box
    cx = x + (np.arange(cols) + 0.5) * w / cols
    cy = y + (np.arange(rows) + 0.5) * h / rows
    xx, yy = np.meshgrid(cx, cy)
    reach = 0.25 * min(w / cols, h / rows) + MARK_LINE_MM
    usable, inside = np.ones(shape, bool), np.zeros(shape, bool)
    for mark in marks:
        half = mark["size"] / 2
        x0, x1 = x + (mark["col"] + 0.5 - half) * w / cols, x + (mark["col"] + 0.5 + half) * w / cols
        y0, y1 = y + (mark["row"] + 0.5 - half) * h / rows, y + (mark["row"] + 0.5 + half) * h / rows
        within = (xx > x0) & (xx < x1) & (yy > y0) & (yy < y1)
        dx = np.where(within, np.minimum(xx - x0, x1 - xx), np.maximum(np.maximum(x0 - xx, xx - x1), 0))
        dy = np.where(within, np.minimum(yy - y0, y1 - yy), np.maximum(np.maximum(y0 - yy, yy - y1), 0))
        to_outline = np.where(within, np.minimum(dx, dy), np.hypot(dx, dy))
        usable &= to_outline > reach
        inside |= within
    return usable, inside & usable


def _printed_levels(intended: np.ndarray, measured: np.ndarray) -> tuple[int, int]:
    """(intended levels, printed levels): the intended grays grouped into LEVEL_BINS levels, and
    how many of them stay apart on paper. Neighbouring levels whose mean printed grays differ by
    less than SAME_GRAY have collapsed into one printed gray: their detail is lost.
    """
    bins = np.clip((intended * LEVEL_BINS).astype(int), 0, LEVEL_BINS - 1)
    occupied = np.unique(bins)
    means = np.array([measured[bins == b].mean() for b in occupied])
    return len(occupied), 1 + int(np.sum(np.diff(means) >= SAME_GRAY))


def _tile_measurements(measured: np.ndarray, intended: np.ndarray, usable: np.ndarray, inside: np.ndarray,
                       paper: float, black: float | None) -> tuple[dict, dict]:
    """(measurements, notes): one panel's measurements, and why any of them is null."""
    m, i = measured[usable], intended[usable]
    correlation = float(np.corrcoef(i, m)[0, 1]) if m.size > 2 and i.std() > 1e-9 and m.std() > 1e-9 else None
    levels, printed = _printed_levels(i, m)
    notes = {}
    if paper is None:
        notes["paper_white_share"] = "this page has no blank paper area to compare with"
    if black is None:
        notes["full_black_share"] = "this page has no black patch to compare with"
    if correlation is None:
        notes["correlation"] = ("too few cells" if m.size <= 2 else
                                "the intended grays are all the same" if i.std() <= 1e-9 else
                                "the printed grays are all the same")
    result = {
        "cells_used": int(m.size),
        "paper_white_share": _round(np.mean(m >= paper - NEAR_PAPER)) if paper is not None else None,
        "full_black_share": _round(np.mean(m <= black + NEAR_BLACK)) if black is not None else None,
        "intended_levels": levels,
        "printed_levels": printed,
        "collapsed_levels": levels - printed,
        "correlation": _round(correlation),
        "mean_abs_difference": _round(np.mean(np.abs(m - i))),
        "contrast": _round(np.percentile(m, 95) - np.percentile(m, 5)),
        "intended_contrast": _round(np.percentile(i, 95) - np.percentile(i, 5)),
        "focus": None,
    }
    outside = usable & ~inside
    if inside.any() and outside.any():
        result["focus"] = {"inside_mean": _round(measured[inside].mean()), "rest_mean": _round(measured[outside].mean()),
                           "difference": _round(measured[inside].mean() - measured[outside].mean()),
                           "intended_difference": _round(intended[inside].mean() - intended[outside].mean())}
    else:
        notes["focus"] = ("no focus box in this tile" if not inside.any() else
                          "the focus box covers every measured cell, so there is nothing to compare it with")
    return result, notes


def _candidate_measurements(panels: list[dict]) -> tuple[dict, dict]:
    """(measurements, notes): one candidate's panels combined; a value is null if any panel's is."""
    values = [panel["measurements"] for panel in panels]
    weights = np.array([v["cells_used"] for v in values], float)
    notes = {}

    def share(key):
        if any(v[key] is None for v in values):
            notes[key] = next(panel["notes"][key] for panel in panels if panel["measurements"][key] is None)
            return None
        return _round(np.average([v[key] for v in values], weights=weights))

    correlations = [v["correlation"] for v in values if v["correlation"] is not None]
    if not correlations:
        notes["correlation"] = "no panel has a correlation: " + "; ".join(
            f"{panel['panel']}: {panel['notes']['correlation']}" for panel in panels)
    focus = {panel["panel"]: panel["measurements"]["focus"]["difference"]
             for panel in panels if panel["measurements"]["focus"]}
    if not focus:
        notes["focus_difference"] = panels[0]["notes"]["focus"]
    measurements = {
        "paper_white_share": share("paper_white_share"), "full_black_share": share("full_black_share"),
        "collapsed_levels": sum(v["collapsed_levels"] for v in values),
        "correlation": _round(np.mean(correlations)) if correlations else None,
        "contrast": _round(np.mean([v["contrast"] for v in values])),
        "focus_difference": focus or None,
    }
    return measurements, notes


def _warnings(measurements: dict) -> list[str]:
    """Short warnings about how a candidate printed (the thresholds are constants at the top)."""
    warnings = []
    if measurements["full_black_share"] is not None and measurements["full_black_share"] > MANY_FULL_BLACK:
        warnings.append("many cells print as full black")
    if measurements["paper_white_share"] is not None and measurements["paper_white_share"] > MANY_PAPER_WHITE:
        warnings.append("many cells print as paper white")
    if measurements["collapsed_levels"] > 0:
        warnings.append(f"{measurements['collapsed_levels']} gray levels merge on paper")
    if measurements["correlation"] is None or measurements["correlation"] < MIN_CORRELATION:
        warnings.append("printed grays do not follow the intended grays well")
    return warnings


def _tile_crop_box(tile: dict, labels: list[dict]) -> Box:
    """The tile's box, grown to hold its printed labels if any reach outside it."""
    boxes = [tile["box"]] + [label["box"] for label in labels
                             if label.get("item_id") == tile["item_id"]
                             and label.get("variation_id") == tile["variation_id"]]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


# ---------------------------------------------------------------- 6. save

def _new_output(out_dir: Path, round_id: int, trial_id: str) -> tuple[Path, Path]:
    """A JSON path and tile folder that do not exist yet."""
    while True:
        stem = f"scan_r{round_id}_{trial_id[:8]}_{uuid.uuid4().hex[:8]}"
        json_path, tile_dir = out_dir / f"{stem}.json", out_dir / f"{stem}_tiles"
        if not json_path.exists() and not tile_dir.exists():
            return json_path, tile_dir


def _save_png(pixels: np.ndarray, path: Path) -> None:
    """Save grays (height x width) or RGB (height x width x 3), 0..1, as they are."""
    Image.fromarray(np.clip(np.round(pixels * 255), 0, 255).astype(np.uint8), "RGB" if pixels.ndim == 3 else "L").save(path)


def _png_name(*parts) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", "__".join(str(part) for part in parts) + ".png")


# ---------------------------------------------------------------- main

def diagnose_scan_alignment(scan_path, manifest_path, out_dir) -> dict:
    """Save candidate/geometry overlays and JSON without updating a session.

    Orange boxes are nominal manifest positions at scan DPI; cyan boxes are the
    accepted geometric transform. Green candidates were selected, blue are other
    valid squares, and red candidates were rejected. Candidate IDs refer to the
    JSON's corner lists, which retain size/fill/contrast rejection reasons.
    Detection and alignment use the same helpers as evaluate_scan().
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scanned, files = _read_scan(scan_path, manifest["config"]["scan"]["accepted_formats"])
    output = Path(out_dir).resolve() / f"alignment_{manifest['trial_id'][:8]}_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True)
    records = []
    for index, (gray, rgb, dpi) in enumerate(scanned):
        record = {"scan_index": index, "dpi": dpi, "width_px": gray.shape[1], "height_px": gray.shape[0],
                  "candidates": {c: _marker_candidates(gray, c, dpi / MM_PER_INCH) for c in IMAGE_CORNERS}}
        try:
            aligned = _align_page(gray, rgb, dpi, manifest["pages"], index, record)
            record["status"] = "ok"
            record["legacy_marker_mean_gray"] = [float(_region(aligned["image"], m["box"], 1).mean())
                                                  for m in aligned["manifest_page"]["regions"]["markers"]]
        except ScanError as error:
            record.update(status="error", error_code=error.code, message=str(error))
        # Put explanations beside the scan so annotations do not obscure evidence.
        source = Image.fromarray(rgb)
        image = Image.new("RGB", (source.width + 850, source.height), "white")
        image.paste(source, (0, 0))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=22)
        draw.text((source.width + 20, 25), "Orange: nominal; cyan: fitted manifest", fill="black", font=font)
        draw.text((source.width + 20, 60), "Green: selected; blue: valid; red: rejected", fill="black", font=font)
        legend_y = 110
        for marker in manifest["pages"][0]["regions"]["markers"]:
            x, y, w, h = np.asarray(marker["box"]) * dpi / MM_PER_INCH
            draw.rectangle((x, y, x + w, y + h), outline="orange", width=4)
            if "mm_to_px" in record:
                x, y, w, h = marker["box"]
                points = np.asarray(record["mm_to_px"]) @ np.array(
                    [[x, x + w, x + w, x], [y, y, y + h, y + h], [1, 1, 1, 1]])
                points = (points[:2] / points[2]).T
                draw.line([tuple(p) for p in points] + [tuple(points[0])], fill="cyan", width=4)
        for corner, candidates in record["candidates"].items():
            short = "".join(word[0].upper() for word in corner.split("_"))
            draw.text((source.width + 20, legend_y), corner, fill="black", font=font)
            legend_y += 32
            for number, candidate in enumerate(candidates):
                x, y, w, h = candidate["box_px"]
                color = "green" if candidate.get("selected") else "blue" if candidate["reason"] == "accepted square" else "red"
                draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
                label = f"{short}{number}"
                draw.text((x, max(0, y - 25)), label, fill=color, font=font)
                draw.text((source.width + 20, legend_y), label + ": " + candidate["reason"], fill=color, font=font)
                legend_y += 30
        path = output / f"scan_{index + 1}_overlay.png"
        image.save(path)
        record["overlay_path"] = str(path)
        records.append(record)
    report = {"trial_id": manifest["trial_id"], "manifest_path": str(Path(manifest_path).resolve()),
              "scan_files": files, "pages": records, "diagnostic_path": str(output / "alignment.json")}
    Path(report["diagnostic_path"]).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def evaluate_scan(scan_path, manifest_path, out_dir=None) -> dict:
    """Line up the scan with the trial manifest, measure every patch and tile, save and return the result.

    scan_path is one file (PDF with all pages, or one image) or a list of files, one per page.
    Nothing is written unless every page is found, identified and lined up.
    """
    manifest_path = Path(manifest_path)
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_pages, scan_config = record["pages"], record["config"]["scan"]
        cells_path = _local(record["cells_path"], manifest_path.parent)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ScanError("manifest_missing", f"cannot read the trial manifest {manifest_path}: {error}") from error
    if not cells_path.is_file():
        raise ScanError("cells_missing", f"the trial's cells file {cells_path.name} is missing")

    # 1-3: read, line up and identify every page before anything is written
    scanned, files = _read_scan(scan_path, scan_config["accepted_formats"])
    for index, (_, _, dpi) in enumerate(scanned):
        if dpi < scan_config["min_dpi"]:
            raise ScanError("low_dpi", f"scan page {index + 1} is about {dpi:.0f} dpi; at least "
                            f"{scan_config['min_dpi']} dpi is needed ({scan_config.get('recommended_dpi')} recommended)")
    pages = [_align_page(gray, rgb, dpi, manifest_pages, index) for index, (gray, rgb, dpi) in enumerate(scanned)]
    numbers = [page["manifest_page"]["page_number"] for page in pages]
    if len(set(numbers)) != len(numbers):
        raise ScanError("page_mismatch", f"a page was scanned more than once: pages {numbers}")
    missing = sorted({p["page_number"] for p in manifest_pages} - set(numbers))
    if missing:
        raise ScanError("page_missing", f"page(s) {missing} of this trial are missing from the scan")
    pages.sort(key=lambda p: p["manifest_page"]["page_number"])

    # 4-5: measure
    marks = {(item["item_id"], panel["name"]): panel.get("marks", []) for item in record["items"] for panel in item["panels"]}
    out_dir = Path(out_dir) if out_dir else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, tile_dir = _new_output(out_dir, record["round_id"], record["trial_id"])
    tile_dir.mkdir()
    page_records, found = [], {}                # found: (item_id, variation_id) -> tile crop and its panels
    bilevel_pages = {index for file in files for index in file.get("bilevel_scan_indices", [])}
    with np.load(cells_path) as cells:
        for page in pages:
            regions, number = page["manifest_page"]["regions"], page["manifest_page"]["page_number"]
            page_record = _page_record(page, _printer_response(page, regions))
            if page["scan_index"] in bilevel_pages:
                page_record["issues"].append("bilevel (1-bit) scan: scanner thresholding/halftoning limits grayscale-tone evidence")
            page_records.append(page_record)
            paper = page_record["paper"]["gray"] if page_record["paper"] else None
            black = page_record["black"]["gray"] if page_record["black"] else None
            for tile in regions["tiles"]:
                png = tile_dir / _png_name(f"p{number}", tile["item_id"], tile["variation_id"], "tile")
                _save_png(_region(page["rgb"], _tile_crop_box(tile, regions["labels"]), 0.0), png)
                found[tile["item_id"], tile["variation_id"]] = {"page_number": number, "tile_path": str(png),
                                                                "panels": []}
            for plot in regions["plots"]:
                key = f"{plot['item_id']}/{plot['variation_id']}/{plot['panel']}"
                shape = tuple(plot["grid_shape"])
                measured = _cell_grays(page["image"], plot["box"], shape)
                usable, inside = _cell_masks(plot["box"], shape, marks.get((plot["item_id"], plot["panel"]), []))
                png = tile_dir / re.sub(r"[^A-Za-z0-9_.-]", "_", f"p{number}_{key.replace('/', '__')}.png")
                _save_png(_region(page["image"], plot["box"], TILE_INSET_MM), png)
                measurements, notes = _tile_measurements(measured, cells[key], usable, inside, paper, black)
                found[plot["item_id"], plot["variation_id"]]["panels"].append(
                    {"panel": plot["panel"], "png": str(png), "measurements": measurements, "notes": notes})

    # one candidate per experiment and variation, in manifest order, unranked
    candidates = []
    for item in record["items"]:
        for variation in record["variations"][item["item_id"]]:
            tile = found[item["item_id"], variation["variation_id"]]
            measurements, notes = _candidate_measurements(tile["panels"])
            candidates.append({
                "experiment_id": item["item_id"], "candidate_id": variation["variation_id"], "role": variation["role"],
                "mapping": variation["mapping"], "page_number": tile["page_number"], "tile_path": tile["tile_path"],
                "panels": tile["panels"], "measurements": measurements, "warnings": _warnings(measurements),
                "notes": notes,
            })

    result = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": json_path.stem.rsplit("_", 1)[-1],
        "session_id": record.get("calibration_session_id"),
        "trial_id": record["trial_id"],
        "round_id": record["round_id"],
        **({"cycle_id": record["cycle_id"], "step_id": record["step_id"]} if "cycle_id" in record else {}),
        "tested_parameter": record["tested_parameter"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "scan_files": files,
        "scan_quality": {"usable": True, "issues": [{"page_number": page["page_number"], "issue": issue}
                                                    for page in page_records for issue in page["issues"]]},
        "pages": page_records,
        "printer_response": _combined_response(page_records),
        "candidates": candidates,
        "measurement_docs": MEASUREMENT_DOCS,
        "result_path": str(json_path),
        "evaluation_path": str(json_path),
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure a scanned trial sheet against its manifest.")
    parser.add_argument("scan", nargs="+", type=Path, help="the scan: one PDF or image, or one image per page")
    parser.add_argument("manifest", type=Path, help="the trial_r..._manifest.json of the printed sheet")
    parser.add_argument("--out", type=Path, help="output folder (default: the manifest's folder)")
    args = parser.parse_args(argv)
    try:
        result = evaluate_scan(args.scan if len(args.scan) > 1 else args.scan[0], args.manifest, args.out)
    except ScanError as error:
        sys.exit(f"error [{error.code}{', rescan needed' if error.needs_rescan else ''}]: {error}")

    print(f"Trial {result['trial_id'][:8]} · round {result['round_id']} · testing {result['tested_parameter']}")
    for page in result["pages"]:
        print(f"  page {page['page_number']}: scan page {page['scan_index'] + 1}, {page['dpi']:.0f} dpi, "
              f"turned {page['orientation']}°, marker fit {page['fit_error_mm']:.2f} mm, "
              f"paper-to-black contrast {page['paper_to_black_contrast']}"
              + (f"  ({'; '.join(page['issues'])})" if page["issues"] else ""))
    combined = result["printer_response"]
    paper, black = combined["paper"], combined["black"]
    print(f"Printer: paper {paper and paper['gray']} (RGB {paper and paper['rgb']}), "
          f"black {black and black['gray']}, grays "
          + " ".join(f"{p['level']:.1f}->{p['measured']:.2f}" for p in combined["gray_patches"]))
    for candidate in result["candidates"]:
        values = candidate["measurements"]
        print(f"  {candidate['experiment_id'][:20]:20} {candidate['candidate_id']:9} {candidate['role']:9} "
              f"corr {values['correlation']}  white {values['paper_white_share']}  "
              f"black {values['full_black_share']}  collapsed {values['collapsed_levels']}"
              + (f"  ! {'; '.join(candidate['warnings'])}" if candidate["warnings"] else ""))
    print("Result:", result["evaluation_path"])


if __name__ == "__main__":
    main()
