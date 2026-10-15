"""Read a scanned trial sheet, line it up with its manifest and measure how every tile printed.

trial_sheet.py writes a trial PDF, a manifest (every region in mm from the paper's top-left
corner) and a cells .npz (the intended gray of every data cell). After the sheet is printed and
scanned, evaluate_scan():
  1. reads the scan (PDF, PNG or JPG; one file with all pages or one image per page),
  2. finds the 4 corner markers on each page and maps the page onto the manifest's mm grid,
  3. identifies each page by its page code,
  4. measures the paper, black and gray patches: how this printer really prints gray,
  5. cuts out every tile and compares its printed cell grays with the intended ones,
  6. saves the measurements as JSON plus one PNG per tile.
It only measures; choosing settings is left to the agents. Grays run from 0 (black) to 1 (white).

Command line:
    python scan_evaluation.py SCAN_FILE [MORE_PAGES ...] MANIFEST [--out DIR]
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
from PIL import Image, ImageOps

SCHEMA_VERSION = "scan_eval/1"
PAGE_W, PAGE_H = 210, 297
MM_PER_INCH = 25.4
PDF_DPI = 300                      # PDF scans are rendered at this resolution
PX_PER_MM = 10                     # every page is resampled onto this grid
MAX_FIT_ERROR_MM = 1.0
DARK = 0.5                         # a pixel below this gray counts as ink
CORNER_SEARCH_MM = 45              # markers are searched this far from each image corner
COARSE_PX_PER_MM = 2               # resolution of the first, rough marker search
NEAR_PAPER, NEAR_BLACK = 0.04, 0.04  # a cell this close to the measured paper / black counts as such
SAME_GRAY = 0.02                   # printed grays closer than this count as one level
LEVEL_BINS = 16                    # intended grays are grouped into this many levels
TILE_INSET_MM = 0.3                # skip the plot's border when cutting it out
MARK_LINE_MM = 0.35                # half width of the box mark's double outline, plus a little

IMAGE_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
UPSIDE_DOWN = {"top_left": "bottom_right", "top_right": "bottom_left",
               "bottom_left": "top_right", "bottom_right": "top_left"}

Box = list[float]


class ScanError(ValueError):
    """The scan cannot be read, lined up or matched to this trial sheet."""


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
    """dpi from the image size, assuming it shows one A4 page in portrait."""
    height, width = gray.shape
    if width > height:
        raise ScanError(f"a scanned page is landscape ({width} x {height} px); scan the sheet in portrait")
    return min(width / (PAGE_W / MM_PER_INCH), height / (PAGE_H / MM_PER_INCH))


def _read_scan(scan_path, accepted: list[str]) -> tuple[list[tuple[np.ndarray, float]], list[dict]]:
    """Every scanned page as (grays 0..1, dpi), and the scan files with their sha256."""
    paths = [Path(p) for p in scan_path] if isinstance(scan_path, (list, tuple)) else [Path(scan_path)]
    if not paths:
        raise ScanError("no scan files given")
    pages, files = [], []
    for path in paths:
        suffix = path.suffix.lower().lstrip(".")
        if not path.is_file():
            raise ScanError(f"scan file not found: {path}")
        if suffix not in accepted:
            raise ScanError(f"{path.name}: scans must be one of {accepted}")
        files.append({"path": str(path.resolve()), "sha256": _sha256(path)})
        try:
            if suffix == "pdf":
                with pymupdf.open(path) as document:
                    for page in document:
                        pixmap = page.get_pixmap(dpi=PDF_DPI, colorspace=pymupdf.csGRAY, alpha=False)
                        samples = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.stride)
                        gray = samples[:, :pixmap.width].astype(np.float32) / 255
                        pages.append((gray, min(_pdf_page_dpi(page), _estimate_dpi(gray))))
            else:
                with Image.open(path) as image:
                    gray = np.asarray(ImageOps.exif_transpose(image).convert("L"), dtype=np.float32) / 255
                pages.append((gray, _estimate_dpi(gray)))
        except ScanError:
            raise
        except Exception as error:
            raise ScanError(f"cannot read {path.name}: {type(error).__name__}: {error}") from error
    return pages, files


# ---------------------------------------------------------------- 2. line up each page

def _box_sum(values: np.ndarray, half: int) -> np.ndarray:
    """Sum over the (2 half + 1)^2 square around every pixel; outside counts as 0."""
    padded = np.pad(values, half + 1)
    total = padded.cumsum(0).cumsum(1)
    size = 2 * half + 1
    return (total[size:, size:] - total[:-size, size:] - total[size:, :-size] + total[:-size, :-size])[
        :values.shape[0], :values.shape[1]]


def _find_marker(gray: np.ndarray, corner: str, px_per_mm: float) -> tuple[float, float]:
    """Centre (x, y) in pixels of the solid 8 mm corner marker nearest this image corner.

    A rough search on a 2 px/mm copy scores each spot by how dark a 6 mm square around it is,
    minus how dark the ring just outside the marker is: only a solid square on white paper
    scores high, not a dark plot. The centre is then the centroid of the ink around that spot.
    """
    height, width = gray.shape
    size = int(CORNER_SEARCH_MM * px_per_mm)
    y0 = 0 if corner.startswith("top") else max(0, height - size)
    x0 = 0 if corner.endswith("left") else max(0, width - size)
    region = gray[y0:y0 + size, x0:x0 + size]

    step = max(1, int(round(px_per_mm / COARSE_PX_PER_MM)))
    rows, cols = region.shape[0] // step, region.shape[1] // step
    dark = 1 - region[:rows * step, :cols * step].reshape(rows, step, cols, step).mean(axis=(1, 3))
    mm = px_per_mm / step                                      # coarse pixels per mm
    inner_h, ring_in, ring_out = (max(1, int(round(r * mm))) for r in (3.0, 5.0, 6.5))
    inner = _box_sum(dark, inner_h) / (2 * inner_h + 1) ** 2
    ring = (_box_sum(dark, ring_out) - _box_sum(dark, ring_in)) / ((2 * ring_out + 1) ** 2 - (2 * ring_in + 1) ** 2)
    score = inner - ring
    if score.max() < 0.5:
        raise ScanError(f"no corner marker found near the {corner.replace('_', ' ')} corner of the scan")
    # the best score is a plateau about 1 mm wide (the 6 mm square fits anywhere inside the
    # 8 mm marker): take its middle, not an arbitrary point of it
    plateau_y, plateau_x = np.nonzero(score >= score.max() - 0.05)
    x, y = x0 + (plateau_x.mean() + 0.5) * step, y0 + (plateau_y.mean() + 0.5) * step

    # refine at full resolution: centroid of the ink within the marker's reach, re-centred until it
    # settles, so the window holds the whole marker and none of its neighbours (5.4 mm away and more)
    reach = 5.2 * px_per_mm
    for _ in range(4):
        ys, xs = slice(max(0, int(y - reach)), int(y + reach)), slice(max(0, int(x - reach)), int(x + reach))
        window = gray[ys, xs]
        ink_y, ink_x = np.nonzero(window < DARK)
        if len(ink_x) < 0.5 * (8 * px_per_mm) ** 2:
            raise ScanError(f"the marker near the {corner.replace('_', ' ')} corner is too small or too faint")
        x, y = xs.start + ink_x.mean() + 0.5, ys.start + ink_y.mean() + 0.5
    # a marker cut off by the edge of the scan has a shifted centre: refuse it
    if (xs.start == 0 and ink_x.min() == 0) or (ys.start == 0 and ink_y.min() == 0) or \
            (xs.start + window.shape[1] == width and ink_x.max() == window.shape[1] - 1) or \
            (ys.start + window.shape[0] == height and ink_y.max() == window.shape[0] - 1):
        raise ScanError(f"the marker near the {corner.replace('_', ' ')} corner is cut off by the edge of the "
                        "scan; place the sheet fully on the scanner glass")
    return x, y


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
    """The page resampled onto the manifest's mm grid at PX_PER_MM (bilinear; outside the scan is white)."""
    out = np.empty((PAGE_H * PX_PER_MM, PAGE_W * PX_PER_MM), dtype=np.float32)
    x_mm = (np.arange(out.shape[1]) + 0.5) / PX_PER_MM
    height, width = gray.shape
    padded = np.pad(gray, 1, constant_values=1.0)
    for start in range(0, out.shape[0], 256):                   # in bands, to keep memory small
        y_mm = (np.arange(start, min(start + 256, out.shape[0])) + 0.5) / PX_PER_MM
        xx, yy = np.meshgrid(x_mm, y_mm)
        u, v, w = (mm_to_px @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)]))
        px = np.clip(u / w - 0.5 + 1, 0, width)                 # +1 for the padding, -0.5 for pixel centres
        py = np.clip(v / w - 0.5 + 1, 0, height)
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - x0, py - y0
        band = (padded[y0, x0] * (1 - fx) * (1 - fy) + padded[y0, x0 + 1] * fx * (1 - fy)
                + padded[y0 + 1, x0] * (1 - fx) * fy + padded[y0 + 1, x0 + 1] * fx * fy)
        out[start:start + len(y_mm)] = band.reshape(len(y_mm), -1)
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


def _align_page(gray: np.ndarray, dpi: float, manifest_pages: list[dict], index: int) -> dict:
    """Line up one scanned page; try it upside down if its page code matches no manifest page.

    The page code holds only the page number and the round, not the trial id: a scan of
    another trial sheet of the same round would also match. Keep scans with their manifest.
    """
    px_per_mm = dpi / MM_PER_INCH
    found = {corner: _find_marker(gray, corner, px_per_mm) for corner in IMAGE_CORNERS}
    regions = manifest_pages[0]["regions"]                       # markers and code squares sit alike on every page
    marker_mm = {marker["corner"]: _centre(marker["box"]) for marker in regions["markers"]}
    tried = []
    for orientation, corner_in_image in ((0, {c: c for c in IMAGE_CORNERS}), (180, UPSIDE_DOWN)):
        mm_points = np.array([marker_mm[corner] for corner in IMAGE_CORNERS])
        px_points = np.array([found[corner_in_image[corner]] for corner in IMAGE_CORNERS])
        page = _warp(gray, _homography(mm_points, px_points))
        code = _read_page_code(page, regions["page_code"])
        tried.append(code)
        matches = [p for p in manifest_pages if p["page_code"] == code]
        if matches:
            break
    else:
        raise ScanError(f"scan page {index + 1}: page code {tried[0]} (or {tried[1]} upside down) matches no page "
                        f"of this trial {[p['page_code'] for p in manifest_pages]}; is it another round's sheet?")

    error = _fit_error_mm(mm_points, px_points)
    if error > MAX_FIT_ERROR_MM:
        raise ScanError(f"scan page {index + 1}: the corner markers fit only within {error:.2f} mm "
                        f"(limit {MAX_FIT_ERROR_MM} mm); rescan the page flat and straight")
    if any(_region(page, marker["box"], 1.0).mean() > 0.35 for marker in matches[0]["regions"]["markers"]):
        raise ScanError(f"scan page {index + 1}: the markers are not dark where the manifest expects them")
    return {"image": page, "manifest_page": matches[0], "scan_index": index, "dpi": dpi,
            "orientation": orientation, "fit_error_mm": error, "page_code_read": code}


# ---------------------------------------------------------------- 4. printer response

def _printer_response(page: np.ndarray, regions: dict) -> dict:
    """Measured gray of the blank paper areas, the black patch and every gray patch on one page."""
    paper = [float(_region(page, area["box"], 1.5).mean()) for area in regions["paper_refs"]]
    black = float(_region(page, regions["black_patch"]["box"], 1.5).mean()) if regions["black_patch"] else None
    patches = []
    for patch in regions["gray_patches"]:
        pixels = _region(page, patch["box"], 2.0)
        patches.append({"level": patch["level"], "measured": _round(pixels.mean()), "std": _round(pixels.std())})
    return {"paper": [_round(v) for v in paper], "paper_mean": _round(np.mean(paper)) if paper else None,
            "black": _round(black), "gray_patches": patches}


def _combined_response(per_page: list[dict]) -> dict:
    """The printer response averaged over all pages: intended level -> measured gray."""
    levels = [patch["level"] for patch in per_page[0]["gray_patches"]]
    measured = np.array([[patch["measured"] for patch in page["gray_patches"]] for page in per_page])
    blacks = [page["black"] for page in per_page if page["black"] is not None]
    return {"paper": _round(np.mean([page["paper_mean"] for page in per_page])),
            "black": _round(np.mean(blacks)) if blacks else None,
            "gray_patches": [{"level": level, "measured": _round(value)}
                             for level, value in zip(levels, measured.mean(axis=0))]}


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
                       paper: float, black: float | None) -> dict:
    m, i = measured[usable], intended[usable]
    correlation = float(np.corrcoef(i, m)[0, 1]) if m.size > 2 and i.std() > 1e-9 and m.std() > 1e-9 else None
    levels, printed = _printed_levels(i, m)
    result = {
        "cells_used": int(m.size),
        "paper_white_share": _round(np.mean(m >= paper - NEAR_PAPER)),
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
    return result


def _variation_summaries(tiles: list[dict], record: dict) -> list[dict]:
    """One summary per (item_id, variation_id), combining its panels, with role and mapping."""
    summaries = []
    for item_id, item_variations in record["variations"].items():
        for variation in item_variations:
            panels = [t for t in tiles if t["item_id"] == item_id and t["variation_id"] == variation["variation_id"]]
            if not panels:
                continue
            weights = np.array([t["measurements"]["cells_used"] for t in panels], float)
            share = lambda key: _round(np.average([t["measurements"][key] or 0 for t in panels], weights=weights))
            correlations = [t["measurements"]["correlation"] for t in panels if t["measurements"]["correlation"] is not None]
            summaries.append({
                "item_id": item_id, "variation_id": variation["variation_id"], "role": variation["role"],
                "mapping": variation["mapping"],
                "paper_white_share": share("paper_white_share"), "full_black_share": share("full_black_share"),
                "collapsed_levels": sum(t["measurements"]["collapsed_levels"] for t in panels),
                "correlation": _round(np.mean(correlations)) if correlations else None,
                "contrast": _round(np.mean([t["measurements"]["contrast"] for t in panels])),
                "focus_difference": {t["panel"]: t["measurements"]["focus"]["difference"]
                                     for t in panels if t["measurements"]["focus"]},
            })
    return summaries


# ---------------------------------------------------------------- 6. save

def _new_output(out_dir: Path, round_id: int, trial_id: str) -> tuple[Path, Path]:
    """A JSON path and tile folder that do not exist yet."""
    while True:
        stem = f"scan_r{round_id}_{trial_id[:8]}_{uuid.uuid4().hex[:8]}"
        json_path, tile_dir = out_dir / f"{stem}.json", out_dir / f"{stem}_tiles"
        if not json_path.exists() and not tile_dir.exists():
            return json_path, tile_dir


def _save_png(pixels: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(np.round(pixels * 255), 0, 255).astype(np.uint8), "L").save(path)


# ---------------------------------------------------------------- main

def evaluate_scan(scan_path, manifest_path, out_dir=None) -> dict:
    """Line up the scan with the trial manifest, measure every patch and tile, save and return the result.

    scan_path is one file (PDF with all pages, or one image) or a list of files, one per page.
    Nothing is written unless every page is found, identified and lined up.
    """
    manifest_path = Path(manifest_path)
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_pages, scan_config = record["pages"], record["config"]["scan"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ScanError(f"cannot read the trial manifest {manifest_path}: {error}") from error
    cells_path = _local(record["cells_path"], manifest_path.parent)
    if not cells_path.is_file():
        raise ScanError(f"the trial's cells file {cells_path.name} is missing")

    # 1-3: read, line up and identify every page before anything is written
    scanned, files = _read_scan(scan_path, scan_config["accepted_formats"])
    for index, (_, dpi) in enumerate(scanned):
        if dpi < scan_config["min_dpi"]:
            raise ScanError(f"scan page {index + 1} is about {dpi:.0f} dpi; at least {scan_config['min_dpi']} "
                            f"dpi is needed ({scan_config.get('recommended_dpi')} recommended)")
    pages = [_align_page(gray, dpi, manifest_pages, index) for index, (gray, dpi) in enumerate(scanned)]
    numbers = [page["manifest_page"]["page_number"] for page in pages]
    if len(set(numbers)) != len(numbers):
        raise ScanError(f"a page was scanned more than once: pages {numbers}")
    missing = sorted({p["page_number"] for p in manifest_pages} - set(numbers))
    if missing:
        raise ScanError(f"page(s) {missing} of this trial are missing from the scan")

    # 4-5: measure
    marks = {(item["item_id"], panel["name"]): panel.get("marks", []) for item in record["items"] for panel in item["panels"]}
    out_dir = Path(out_dir) if out_dir else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, tile_dir = _new_output(out_dir, record["round_id"], record["trial_id"])
    tile_dir.mkdir()
    responses, tiles = [], []
    with np.load(cells_path) as cells:
        for page in sorted(pages, key=lambda p: p["manifest_page"]["page_number"]):
            regions, number = page["manifest_page"]["regions"], page["manifest_page"]["page_number"]
            response = _printer_response(page["image"], regions) | {"page_number": number}
            responses.append(response)
            for plot in regions["plots"]:
                key = f"{plot['item_id']}/{plot['variation_id']}/{plot['panel']}"
                shape = tuple(plot["grid_shape"])
                measured = _cell_grays(page["image"], plot["box"], shape)
                usable, inside = _cell_masks(plot["box"], shape, marks.get((plot["item_id"], plot["panel"]), []))
                png = tile_dir / re.sub(r"[^A-Za-z0-9_.-]", "_", f"p{number}_{key.replace('/', '__')}.png")
                _save_png(_region(page["image"], plot["box"], TILE_INSET_MM), png)
                tiles.append({"page_number": number, "item_id": plot["item_id"], "variation_id": plot["variation_id"],
                              "panel": plot["panel"], "box": plot["box"], "png": str(png),
                              "measurements": _tile_measurements(measured, cells[key], usable, inside,
                                                                 response["paper_mean"], response["black"])})

    result = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": json_path.stem.rsplit("_", 1)[-1],
        "trial_id": record["trial_id"],
        "round_id": record["round_id"],
        "tested_parameter": record["tested_parameter"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "scan_files": files,
        "pages": [{"page_number": p["manifest_page"]["page_number"], "scan_index": p["scan_index"],
                   "dpi": _round(p["dpi"], 1), "orientation": p["orientation"],
                   "fit_error_mm": _round(p["fit_error_mm"], 3), "page_code_read": p["page_code_read"]}
                  for p in sorted(pages, key=lambda p: p["manifest_page"]["page_number"])],
        "printer_response": {"pages": responses, "combined": _combined_response(responses)},
        "tiles": tiles,
        "variations": _variation_summaries(tiles, record),
        "result_path": str(json_path),
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
        sys.exit(f"error: {error}")

    print(f"Trial {result['trial_id'][:8]} · round {result['round_id']} · testing {result['tested_parameter']}")
    for page in result["pages"]:
        print(f"  page {page['page_number']}: scan page {page['scan_index'] + 1}, {page['dpi']:.0f} dpi, "
              f"turned {page['orientation']}°, marker fit {page['fit_error_mm']:.2f} mm")
    combined = result["printer_response"]["combined"]
    print(f"Printer: paper {combined['paper']}, black {combined['black']}, grays "
          + " ".join(f"{p['level']:.1f}->{p['measured']:.2f}" for p in combined["gray_patches"]))
    for summary in result["variations"]:
        print(f"  {summary['item_id'][:20]:20} {summary['variation_id']:6} {summary['role']:9} "
              f"corr {summary['correlation']}  white {summary['paper_white_share']}  "
              f"black {summary['full_black_share']}  collapsed {summary['collapsed_levels']}")
    print("Result:", result["result_path"])


if __name__ == "__main__":
    main()
