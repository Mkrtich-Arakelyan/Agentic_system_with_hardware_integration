"""Draw the printable trial pages for one tuning round and a manifest of where everything is.

Each page holds config page.items_per_page items, each shown in page.variations_per_item
variations (tiles). Next to the PDF goes a manifest (the trial record) with the position of
every region in mm from the top-left corner of the paper, so a scan can be measured later.
Every box is [x, y, width, height] in mm.
"""
import copy
import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from grayscale_render import (MAPPING_PARAMETERS, MappingError, _is_number, apply_mapping, changed_parameters,
                              compute_limits, load_config, mapping_label, scale_bar, validate_config,
                              validate_mapping)

SCHEMA_VERSION = "trial/1"
PAGE_W, PAGE_H = 210, 297
MM_PER_INCH = 25.4
PT_TO_MM = MM_PER_INCH / 72
DPI = 300
ENLARGE = 8                       # each data cell becomes an ENLARGE x ENLARGE block of pixels

MARKER = 8.0
CODE_BITS, CODE_SQUARE, CODE_GAP = 6, 3.0, 1.0
PAPER_REF_W, PAPER_REF_H = 15.0, 12.0
PATCH = 12.0
TICK_GAP, TICK_LEN = 0.4, 1.5     # paper area corner ticks, outside the area
ITEM_GAP, ITEM_HEADER_H = 4.0, 4.5
TILE_GAP, TILE_PAD = 2.5, 1.2
PANEL_GAP, PANEL_NAME_H = 2.0, 2.6   # gap between panels, line for the panel name above them
BAR_GAP, BAR_H, BAR_TEXT_H, LABEL_H, BOTTOM_PAD = 1.0, 1.6, 2.4, 3.0, 0.8
MIN_PANEL = 5.0
THIN, THICK = 0.3, 1.2            # outline widths in pt

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_PRINT_SETTINGS = {"printer": None, "monochrome": True, "scaling": "none", "paper": "A4",
                          "tray": None, "copies": 1}
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")

Box = tuple[float, float, float, float]


class TrialSheetError(ValueError):
    """The selection, variations or data cannot be drawn as a trial sheet for this round."""


def _box(x: float, y: float, w: float, h: float) -> list[float]:
    return [round(float(v), 2) for v in (x, y, w, h)]


def page_code_bits(page_number: int, round_id: int) -> list[int]:
    """The 6 page code bits: page number in the first 3 (it wraps after 7), round_id in the last 3."""
    return [int(b) for b in f"{page_number % 8:03b}{round_id:03b}"]


# ---------------------------------------------------------------- checks

def _round_parameter(round_id: int, config: dict) -> str:
    rounds = {entry["round_id"]: entry["parameter"] for entry in config["rounds"]}
    if isinstance(round_id, bool) or round_id not in rounds:
        raise TrialSheetError(f"round_id {round_id!r} is not in the config rounds {list(rounds)}")
    if round_id >= 2 ** (CODE_BITS // 2):
        raise TrialSheetError(f"round_id {round_id} does not fit in the 3 page code bits")
    return rounds[round_id]


def _check_selection(selection: dict, config: dict) -> None:
    """Structure only: ids, panel counts, names and kinds. The data is checked when loaded."""
    if not isinstance(selection, dict):
        raise TrialSheetError("selection must be a dict")
    missing = [key for key in ("run_id", "selection_id", "selection_version", "items") if key not in selection]
    if missing:
        raise TrialSheetError(f"selection is missing {missing}")
    items = selection["items"]
    if not isinstance(items, list) or not items:
        raise TrialSheetError("selection items must be a non-empty list")

    low, high = config["panels"]["min_per_item"], config["panels"]["max_per_item"]
    kinds = config["panels"]["allowed_kinds"]
    seen = set()
    for item in items:
        item_id = item.get("item_id") if isinstance(item, dict) else None
        if not isinstance(item_id, str) or not ID_PATTERN.match(item_id):
            raise TrialSheetError(f"item_id {item_id!r} must use only letters, digits, _ and -")
        if item_id in seen:
            raise TrialSheetError(f"item_id {item_id!r} appears more than once")
        seen.add(item_id)
        panels = item.get("panels")
        if not isinstance(panels, list) or not low <= len(panels) <= high:
            raise TrialSheetError(f"item {item_id}: needs {low} to {high} panels")
        names = [panel.get("name") if isinstance(panel, dict) else None for panel in panels]
        if any(not isinstance(name, str) or not name or "/" in name for name in names) \
                or len(set(names)) != len(names):
            raise TrialSheetError(f"item {item_id}: panel names must be unique non-empty strings "
                                  f"without '/', got {names}")
        for panel in panels:
            if panel.get("kind") not in kinds:
                raise TrialSheetError(f"item {item_id} panel {panel['name']}: kind {panel.get('kind')!r} "
                                      f"is not in {kinds}")
            if "data_path" not in panel:
                raise TrialSheetError(f"item {item_id} panel {panel['name']}: data_path is missing")


def _check_item_variations(item_id: str, item_variations, parameter: str, config: dict) -> None:
    """One reference, tests that each change only the round's parameter, no two tests the same."""
    wanted = config["page"]["variations_per_item"]
    if not isinstance(item_variations, list) or len(item_variations) != wanted:
        count = len(item_variations) if isinstance(item_variations, list) else 0
        raise TrialSheetError(f"item {item_id}: needs {wanted} variations, got {count}")

    seen = set()
    for variation in item_variations:
        variation_id = variation.get("variation_id") if isinstance(variation, dict) else None
        if not isinstance(variation_id, str) or not ID_PATTERN.match(variation_id) or variation_id == "data":
            raise TrialSheetError(f"item {item_id}: variation_id {variation_id!r} must use only letters, "
                                  "digits, _ and - and must not be 'data'")
        if variation_id in seen:
            raise TrialSheetError(f"item {item_id}: variation_id {variation_id!r} appears more than once")
        seen.add(variation_id)
        where = f"item {item_id} variation {variation_id}"
        if variation.get("role") not in ("reference", "test"):
            raise TrialSheetError(f"{where}: role must be 'reference' or 'test', got {variation.get('role')!r}")
        try:
            validate_mapping(variation.get("mapping"), config)
        except MappingError as error:
            raise TrialSheetError(f"{where}: {error}") from error

    rules = config["variation_rules"]
    references = [v["mapping"] for v in item_variations if v["role"] == "reference"]
    if len(references) != rules["reference_count"]:
        raise TrialSheetError(f"item {item_id}: needs exactly {rules['reference_count']} reference "
                              f"variation(s), got {len(references)}")
    reference = references[0]
    if any(mapping != reference for mapping in references):
        raise TrialSheetError(f"item {item_id}: the reference variations have different mappings")

    tested = [reference]
    for variation in (v for v in item_variations if v["role"] == "test"):
        changed = changed_parameters(reference, variation["mapping"])
        where = f"item {item_id} variation {variation['variation_id']}"
        if not changed:
            raise TrialSheetError(f"{where}: the test mapping is the same as the reference")
        if rules["tests_change_only_round_parameter"] and changed != {parameter}:
            raise TrialSheetError(f"{where}: round tests {parameter!r} but the mapping changes {sorted(changed)}")
        if variation["mapping"] in tested:
            raise TrialSheetError(f"{where}: the same mapping is already tested by another variation")
        tested.append(variation["mapping"])


def _check_variations(items: list, variations: dict, parameter: str, config: dict) -> None:
    if not isinstance(variations, dict):
        raise TrialSheetError("variations must be a dict {item_id: [variation, ...]}")
    item_ids = [item["item_id"] for item in items]
    extra = set(variations) - set(item_ids)
    if extra:
        raise TrialSheetError(f"variations given for unknown items {sorted(extra)}")
    for item_id in item_ids:
        _check_item_variations(item_id, variations.get(item_id), parameter, config)

    # Shared parameters (gray_levels: the printer's and paper's endpoints) must be the same in every
    # round: the k-th variation of every item uses the same value, so items stay comparable.
    if not config["variation_rules"]["shared_parameters_same_tests_for_all_items"]:
        return
    keys = sorted(key for key, name in MAPPING_PARAMETERS.items() if config["parameters"][name]["scope"] == "shared")
    first = item_ids[0]
    expected = [(v["role"], [v["mapping"][key] for key in keys]) for v in variations[first]]
    for item_id in item_ids[1:]:
        for k, variation in enumerate(variations[item_id]):
            role, values = variation["role"], [variation["mapping"][key] for key in keys]
            if role != expected[k][0]:
                raise TrialSheetError(f"variation {k + 1} is a {role} in item {item_id} but a {expected[k][0]} "
                                      f"in item {first}; variations must correspond across items")
            if values != expected[k][1]:
                raise TrialSheetError(f"{keys} are shared by all items, but variation {k + 1} of item {item_id} "
                                      f"uses {values} and variation {k + 1} of item {first} uses {expected[k][1]}")
    references = {item_id: next(v["mapping"] for v in variations[item_id] if v["role"] == "reference")
                  for item_id in item_ids}
    reference_values = {item_id: [mapping[key] for key in keys] for item_id, mapping in references.items()}
    if len({json.dumps(values) for values in reference_values.values()}) > 1:
        raise TrialSheetError(f"{keys} are shared, but the references differ between items: {reference_values}")


def load_panel_data(path, where: str = "panel") -> np.ndarray:
    """Read a panel .npy as float64; raise TrialSheetError unless it is a finite, non-empty 2D array."""
    path = Path(str(path))
    if not path.is_file():
        raise TrialSheetError(f"{where}: data file {path} does not exist")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as error:
        raise TrialSheetError(f"{where}: cannot read {path}: {error}") from error
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.size == 0:
        raise TrialSheetError(f"{where}: {path} must hold one non-empty 2D array")
    if not (np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)):
        raise TrialSheetError(f"{where}: {path} has non-numeric dtype {data.dtype}")
    data = data.astype(np.float64)
    if not np.all(np.isfinite(data)):
        raise TrialSheetError(f"{where}: {path} has NaN or infinite values")
    return data


def _finite(value) -> bool:
    """A real, finite number (not a bool, NaN or infinity)."""
    try:
        return _is_number(value) and math.isfinite(value)
    except OverflowError:                     # an int too large for a float
        return False


def _check_marks(marks, where: str, shape: tuple[int, int]) -> list[dict]:
    if marks is None:
        return []
    if not isinstance(marks, list):
        raise TrialSheetError(f"{where}: marks must be a list")
    for mark in marks:
        if not isinstance(mark, dict) or mark.get("type") != "box" or \
                not all(_finite(mark.get(key)) for key in ("row", "col", "size")) or mark["size"] <= 0:
            raise TrialSheetError(f"{where}: a mark must be a box with finite row, col and size > 0, got {mark!r}")
        if not (0 <= mark["row"] < shape[0] and 0 <= mark["col"] < shape[1]):
            raise TrialSheetError(f"{where}: mark centre ({mark['row']}, {mark['col']}) is outside the "
                                  f"{shape} data")
    return [dict(mark) for mark in marks]


def _check_limits(limits, item_id: str) -> dict:
    lo, hi = (limits.get("lo"), limits.get("hi")) if isinstance(limits, dict) else (None, None)
    if not all(_is_number(v) and math.isfinite(v) for v in (lo, hi)) or hi <= lo:
        raise TrialSheetError(f"item {item_id}: limits need finite lo < hi, got {lo!r}, {hi!r}")
    return copy.deepcopy(limits)


def _mark_cells(data: np.ndarray, mark: dict) -> np.ndarray:
    """The data cells inside a box mark."""
    half = mark["size"] / 2
    r0, r1 = max(0, math.floor(mark["row"] + 0.5 - half)), math.ceil(mark["row"] + 0.5 + half)
    c0, c1 = max(0, math.floor(mark["col"] + 0.5 - half)), math.ceil(mark["col"] + 0.5 + half)
    return data[r0:r1, c0:c1]


def _new_limits(arrays: list[np.ndarray], panels: list[dict], config: dict) -> dict:
    """The config's percentile limits, with hi raised to the brightest cell inside any box mark.

    A focus spot covers only a few cells, so the high percentile alone would clip it and print an
    Initial peak of 5 and a Final peak of 10 as the same white. Raising hi keeps that difference.
    """
    limits = compute_limits(arrays, config)
    peaks = [float(cells.max()) for data, panel in zip(arrays, panels) for mark in panel["marks"]
             if (cells := _mark_cells(data, mark)).size]
    if peaks and max(peaks) > limits["hi"]:
        limits["hi"] = max(peaks)
        limits["hi_raised_to_focus_peak"] = True
    return limits


def _load_items(selection_items: list, variations: dict, limits: dict, config: dict) -> tuple[list, dict]:
    """Load and check every panel; return the record items and the cells for the npz."""
    items, cells = [], {}
    for item in selection_items:
        item_id = item["item_id"]
        panels, arrays = [], []
        for panel in item["panels"]:
            where = f"item {item_id} panel {panel['name']}"
            data = load_panel_data(panel["data_path"], where)
            marks = _check_marks(item.get("marks"), where, data.shape) + \
                _check_marks(panel.get("marks"), where, data.shape)
            arrays.append(data)
            cells[f"{item_id}/data/{panel['name']}"] = data
            panels.append({"name": panel["name"], "kind": panel["kind"], "data_path": str(panel["data_path"]),
                           "sha256": hashlib.sha256(Path(panel["data_path"]).read_bytes()).hexdigest(),
                           "grid_shape": list(data.shape), "marks": marks})

        item_limits = _check_limits(limits[item_id], item_id) if item_id in limits \
            else _new_limits(arrays, panels, config)
        for panel, data in zip(panels, arrays):
            # cells outside the limits print at the end grays; the sheet and the manifest say how many
            panel["clipped"] = {"below_lo": int((data < item_limits["lo"]).sum()),
                                "above_hi": int((data > item_limits["hi"]).sum())}
            for variation in variations[item_id]:
                cells[f"{item_id}/{variation['variation_id']}/{panel['name']}"] = \
                    apply_mapping(data, item_limits, variation["mapping"])
        items.append({"item_id": item_id, "label": str(item.get("label", item_id)), "limits": item_limits,
                      "panels": panels})
    return items, cells


# ---------------------------------------------------------------- layout

def _panel_size(layout: dict, count: int) -> float:
    """Side of the square panels when count panels sit side by side in one tile."""
    return min((layout["tile_w"] - 2 * TILE_PAD - (count - 1) * PANEL_GAP) / count,
               layout["tile_h"] - TILE_PAD - PANEL_NAME_H - layout["below"])


def _layout(config: dict, panel_counts: list[int]) -> dict:
    """Positions in mm that are the same on every page."""
    margin = float(config["page"]["margin_mm"])
    width = PAGE_W - 2 * margin

    # reference strip: one row above the bottom markers
    refs = config["references"]
    n_paper, n_gray, black = refs["paper_refs"], refs["gray_patches"], int(bool(refs["black_patch"]))
    widths = [PAPER_REF_W] * n_paper + [PATCH] * (black + n_gray)
    gap = (width - sum(widths)) / max(len(widths) - 1, 1)
    if gap < 1.0:
        raise TrialSheetError("the reference strip does not fit across the page; use fewer patches")
    strip_y = PAGE_H - margin - MARKER - 1.5 - LABEL_H - PATCH
    xs = list(margin + np.concatenate([[0], np.cumsum(np.array(widths[:-1]) + gap)]))
    paper_x, black_x, gray_x = xs[:n_paper], xs[n_paper:n_paper + black], xs[n_paper + black:]

    # item blocks: between the header and the strip, each a grid of tiles
    top = margin + MARKER + 3.0
    bottom = strip_y - TICK_GAP - TICK_LEN - 1.0
    per_page = config["page"]["items_per_page"]
    block_h = (bottom - top - (per_page - 1) * ITEM_GAP) / per_page
    n = config["page"]["variations_per_item"]
    rows = max(1, math.isqrt(n))
    cols = math.ceil(n / rows)                      # e.g. 6 -> 2 rows of 3

    layout = {
        "margin": margin, "width": width, "strip_y": strip_y, "top": top, "block_h": block_h,
        "cols": cols,
        "tile_w": (width - (cols - 1) * TILE_GAP) / cols,
        "tile_h": (block_h - ITEM_HEADER_H - (rows - 1) * TILE_GAP) / rows,
        "below": BAR_GAP + BAR_H + BAR_TEXT_H + LABEL_H + BOTTOM_PAD,
        "paper_refs": [(x, strip_y, PAPER_REF_W, PAPER_REF_H) for x in paper_x],
        "black_patch": (black_x[0], strip_y, PATCH, PATCH) if black else None,
        "gray_patches": [(float(level), (x, strip_y, PATCH, PATCH))
                         for level, x in zip(np.linspace(0.0, 1.0, n_gray), gray_x)],
    }
    for count in set(panel_counts):
        if _panel_size(layout, count) < MIN_PANEL:
            raise TrialSheetError(f"{count} panels per item do not fit in a tile of "
                                  f"{layout['tile_w']:.1f} x {layout['tile_h']:.1f} mm")
    return layout


def _tile_geometry(layout: dict, tile_x: float, tile_y: float, count: int) -> tuple[list[Box], Box]:
    """Panel boxes and scale bar box for a tile with count panels, centred in the tile.

    Above the panels is a PANEL_NAME_H line for their names, below them the scale bar and label.
    """
    size = _panel_size(layout, count)
    row_w = count * size + (count - 1) * PANEL_GAP
    x0 = tile_x + (layout["tile_w"] - row_w) / 2
    y0 = tile_y + (layout["tile_h"] - PANEL_NAME_H - size - layout["below"] + TILE_PAD) / 2 + PANEL_NAME_H
    panels = [(x0 + i * (size + PANEL_GAP), y0, size, size) for i in range(count)]
    return panels, (x0, y0 + size + BAR_GAP, row_w, BAR_H)


# ---------------------------------------------------------------- drawing

class _Page:
    """One A4 figure with a full-page axes in mm (y pointing down) and the regions drawn on it."""

    def __init__(self):
        self.fig = Figure(figsize=(PAGE_W / MM_PER_INCH, PAGE_H / MM_PER_INCH), dpi=DPI)
        self.renderer = FigureCanvasAgg(self.fig).get_renderer()
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, PAGE_W)
        self.ax.set_ylim(PAGE_H, 0)
        self.ax.set_axis_off()
        self.regions = {key: [] for key in ("markers", "page_code", "tiles", "plots", "scale_bars",
                                            "paper_refs", "gray_patches", "labels")}
        self.regions["black_patch"] = None

    def rect(self, box: Box, fill=None, edge=None, width: float = THIN) -> None:
        x, y, w, h = box
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill or "none", edgecolor=edge or "none",
                                    linewidth=width if edge else 0))

    def line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.ax.plot([x0, x1], [y0, y1], color="black", linewidth=THIN, solid_capstyle="butt")

    def label(self, kind: str, x: float, y: float, text: str, max_w: float, size: float = 7,
              ids: dict | None = None, **kwargs) -> None:
        """Draw text that fits in max_w mm (shrinking to 6 pt, then shortening it) and record its box."""
        artist = self.ax.text(x, y, text, fontsize=size, **({"ha": "left", "va": "top"} | kwargs))
        while True:
            extent = artist.get_window_extent(self.renderer)
            scale = MM_PER_INCH / self.fig.dpi
            box = _box(extent.x0 * scale, PAGE_H - extent.y1 * scale, extent.width * scale,
                       extent.height * scale)
            if box[2] <= max_w or len(artist.get_text()) <= 2:
                break
            if artist.get_fontsize() > 6:
                artist.set_fontsize(max(6, artist.get_fontsize() - 0.5))
            else:
                artist.set_text(artist.get_text()[:-2] + "…")
        self.regions["labels"].append({"kind": kind, "text": artist.get_text(), "box": box} | (ids or {}))

    def image(self, box: Box, grays: np.ndarray):
        """Draw grays (0 black, 1 white) in box, each cell enlarged so nothing gets smoothed."""
        x, y, w, h = box
        ax = self.fig.add_axes([x / PAGE_W, 1 - (y + h) / PAGE_H, w / PAGE_W, h / PAGE_H])
        ax.set_axis_off()
        rows, cols = grays.shape
        ax.imshow(np.kron(grays, np.ones((ENLARGE, ENLARGE))), cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="nearest", aspect="auto", extent=(0, cols, rows, 0))
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        return ax


def _draw_box_mark(ax, mark: dict, shape: tuple[int, int], box_mm: Box) -> None:
    """Thin double outline: black outside, white just inside it, in data cell units."""
    rows, cols = shape
    black_w, white_w = 0.6, 0.4
    inset = (black_w + white_w) / 2 * PT_TO_MM
    dx, dy = inset * cols / box_mm[2], inset * rows / box_mm[3]
    x0, y0, size = mark["col"] + 0.5 - mark["size"] / 2, mark["row"] + 0.5 - mark["size"] / 2, mark["size"]
    ax.add_patch(Rectangle((x0, y0), size, size, facecolor="none", edgecolor="black", linewidth=black_w))
    ax.add_patch(Rectangle((x0 + dx, y0 + dy), size - 2 * dx, size - 2 * dy, facecolor="none",
                           edgecolor="white", linewidth=white_w))


def _draw_frame(page: _Page, layout: dict, bits: list[int], header: str) -> None:
    """Corner markers, page code squares and the header line."""
    margin = layout["margin"]
    far_x, far_y = PAGE_W - margin - MARKER, PAGE_H - margin - MARKER
    for corner, (x, y) in zip(CORNERS, ((margin, margin), (far_x, margin), (margin, far_y), (far_x, far_y))):
        page.rect((x, y, MARKER, MARKER), fill="black")
        page.regions["markers"].append({"corner": corner, "box": _box(x, y, MARKER, MARKER)})

    code_x, code_y = margin + MARKER + 2.0, margin + (MARKER - CODE_SQUARE) / 2
    for i, bit in enumerate(bits):
        box = (code_x + i * (CODE_SQUARE + CODE_GAP), code_y, CODE_SQUARE, CODE_SQUARE)
        page.rect(box, fill="black" if bit else "white", edge="black", width=0.4)
        page.regions["page_code"].append({"bit": bit, "box": _box(*box)})

    header_x = code_x + CODE_BITS * (CODE_SQUARE + CODE_GAP) + 2.0
    page.label("header", header_x, margin + MARKER / 2, header, far_x - 2.0 - header_x, size=8, va="center")


def _draw_tile(page: _Page, layout: dict, item: dict, variation: dict, tile: Box, cells: dict) -> None:
    """One variation: the item's panels side by side, a scale bar and the mapping label."""
    ids = {"item_id": item["item_id"], "variation_id": variation["variation_id"]}
    is_reference = variation["role"] == "reference"
    page.rect(tile, edge="black", width=THICK if is_reference else THIN)
    page.regions["tiles"].append(ids | {"role": variation["role"], "box": _box(*tile)})

    panel_boxes, bar = _tile_geometry(layout, tile[0], tile[1], len(item["panels"]))
    for panel, box in zip(item["panels"], panel_boxes):
        page.label("panel_name", box[0] + box[2] / 2, box[1] - PANEL_NAME_H + 0.2, panel["name"], box[2],
                   size=6, ids=ids | {"panel": panel["name"]}, ha="center")
        grays = cells[f"{item['item_id']}/{variation['variation_id']}/{panel['name']}"]
        ax = page.image(box, grays)
        for mark in panel["marks"]:
            _draw_box_mark(ax, mark, grays.shape, box)
        page.regions["plots"].append(ids | {"panel": panel["name"], "box": _box(*box),
                                            "grid_shape": list(grays.shape)})

    limits = item["limits"]
    bar_data = scale_bar(limits, variation["mapping"])
    page.image(bar, bar_data["grays"][np.newaxis, :])
    page.rect(bar, edge="black")
    page.regions["scale_bars"].append(ids | {"box": _box(*bar), "lo": limits["lo"], "hi": limits["hi"]})
    text_y = bar[1] + BAR_H + 0.3
    tick_x = (bar[0], bar[0] + bar[2] / 2, bar[0] + bar[2])
    for tick, x, align in zip(bar_data["ticks"], tick_x, ("left", "center", "right")):
        page.label("scale_tick", x, text_y, tick["label"], bar[2] / 3, size=6, ids=ids, ha=align)

    text = ("best so far · " if is_reference else "") + mapping_label(variation["mapping"])
    page.label("tile", tile[0] + TILE_PAD, text_y + BAR_TEXT_H, text, layout["tile_w"] - 2 * TILE_PAD, ids=ids)


def _draw_item(page: _Page, layout: dict, slot: int, item: dict, item_variations: list, cells: dict) -> None:
    """The item header line and one tile per variation, in a grid."""
    margin = layout["margin"]
    block_y = layout["top"] + slot * (layout["block_h"] + ITEM_GAP)
    lo, hi = item["limits"]["lo"], item["limits"]["hi"]
    above = sum(panel["clipped"]["above_hi"] for panel in item["panels"])
    clipped = f" · {above} cells above hi are clipped" if above else ""
    page.label("item_header", margin, block_y,
               f"{item['label']} · {item['item_id']} · limits {lo:.4g} to {hi:.4g}{clipped}",
               layout["width"], ids={"item_id": item["item_id"]})
    for index, variation in enumerate(item_variations):
        row, col = divmod(index, layout["cols"])
        tile = (margin + col * (layout["tile_w"] + TILE_GAP),
                block_y + ITEM_HEADER_H + row * (layout["tile_h"] + TILE_GAP), layout["tile_w"], layout["tile_h"])
        _draw_tile(page, layout, item, variation, tile, cells)


def _draw_reference_strip(page: _Page, layout: dict) -> None:
    """Blank paper areas with corner ticks, the black patch and the gray patches with their levels."""
    label_y = layout["strip_y"] + PATCH + TICK_GAP + 0.6
    for box in layout["paper_refs"]:
        x, y, w, h = box
        for cx, cy, sign in ((x, y, -1), (x + w, y, -1), (x, y + h, 1), (x + w, y + h, 1)):
            page.line(cx, cy + sign * TICK_GAP, cx, cy + sign * (TICK_GAP + TICK_LEN))
        page.regions["paper_refs"].append({"box": _box(*box)})
        page.label("paper_ref", x + w / 2, label_y, "paper", w, size=6, ha="center")
    if layout["black_patch"]:
        box = layout["black_patch"]
        page.rect(box, fill="black")
        page.regions["black_patch"] = {"box": _box(*box)}
        page.label("black_patch", box[0] + PATCH / 2, label_y, "black", PATCH, size=6, ha="center")
    for level, box in layout["gray_patches"]:
        page.rect(box, fill=(level, level, level))
        page.regions["gray_patches"].append({"level": round(level, 4), "box": _box(*box)})
        page.label("patch_level", box[0] + PATCH / 2, label_y, f"{level:.2f}", PATCH, size=6, ha="center")


def _write_pdf(path: Path, items: list, variations: dict, cells: dict, layout: dict, per_page: int,
               round_id: int, parameter: str, trial_id: str) -> list[dict]:
    """Draw every page into one PDF; return the manifest pages."""
    chunks = [items[i:i + per_page] for i in range(0, len(items), per_page)]
    pages = []
    with rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42}), \
            PdfPages(path) as pdf:
        for number, chunk in enumerate(chunks, start=1):
            page = _Page()
            bits = page_code_bits(number, round_id)
            _draw_frame(page, layout, bits, f"Trial round {round_id} · testing {parameter} · "
                                            f"page {number} of {len(chunks)} · trial {trial_id[:8]}")
            for slot, item in enumerate(chunk):
                _draw_item(page, layout, slot, item, variations[item["item_id"]], cells)
            _draw_reference_strip(page, layout)
            pdf.savefig(page.fig)
            pages.append({"page_number": number, "page_code": bits, "regions": page.regions})
    return pages


# ---------------------------------------------------------------- main

def _check_same_selection(previous: dict, selection: dict, items: list, calibration_session_id: str) -> None:
    """A later round must print the same experiments and data, in the same order, in the same session.

    items are the loaded items of this round. The data is compared by its sha256, not its path,
    so the same file at a new path is fine and changed content at the same path is refused.
    """
    for key, value in (("selection_id", selection["selection_id"]), ("run_id", selection["run_id"]),
                       ("selection_version", selection["selection_version"]),
                       ("calibration_session_id", calibration_session_id)):
        if previous.get(key) != value:
            raise TrialSheetError(f"{key} {value!r} differs from the previous round's {previous.get(key)!r}")
    before = [item["item_id"] for item in previous["items"]]
    now = [item["item_id"] for item in items]
    if before != now:
        raise TrialSheetError(f"items {now} differ from the previous round's {before} (same items, same order)")
    if any("sha256" not in panel for item in previous["items"] for panel in item["panels"]):
        raise TrialSheetError("the previous record has no data hashes (an older manifest); "
                              "start a fresh calibration session")

    for old, new in zip(previous["items"], json.loads(json.dumps(items))):
        layout = [[{key: panel[key] for key in ("name", "kind", "grid_shape", "marks")} for panel in item["panels"]]
                  for item in (old, new)]
        if layout[0] != layout[1]:
            raise TrialSheetError(f"item {new['item_id']}: the panels (names, kinds, order, shapes or marks) "
                                  "differ from the previous round")
        if [panel["sha256"] for panel in old["panels"]] != [panel["sha256"] for panel in new["panels"]]:
            raise TrialSheetError(f"item {new['item_id']}: the panel data (sha256) changed since the previous round")


def create_trial_sheet(selection: dict, variations: dict, round_id: int, out_dir, calibration_session_id: str,
                       config: dict | None = None, limits: dict | None = None,
                       print_settings: dict | None = None, previous_record: dict | None = None) -> dict:
    """Check the inputs, draw the trial PDF and write the cells and manifest; return the trial record.

    previous_record is the record of an earlier round of the same calibration session: the
    selection must then be the same (id, items and order) and its limits are kept.
    Files are named trial_r{round_id}_{first 8 characters of trial_id} and never overwrite
    existing files, so earlier sheets, manifests and cells stay available.
    """
    if config is None:
        config = load_config()
    else:
        validate_config(config)
    parameter = _round_parameter(round_id, config)
    _check_selection(selection, config)
    if previous_record is not None:
        previous_limits = {item["item_id"]: item["limits"] for item in previous_record["items"]}
        conflicting = [item_id for item_id, value in (limits or {}).items() if previous_limits.get(item_id) != value]
        if conflicting:
            raise TrialSheetError(f"limits for {conflicting} differ from the previous round's; "
                                  "a calibration session keeps its limits")
        limits = previous_limits
    _check_variations(selection["items"], variations, parameter, config)
    items, cells = _load_items(selection["items"], variations, limits or {}, config)
    if previous_record is not None:
        _check_same_selection(previous_record, selection, items, calibration_session_id)
    layout = _layout(config, [len(item["panels"]) for item in items])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:            # a new trial_id until no file of that name exists yet
        trial_id = str(uuid.uuid4())
        stem = out_dir / f"trial_r{round_id}_{trial_id[:8]}"
        paths = [Path(f"{stem}{end}") for end in (".pdf", "_cells.npz", "_manifest.json")]
        if not any(path.exists() for path in paths):
            break
    pdf_path, cells_path, manifest_path = paths

    pages = _write_pdf(pdf_path, items, variations, cells, layout, config["page"]["items_per_page"],
                       round_id, parameter, trial_id)
    np.savez_compressed(cells_path, **cells)

    record = {
        "schema_version": SCHEMA_VERSION,
        "trial_id": trial_id,
        "run_id": selection["run_id"],
        "selection_id": selection["selection_id"],
        "selection_version": selection["selection_version"],
        "calibration_session_id": calibration_session_id,
        "round_id": round_id,
        "tested_parameter": parameter,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": copy.deepcopy(config),
        "items": items,
        "variations": {item["item_id"]: copy.deepcopy(variations[item["item_id"]]) for item in items},
        "pdf": {"path": str(pdf_path), "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                "pages": len(pages), "page_size_mm": [PAGE_W, PAGE_H]},
        "print_settings": DEFAULT_PRINT_SETTINGS | (print_settings or {}),
        "cells_path": str(cells_path),
        "manifest_path": str(manifest_path),
        "pages": pages,
    }
    manifest_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return record
