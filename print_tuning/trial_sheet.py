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

from print_tuning.grayscale_render import (MAPPING_PARAMETERS, MappingError, apply_mapping,
                                           changed_parameters, compute_limits, load_config,
                                           mapping_label, scale_bar, validate_config, validate_mapping)

SCHEMA_VERSION = "trial/1"
PAGE_W, PAGE_H = 210.0, 297.0
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
PANEL_GAP = 2.0
BAR_GAP, BAR_H, BAR_TEXT_H, LABEL_H, BOTTOM_PAD = 1.0, 1.6, 2.4, 3.0, 0.8
MIN_PANEL = 5.0
THIN, THICK = 0.3, 1.2            # outline widths in pt

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_PRINT_SETTINGS = {"printer": None, "monochrome": True, "scaling": "none", "paper": "A4",
                          "tray": None, "copies": 1}
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


class TrialSheetError(ValueError):
    """The selection, variations or data cannot be drawn as a trial sheet for this round."""


def _box(x, y, w, h):
    return [round(float(v), 2) for v in (x, y, w, h)]


def _grid(n):
    """Rows and columns for n tiles, e.g. 6 -> 2 rows of 3."""
    rows = max(1, math.isqrt(n))
    return rows, math.ceil(n / rows)


def page_code_bits(page_number, round_id):
    """The 6 page code bits: page number in the first 3 (it wraps after 7), round_id in the last 3."""
    return [int(b) for b in f"{page_number % 8:03b}{round_id:03b}"]


# ---------------------------------------------------------------- checks

def _round_parameter(round_id, config):
    for entry in config["rounds"]:
        if entry["round_id"] == round_id and not isinstance(round_id, bool):
            if round_id >= 2 ** (CODE_BITS // 2):
                raise TrialSheetError(f"round_id {round_id} does not fit in the 3 page code bits")
            return entry["parameter"]
    known = [entry["round_id"] for entry in config["rounds"]]
    raise TrialSheetError(f"round_id {round_id!r} is not in the config rounds {known}")


def _check_marks(marks, where):
    if marks is None:
        return []
    if not isinstance(marks, list):
        raise TrialSheetError(f"{where}: marks must be a list")
    for mark in marks:
        if not isinstance(mark, dict) or mark.get("type") != "box":
            raise TrialSheetError(f"{where}: each mark must be a dict with type 'box', got {mark!r}")
        if not all(isinstance(mark.get(key), (int, float)) and not isinstance(mark.get(key), bool)
                   for key in ("row", "col", "size")) or mark["size"] <= 0:
            raise TrialSheetError(f"{where}: a box mark needs numeric row, col and a size > 0, got {mark!r}")
    return [dict(mark) for mark in marks]


def _load_panel(path, where):
    path = Path(str(path))
    if not path.is_file():
        raise TrialSheetError(f"{where}: data file {path} does not exist")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as error:
        raise TrialSheetError(f"{where}: cannot read {path}: {error}") from error
    if not isinstance(data, np.ndarray):
        raise TrialSheetError(f"{where}: {path} must hold a single array (.npy)")
    if data.ndim != 2 or data.size == 0:
        raise TrialSheetError(f"{where}: {path} must be a non-empty 2D array, got shape {data.shape}")
    if not (np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)):
        raise TrialSheetError(f"{where}: {path} has non-numeric dtype {data.dtype}")
    data = data.astype(np.float64)
    if not np.all(np.isfinite(data)):
        raise TrialSheetError(f"{where}: {path} has NaN or infinite values")
    return data


def _check_selection(selection, config):
    if not isinstance(selection, dict):
        raise TrialSheetError("selection must be a dict")
    for key in ("run_id", "selection_id", "selection_version", "items"):
        if key not in selection:
            raise TrialSheetError(f"selection is missing {key!r}")
    items = selection["items"]
    if not isinstance(items, list) or not items:
        raise TrialSheetError("selection items must be a non-empty list")
    panels_cfg = config["panels"]
    seen = set()
    for item in items:
        item_id = item.get("item_id") if isinstance(item, dict) else None
        if not isinstance(item_id, str) or not ID_PATTERN.match(item_id):
            raise TrialSheetError(f"item_id {item_id!r} must use only letters, digits, _ and -")
        if item_id in seen:
            raise TrialSheetError(f"item_id {item_id!r} appears more than once")
        seen.add(item_id)
        _check_marks(item.get("marks"), f"item {item_id}")
        panels = item.get("panels")
        if not isinstance(panels, list) or \
                not panels_cfg["min_per_item"] <= len(panels) <= panels_cfg["max_per_item"]:
            raise TrialSheetError(f"item {item_id}: needs {panels_cfg['min_per_item']} to "
                                  f"{panels_cfg['max_per_item']} panels, got "
                                  f"{len(panels) if isinstance(panels, list) else panels!r}")
        names = set()
        for panel in panels:
            name = panel.get("name") if isinstance(panel, dict) else None
            if not isinstance(name, str) or not name or "/" in name or name in names:
                raise TrialSheetError(f"item {item_id}: panel names must be unique non-empty strings "
                                      f"without '/', got {name!r}")
            names.add(name)
            if panel.get("kind") not in panels_cfg["allowed_kinds"]:
                raise TrialSheetError(f"item {item_id} panel {name}: kind {panel.get('kind')!r} is not in "
                                      f"{panels_cfg['allowed_kinds']}")
            if "data_path" not in panel:
                raise TrialSheetError(f"item {item_id} panel {name}: data_path is missing")
            _check_marks(panel.get("marks"), f"item {item_id} panel {name}")


def _check_variations(items, variations, parameter, config):
    if not isinstance(variations, dict):
        raise TrialSheetError("variations must be a dict {item_id: [variation, ...]}")
    item_ids = [item["item_id"] for item in items]
    extra = set(variations) - set(item_ids)
    if extra:
        raise TrialSheetError(f"variations given for unknown items {sorted(extra)}")
    wanted = config["page"]["variations_per_item"]
    rules = config["variation_rules"]
    shared = config["parameters"][parameter]["scope"] == "shared"
    keys = sorted(key for key, name in MAPPING_PARAMETERS.items() if name == parameter)
    shared_values = {}

    for item_id in item_ids:
        item_variations = variations.get(item_id)
        if not isinstance(item_variations, list) or len(item_variations) != wanted:
            count = len(item_variations) if isinstance(item_variations, list) else 0
            raise TrialSheetError(f"item {item_id}: needs {wanted} variations, got {count}")
        ids = set()
        for variation in item_variations:
            variation_id = variation.get("variation_id") if isinstance(variation, dict) else None
            if not isinstance(variation_id, str) or not ID_PATTERN.match(variation_id) or variation_id == "data":
                raise TrialSheetError(f"item {item_id}: variation_id {variation_id!r} must use only letters, "
                                      "digits, _ and - and must not be 'data'")
            if variation_id in ids:
                raise TrialSheetError(f"item {item_id}: variation_id {variation_id!r} appears more than once")
            ids.add(variation_id)
            if variation.get("role") not in ("reference", "test"):
                raise TrialSheetError(f"item {item_id} variation {variation_id}: role must be 'reference' "
                                      f"or 'test', got {variation.get('role')!r}")
            try:
                validate_mapping(variation.get("mapping"), config)
            except MappingError as error:
                raise TrialSheetError(f"item {item_id} variation {variation_id}: {error}") from error

        references = [v for v in item_variations if v["role"] == "reference"]
        if len(references) != rules["reference_count"]:
            raise TrialSheetError(f"item {item_id}: needs exactly {rules['reference_count']} reference "
                                  f"variation(s), got {len(references)}")
        reference = references[0]["mapping"]
        if any(v["mapping"] != reference for v in references[1:]):
            raise TrialSheetError(f"item {item_id}: the reference variations have different mappings")

        test_values = set()
        for variation in item_variations:
            if variation["role"] != "test":
                continue
            changed = changed_parameters(reference, variation["mapping"])
            where = f"item {item_id} variation {variation['variation_id']}"
            if not changed:
                raise TrialSheetError(f"{where}: the test mapping is the same as the reference")
            if rules["tests_change_only_round_parameter"] and changed != {parameter}:
                raise TrialSheetError(f"{where}: round tests {parameter!r} but the mapping changes "
                                      f"{sorted(changed)}")
            test_values.add(tuple(variation["mapping"][key] for key in keys))

        if shared and rules["shared_parameters_same_tests_for_all_items"]:
            values = (tuple(reference[key] for key in keys), test_values)
            if shared_values and values != next(iter(shared_values.values())):
                first = next(iter(shared_values))
                raise TrialSheetError(f"{parameter!r} is shared, but items {first} and {item_id} have a "
                                      "different reference value or set of test values")
            shared_values[item_id] = values


def _check_limits(limits, item_id):
    if not isinstance(limits, dict):
        raise TrialSheetError(f"item {item_id}: limits must be a dict")
    lo, hi = limits.get("lo"), limits.get("hi")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (lo, hi)) \
            or hi <= lo:
        raise TrialSheetError(f"item {item_id}: limits need finite lo < hi, got {lo!r}, {hi!r}")
    return copy.deepcopy(limits)


# ---------------------------------------------------------------- layout

def _layout(config, panel_counts):
    """Positions in mm that are the same on every page."""
    margin = float(config["page"]["margin_mm"])
    width = PAGE_W - 2 * margin
    layout = {"margin": margin, "width": width}

    # reference strip, one row above the bottom markers
    refs = config["references"]
    n_paper, n_gray = refs["paper_refs"], refs["gray_patches"]
    black = bool(refs["black_patch"])
    widths = [PAPER_REF_W] * n_paper + [PATCH] * (int(black) + n_gray)
    gap = (width - sum(widths)) / max(len(widths) - 1, 1)
    if gap < 1.0:
        raise TrialSheetError("the reference strip does not fit across the page; use fewer patches")
    patch_y = PAGE_H - margin - MARKER - 1.5 - LABEL_H - PATCH
    xs = margin + np.concatenate([[0], np.cumsum(np.array(widths[:-1]) + gap)])
    layout["paper_refs"] = [(x, patch_y, PAPER_REF_W, PAPER_REF_H) for x in xs[:n_paper]]
    layout["black_patch"] = (xs[n_paper], patch_y, PATCH, PATCH) if black else None
    gray_x = xs[n_paper + int(black):]
    layout["gray_patches"] = [(float(level), (x, patch_y, PATCH, PATCH))
                              for level, x in zip(np.linspace(0.0, 1.0, n_gray), gray_x)]
    layout["strip_y"] = patch_y

    # item blocks between header and strip
    top = margin + MARKER + 3.0
    bottom = patch_y - TICK_GAP - TICK_LEN - 1.0
    per_page = config["page"]["items_per_page"]
    block_h = (bottom - top - (per_page - 1) * ITEM_GAP) / per_page
    rows, cols = _grid(config["page"]["variations_per_item"])
    tile_w = (width - (cols - 1) * TILE_GAP) / cols
    tile_h = (block_h - ITEM_HEADER_H - (rows - 1) * TILE_GAP) / rows
    below = BAR_GAP + BAR_H + BAR_TEXT_H + LABEL_H + BOTTOM_PAD
    for count in set(panel_counts):
        size = min((tile_w - 2 * TILE_PAD - (count - 1) * PANEL_GAP) / count, tile_h - TILE_PAD - below)
        if size < MIN_PANEL:
            raise TrialSheetError(f"{count} panels per item do not fit in a tile of "
                                  f"{tile_w:.1f} x {tile_h:.1f} mm")
    layout.update(top=top, block_h=block_h, rows=rows, cols=cols, tile_w=tile_w, tile_h=tile_h, below=below)
    return layout


def _tile_geometry(layout, tile_x, tile_y, count):
    """Panel boxes and scale bar box for a tile with count panels."""
    size = min((layout["tile_w"] - 2 * TILE_PAD - (count - 1) * PANEL_GAP) / count,
               layout["tile_h"] - TILE_PAD - layout["below"])
    row_w = count * size + (count - 1) * PANEL_GAP
    x0 = tile_x + (layout["tile_w"] - row_w) / 2
    y0 = tile_y + (layout["tile_h"] - size - layout["below"] + TILE_PAD) / 2   # centred in the tile
    panels = [(x0 + i * (size + PANEL_GAP), y0, size, size) for i in range(count)]
    bar = (x0, y0 + size + BAR_GAP, row_w, BAR_H)
    return panels, bar


# ---------------------------------------------------------------- drawing

class _Page:
    """One A4 figure with a full-page axes whose data units are mm, y pointing down."""

    def __init__(self):
        self.fig = Figure(figsize=(PAGE_W / MM_PER_INCH, PAGE_H / MM_PER_INCH), dpi=DPI)
        self.canvas = FigureCanvasAgg(self.fig)
        self.renderer = self.canvas.get_renderer()
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, PAGE_W)
        self.ax.set_ylim(PAGE_H, 0)
        self.ax.set_axis_off()

    def rect(self, box, fill=None, edge=None, width=THIN):
        x, y, w, h = box
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill or "none", edgecolor=edge or "none",
                                    linewidth=width if edge else 0))

    def line(self, x0, y0, x1, y1, width=THIN):
        self.ax.plot([x0, x1], [y0, y1], color="black", linewidth=width, solid_capstyle="butt")

    def text(self, x, y, text, max_w, size=7, min_size=6, **kwargs):
        """Draw text that fits in max_w mm (shrinking, then shortening it) and return (text, box)."""
        kwargs = {"ha": "left", "va": "top", "color": "black"} | kwargs
        artist = self.ax.text(x, y, text, fontsize=size, **kwargs)
        while True:
            box = self._extent(artist)
            if box[2] <= max_w or len(artist.get_text()) <= 2:
                return artist.get_text(), box
            if artist.get_fontsize() > min_size:
                artist.set_fontsize(max(min_size, artist.get_fontsize() - 0.5))
            else:
                artist.set_text(artist.get_text()[:-2] + "…")

    def _extent(self, artist):
        extent = artist.get_window_extent(self.renderer)
        scale = MM_PER_INCH / self.fig.dpi
        return _box(extent.x0 * scale, PAGE_H - extent.y1 * scale, extent.width * scale, extent.height * scale)

    def image(self, box, grays, extent=None):
        x, y, w, h = box
        ax = self.fig.add_axes([x / PAGE_W, 1 - (y + h) / PAGE_H, w / PAGE_W, h / PAGE_H])
        ax.set_axis_off()
        rows, cols = grays.shape
        big = np.kron(grays, np.ones((ENLARGE, ENLARGE)))
        ax.imshow(big, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto",
                  extent=extent or (0, cols, rows, 0))
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        return ax


def _draw_box_mark(ax, mark, shape, box_mm):
    """Thin double outline: black outside, white just inside it, in data cell units."""
    rows, cols = shape
    per_mm_x, per_mm_y = cols / box_mm[2], rows / box_mm[3]
    black_w, white_w = 0.6, 0.4
    inset = (black_w + white_w) / 2 * PT_TO_MM
    half = mark["size"] / 2
    x0, y0 = mark["col"] + 0.5 - half, mark["row"] + 0.5 - half
    ax.add_patch(Rectangle((x0, y0), mark["size"], mark["size"], facecolor="none", edgecolor="black",
                           linewidth=black_w))
    dx, dy = inset * per_mm_x, inset * per_mm_y
    ax.add_patch(Rectangle((x0 + dx, y0 + dy), mark["size"] - 2 * dx, mark["size"] - 2 * dy,
                           facecolor="none", edgecolor="white", linewidth=white_w))


def _draw_page(page_number, n_pages, page_items, round_id, parameter, trial_id, layout, cells):
    page = _Page()
    margin, width = layout["margin"], layout["width"]
    regions = {key: [] for key in ("markers", "page_code", "tiles", "plots", "scale_bars", "paper_refs",
                                   "gray_patches", "labels")}
    regions["black_patch"] = None

    def label(kind, x, y, text, max_w, **kwargs):
        extra = {key: kwargs.pop(key) for key in ("item_id", "variation_id") if key in kwargs}
        text, box = page.text(x, y, text, max_w, **kwargs)
        regions["labels"].append({"kind": kind, "text": text, "box": box} | extra)

    # corner markers
    far_x, far_y = PAGE_W - margin - MARKER, PAGE_H - margin - MARKER
    for corner, (x, y) in zip(CORNERS, ((margin, margin), (far_x, margin), (margin, far_y), (far_x, far_y))):
        page.rect((x, y, MARKER, MARKER), fill="black")
        regions["markers"].append({"corner": corner, "box": _box(x, y, MARKER, MARKER)})

    # page code and header
    bits = page_code_bits(page_number, round_id)
    code_x, code_y = margin + MARKER + 2.0, margin + (MARKER - CODE_SQUARE) / 2
    for i, bit in enumerate(bits):
        box = (code_x + i * (CODE_SQUARE + CODE_GAP), code_y, CODE_SQUARE, CODE_SQUARE)
        page.rect(box, fill="black" if bit else "white", edge="black", width=0.4)
        regions["page_code"].append({"bit": bit, "box": _box(*box)})
    header_x = code_x + CODE_BITS * (CODE_SQUARE + CODE_GAP) + 2.0
    label("header", header_x, margin + MARKER / 2,
          f"Trial round {round_id} · testing {parameter} · page {page_number} of {n_pages} · "
          f"trial {trial_id[:8]}", far_x - 2.0 - header_x, size=8, va="center")

    # item blocks
    for slot, item in enumerate(page_items):
        block_y = layout["top"] + slot * (layout["block_h"] + ITEM_GAP)
        limits = item["limits"]
        label("item_header", margin, block_y,
              f"{item['label']} · {item['item_id']} · limits {limits['lo']:.4g} to {limits['hi']:.4g}",
              width, item_id=item["item_id"])
        for index, variation in enumerate(item["variations"]):
            row, col = divmod(index, layout["cols"])
            tile_x = margin + col * (layout["tile_w"] + TILE_GAP)
            tile_y = block_y + ITEM_HEADER_H + row * (layout["tile_h"] + TILE_GAP)
            tile = (tile_x, tile_y, layout["tile_w"], layout["tile_h"])
            is_reference = variation["role"] == "reference"
            page.rect(tile, edge="black", width=THICK if is_reference else THIN)
            ids = {"item_id": item["item_id"], "variation_id": variation["variation_id"]}
            regions["tiles"].append(ids | {"role": variation["role"], "box": _box(*tile)})

            panel_boxes, bar = _tile_geometry(layout, tile_x, tile_y, len(item["panels"]))
            for panel, box in zip(item["panels"], panel_boxes):
                grays = cells[f"{item['item_id']}/{variation['variation_id']}/{panel['name']}"]
                ax = page.image(box, grays)
                for mark in panel["marks"]:
                    _draw_box_mark(ax, mark, grays.shape, box)
                regions["plots"].append(ids | {"panel": panel["name"], "box": _box(*box),
                                               "grid_shape": list(grays.shape)})

            bar_data = scale_bar(limits, variation["mapping"])
            page.image(bar, bar_data["grays"][np.newaxis, :])
            page.rect(bar, edge="black", width=THIN)
            regions["scale_bars"].append(ids | {"box": _box(*bar), "lo": limits["lo"], "hi": limits["hi"]})
            text_y = bar[1] + BAR_H + 0.3
            for tick, x, align in zip(bar_data["ticks"], (bar[0], bar[0] + bar[2] / 2, bar[0] + bar[2]),
                                      ("left", "center", "right")):
                label("scale_tick", x, text_y, tick["label"], bar[2] / 3, size=6, ha=align, **ids)

            prefix = "best so far · " if is_reference else ""
            label("tile", tile_x + TILE_PAD, text_y + BAR_TEXT_H, prefix + mapping_label(variation["mapping"]),
                  layout["tile_w"] - 2 * TILE_PAD, **ids)

    # reference strip
    label_y = layout["strip_y"] + PATCH + TICK_GAP + 0.6
    for box in layout["paper_refs"]:
        x, y, w, h = box
        for cx, cy, sy in ((x, y, -1), (x + w, y, -1), (x, y + h, 1), (x + w, y + h, 1)):
            page.line(cx, cy + sy * TICK_GAP, cx, cy + sy * (TICK_GAP + TICK_LEN))
        regions["paper_refs"].append({"box": _box(*box)})
        label("paper_ref", x + w / 2, label_y, "paper", w, size=6, ha="center")
    if layout["black_patch"]:
        box = layout["black_patch"]
        page.rect(box, fill="black")
        regions["black_patch"] = {"box": _box(*box)}
        label("black_patch", box[0] + PATCH / 2, label_y, "black", PATCH, size=6, ha="center")
    for level, box in layout["gray_patches"]:
        page.rect(box, fill=(level, level, level))
        regions["gray_patches"].append({"level": round(level, 4), "box": _box(*box)})
        label("patch_level", box[0] + PATCH / 2, label_y, f"{level:.2f}", PATCH, size=6, ha="center")

    return page.fig, {"page_number": page_number, "page_code": bits, "regions": regions}


# ---------------------------------------------------------------- main

def create_trial_sheet(selection, variations, round_id, out_dir, calibration_session_id,
                       config=None, limits=None, print_settings=None):
    """Check the inputs, draw the trial PDF and write the cells and manifest; return the trial record."""
    if config is None:
        config = load_config()
    else:
        validate_config(config)
    parameter = _round_parameter(round_id, config)
    _check_selection(selection, config)
    items_in = selection["items"]
    _check_variations(items_in, variations, parameter, config)
    limits = limits or {}

    items, cells = [], {}
    for item in items_in:
        item_id = item["item_id"]
        item_marks = _check_marks(item.get("marks"), f"item {item_id}")
        panels, arrays = [], []
        for panel in item["panels"]:
            data = _load_panel(panel["data_path"], f"item {item_id} panel {panel['name']}")
            marks = item_marks + _check_marks(panel.get("marks"), f"item {item_id} panel {panel['name']}")
            for mark in marks:
                if not (0 <= mark["row"] < data.shape[0] and 0 <= mark["col"] < data.shape[1]):
                    raise TrialSheetError(f"item {item_id} panel {panel['name']}: mark centre "
                                          f"({mark['row']}, {mark['col']}) is outside the {data.shape} data")
            arrays.append(data)
            cells[f"{item_id}/data/{panel['name']}"] = data
            panels.append({"name": panel["name"], "kind": panel["kind"], "data_path": str(panel["data_path"]),
                           "grid_shape": list(data.shape), "marks": marks})
        if item_id in limits:
            item_limits = _check_limits(limits[item_id], item_id)
        else:
            item_limits = compute_limits(arrays, config)
        for panel, data in zip(panels, arrays):
            for variation in variations[item_id]:
                cells[f"{item_id}/{variation['variation_id']}/{panel['name']}"] = \
                    apply_mapping(data, item_limits, variation["mapping"])
        items.append({"item_id": item_id, "label": str(item.get("label", item_id)), "limits": item_limits,
                      "panels": panels})

    layout = _layout(config, [len(item["panels"]) for item in items])
    per_page = config["page"]["items_per_page"]
    chunks = [items[i:i + per_page] for i in range(0, len(items), per_page)]

    trial_id = str(uuid.uuid4())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"trial_r{round_id}.pdf"
    cells_path = out_dir / f"trial_r{round_id}_cells.npz"
    manifest_path = out_dir / f"trial_r{round_id}_manifest.json"

    pages = []
    with rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42}):
        with PdfPages(pdf_path) as pdf:
            for number, chunk in enumerate(chunks, start=1):
                drawn = [item | {"variations": variations[item["item_id"]]} for item in chunk]
                fig, page = _draw_page(number, len(chunks), drawn, round_id, parameter, trial_id, layout, cells)
                pdf.savefig(fig)
                pages.append(page)
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
                "pages": len(pages), "page_size_mm": [int(PAGE_W), int(PAGE_H)]},
        "print_settings": DEFAULT_PRINT_SETTINGS | (print_settings or {}),
        "cells_path": str(cells_path),
        "pages": pages,
    }
    manifest_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record
