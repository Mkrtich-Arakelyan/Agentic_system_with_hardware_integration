"""Numerical heatmap figures using frozen selection, arrays and normalization.

The report uses render_report_comparison: one E1-E4 row per frozen experiment,
scientific Initial/Final before tuning and Initial/Final after tuning. Mappings
come from the original cycle-1 reference and the last completed cycle's winners.
render_final_heatmaps retains the legacy final-mapping-only view for tuning clients.

The session is read with tuning_tools' key names (_session_inputs, _load_selection). The tuned
mappings come from the session's final_settings (experiment_id -> complete mapping), read only in
_final_settings. The limits are the fixed ones in selected_experiments[].limits: none are recomputed,
so the grays are exactly those apply_mapping gives and the trial sheets printed.
"""
import hashlib
import json
import math
import uuid
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from app.report_policy import COMPARISON_VERSION
from app.print_tuning.grayscale_render import apply_mapping, mapping_label, scale_bar, validate_mapping
from app.print_tuning.trial_sheet import _check_marks, _draw_box_mark, load_panel_data
from app.print_tuning.tuning_tools import _load_selection, _session_inputs, _ToolError

CONTRACT_VERSION = 1
MM_PER_INCH = 25.4
DPI = 300
PANELS = ("Initial", "Final")
MARGIN, TITLE_H = 6.0, 5.0             # mm: page margin, column title line
LABEL_H, LABEL_LINE_H = 8.0, 3.4      # mm: the two row label lines above each row, and one line's height
PANEL_GAP = 4.0                        # mm: gap between the two panels
BAR_GAP, BAR_H, TICK_H, ROW_GAP = 1.2, 2.0, 3.0, 3.0   # mm: below each row's panels
FONT_PT, SMALL_PT, MIN_PT = 7, 6, 5    # a text reaching outside the figure shrinks, but not below MIN_PT
SHORT_ID = 8                           # characters of the experiment id shown in the row label
ID_PREFIX = "experiment_"


def _final_settings(session: dict, experiment_ids: list[str], config: dict) -> dict[str, dict]:
    """experiment_id -> tuned mapping, checked; the only place that knows the session key final_settings."""
    settings = session.get("final_settings")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("tuning session: final_settings (experiment_id -> mapping) is missing; "
                         "finish round 3 first")
    missing = [e for e in experiment_ids if e not in settings]
    if missing:
        raise ValueError(f"tuning session: final_settings has no mapping for {missing}")
    mappings = {}
    for experiment_id in experiment_ids:
        mapping = settings[experiment_id]
        try:
            validate_mapping(mapping, config)
        except ValueError as error:
            raise ValueError(f"tuning session: final_settings[{experiment_id!r}] is not a valid mapping: "
                             f"{error}") from error
        mappings[experiment_id] = dict(mapping)
    pairs = {e: (m["dark_level"], m["light_level"]) for e, m in mappings.items()}
    if len(set(pairs.values())) > 1:
        raise ValueError(f"tuning session: dark_level and light_level are shared and must be the same for every "
                         f"experiment, got {pairs}")
    return mappings


def _fixed_limits(limits, experiment_id: str) -> dict:
    lo, hi = (limits.get("lo"), limits.get("hi")) if isinstance(limits, dict) else (None, None)
    finite = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (lo, hi))
    if not finite or hi <= lo:
        raise ValueError(f"tuning session: selected_experiments[{experiment_id!r}].limits needs finite lo < hi, "
                         f"got {limits!r}; prepare_specialist_previews computes them before round 1")
    return dict(limits)


def _axes(fig: Figure, box: tuple, width_mm: float, height_mm: float):
    """An axes at box = (x, y, w, h) in mm from the figure's top-left corner."""
    x, y, w, h = box
    ax = fig.add_axes([x / width_mm, 1 - (y + h) / height_mm, w / width_mm, h / height_mm])
    ax.set_axis_off()
    return ax


def _frame(box: tuple) -> Rectangle:
    """A thin black outline around an mm box, drawn on the full-figure axes."""
    x, y, w, h = box
    return Rectangle((x, y), w, h, facecolor="none", edgecolor="black", linewidth=0.3)


def _short_id(experiment_id: str) -> str:
    """The first SHORT_ID characters after "experiment_", or of the whole id without that prefix."""
    return experiment_id.removeprefix(ID_PREFIX)[:SHORT_ID]


def _fit_texts(fig: Figure) -> None:
    """Shrink every text reaching outside the figure, 0.5 pt at a time and not below MIN_PT."""
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for ax in fig.axes:
        for text in ax.texts:
            while True:
                extent = text.get_window_extent(renderer)
                inside = (extent.x0 >= bounds.x0 and extent.x1 <= bounds.x1
                          and extent.y0 >= bounds.y0 and extent.y1 <= bounds.y1)
                if inside or text.get_fontsize() <= MIN_PT:
                    break
                text.set_fontsize(max(MIN_PT, text.get_fontsize() - 0.5))


def _new_path(out_dir: Path, session_id: str) -> Path:
    path = out_dir / f"final_heatmaps_{session_id[:8]}.png"
    while path.exists():
        path = out_dir / f"final_heatmaps_{session_id[:8]}_{uuid.uuid4().hex[:6]}.png"
    return path


def render_final_heatmaps(session_path, out_dir, *, figsize_mm: tuple[float, float] = (180, 200), compact=False) -> dict:
    """Draw the selected experiments' Initial and Final heatmaps with their final settings; save a gray PNG.

    Raises ValueError for a bad session or missing or invalid final settings, and TrialSheetError
    (a ValueError) for bad panel data. Never overwrites an earlier figure.
    """
    if compact:
        return render_report_comparison(session_path, out_dir)
    session_path = Path(session_path)
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the tuning session {session_path}: {error}") from error
    if not isinstance(session, dict):
        raise ValueError(f"the tuning session {session_path} must hold a JSON object")
    try:
        inputs = _session_inputs(session, None)
        selection, item_ids = _load_selection(inputs)
    except _ToolError as error:
        raise ValueError(str(error)) from error
    experiment_ids, config = inputs["experiment_ids"], inputs["config"]
    mappings = _final_settings(session, experiment_ids, config)
    limits = {e: _fixed_limits(inputs["limits"][e], e) for e in experiment_ids}
    items = {item["item_id"]: item for item in selection["items"]}

    # load every panel before drawing anything
    rows = []
    for experiment_id in experiment_ids:
        item = items[item_ids[experiment_id]]
        by_name = {panel["name"]: panel for panel in item["panels"]}
        missing = [name for name in PANELS if name not in by_name]
        if missing:
            raise ValueError(f"selection item {item['item_id']} has no {missing} panel")
        panels = []
        for name in PANELS:          # the same loading and mark checks as trial_sheet._load_items
            panel, where = by_name[name], f"item {item['item_id']} panel {name}"
            data = load_panel_data(panel["data_path"], where)
            if panel.get("sha256") and hashlib.sha256(Path(panel["data_path"]).read_bytes()).hexdigest() != panel["sha256"]:
                raise ValueError("Original scientific array changed after selection: " + where)
            marks = _check_marks(item.get("marks"), where, data.shape) + \
                _check_marks(panel.get("marks"), where, data.shape)
            panels.append({"name": name, "grays": apply_mapping(data, limits[experiment_id], mappings[experiment_id]),
                           "marks": marks})
        rows.append({"experiment_id": experiment_id, "label": str(item.get("label", item["item_id"])),
                     "panels": panels})

    # layout in mm: square panels as large as the rows allow, centred
    width_mm, height_mm = (float(v) for v in figsize_mm)
    row_h = (height_mm - 2 * MARGIN - TITLE_H) / len(rows)
    side = min(row_h - LABEL_H - BAR_GAP - BAR_H - TICK_H - ROW_GAP, (width_mm - 2 * MARGIN - PANEL_GAP) / 2)
    if side <= 5:
        raise ValueError(f"figsize_mm {figsize_mm} is too small for {len(rows)} rows of heatmaps")
    pair_w = 2 * side + PANEL_GAP
    x0 = (width_mm - pair_w) / 2
    fig = Figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), dpi=DPI, facecolor="white")
    FigureCanvasAgg(fig)
    page = _axes(fig, (0, 0, width_mm, height_mm), width_mm, height_mm)
    page.set_xlim(0, width_mm)
    page.set_ylim(height_mm, 0)
    for k, name in enumerate(PANELS):
        page.text(x0 + k * (side + PANEL_GAP) + side / 2, MARGIN + TITLE_H / 2, name, fontsize=FONT_PT + 1,
                  ha="center", va="center", weight="bold")

    for index, row in enumerate(rows):
        experiment_id = row["experiment_id"]
        mapping, row_limits = mappings[experiment_id], limits[experiment_id]
        top = MARGIN + TITLE_H + index * row_h
        line_y = top + (LABEL_H - 2 * LABEL_LINE_H) / 2 + LABEL_LINE_H / 2
        page.text(x0, line_y, f"Experiment {index + 1} · {_short_id(experiment_id)} · {row['label']}",
                  fontsize=FONT_PT, ha="left", va="center")
        page.text(x0, line_y + LABEL_LINE_H, mapping_label(mapping), fontsize=FONT_PT, ha="left", va="center")
        panel_y = top + LABEL_H
        for k, panel in enumerate(row["panels"]):
            box = (x0 + k * (side + PANEL_GAP), panel_y, side, side)
            ax = _axes(fig, box, width_mm, height_mm)
            grays = panel["grays"]
            n_rows, n_cols = grays.shape
            ax.imshow(grays, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto",
                      extent=(0, n_cols, n_rows, 0))
            ax.set_xlim(0, n_cols)
            ax.set_ylim(n_rows, 0)
            for mark in panel["marks"]:
                _draw_box_mark(ax, mark, grays.shape, box)
            page.add_patch(_frame(box))

        bar_box = (x0, panel_y + side + BAR_GAP, pair_w, BAR_H)
        bar = scale_bar(row_limits, mapping)
        bar_ax = _axes(fig, bar_box, width_mm, height_mm)
        bar_ax.imshow(bar["grays"][np.newaxis, :], cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                      aspect="auto", extent=(0, 1, 1, 0))
        page.add_patch(_frame(bar_box))
        tick_y = bar_box[1] + BAR_H + 0.3
        page.text(x0, tick_y, f"{bar['ticks'][0]['label']}", fontsize=SMALL_PT, ha="left", va="top")
        page.text(x0 + pair_w, tick_y, f"{bar['ticks'][-1]['label']}", fontsize=SMALL_PT, ha="right", va="top")
        page.text(x0 + pair_w / 2, tick_y, "data value", fontsize=SMALL_PT, ha="center", va="top", color="0.35")

    # render, then store as 8-bit gray so it never prints in colour
    fig.canvas.draw()
    _fit_texts(fig)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _new_path(out_dir, inputs["session_id"])
    Image.fromarray(rgba[..., :3]).convert("L").save(path, dpi=(DPI, DPI))
    return {"contract_version": CONTRACT_VERSION, "status": "ok", "figure_path": str(path),
            "figure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width_mm": width_mm, "height_mm": height_mm,
            "experiments": [{"experiment_id": row["experiment_id"], "label": row["label"],
                             "mapping": mappings[row["experiment_id"]], "limits": limits[row["experiment_id"]]}
                            for row in rows]}


COMPARISON_WIDTH_MM, COMPARISON_HEIGHT_MM = 180., 184.
COMPARISON_FONT_PT = 10
COMPARISON_GROUPS = ("Before printer tuning", "After printer tuning")
COMPARISON_COLUMNS = ["Initial", "Final", "Initial", "Final"]


def comparison_metadata(session):
    """Bind both mappings to recorded history, never the latest cycle's baseline.

    Version 1 sessions have one cycle in rounds; version 2 retains cycle 1 in
    cycles. Fail closed for absent or conflicting historical settings.
    """
    ids = [row["experiment_id"] for row in session["selected_experiments"]]
    config = session["config_snapshot"]
    after = _final_settings(session, ids, config)
    cycle = session.get("active_cycle", 1)
    if session.get("contract_version") == 2:
        first = session["cycles"]["1"]["steps"]
        last = session["cycles"][str(cycle)]["steps"]
    else:
        first = last = session["rounds"]
    before = first["1"].get("reference_settings")
    if not isinstance(before, dict) or set(before) != set(ids):
        raise ValueError("Original cycle-1 step-1 reference settings are missing")
    if session.get("original_settings") is not None and session["original_settings"] != before:
        raise ValueError("Original baseline differs from cycle-1 step-1 reference")
    previews = session.get("previews") or {}
    if previews.get("reference_mappings") is not None and previews["reference_mappings"] != before:
        raise ValueError("Frozen preview baseline differs from cycle-1 step-1 reference")
    if last["3"].get("winners") != after:
        raise ValueError("Final settings differ from the last completed cycle's accepted step-3 mappings")
    for mapping in before.values():
        validate_mapping(mapping, config)
    return {"comparison_version": COMPARISON_VERSION, "groups": list(COMPARISON_GROUPS),
            "columns": list(COMPARISON_COLUMNS), "minimum_font_pt": COMPARISON_FONT_PT,
            "width_mm": COMPARISON_WIDTH_MM, "height_mm": COMPARISON_HEIGHT_MM,
            "mapping_sources": {"before": {"cycle_id": 1, "step_id": 1, "role": "reference"},
                                "after": {"cycle_id": cycle, "step_id": 3, "role": "accepted"}},
            "experiments": [{"experiment_id": eid, "label": f"E{i + 1}",
                "before_mapping": dict(before[eid]), "after_mapping": dict(after[eid]),
                "mapping": dict(after[eid]),  # compatibility with final-settings consumers
                "limits": _fixed_limits(row["limits"], eid)}
                for i, (eid, row) in enumerate(zip(ids, session["selected_experiments"]))]}


def render_report_comparison(session_path, out_dir):
    """Four panels per experiment from unchanged arrays and frozen limits.

    Two equal-sized Initial/Final pairs use the original and accepted mappings.
    The PNG owns its labels: embed at full 180 mm width to retain 10 pt text.
    """
    session = json.loads(Path(session_path).read_text(encoding="utf-8"))
    metadata = comparison_metadata(session)
    inputs = _session_inputs(session, None)
    selection, item_ids = _load_selection(inputs)
    items = {item["item_id"]: item for item in selection["items"]}
    if len(metadata["experiments"]) != 4:
        raise ValueError("Report comparison requires four frozen experiments")
    rows = []
    for row in metadata["experiments"]:
        item = items[item_ids[row["experiment_id"]]]
        by_name = {panel["name"]: panel for panel in item["panels"]}
        arrays = []
        for name in PANELS:
            panel = by_name.get(name)
            if panel is None:
                raise ValueError(f"Selection item {item['item_id']} has no {name} panel")
            data = load_panel_data(panel["data_path"], f"item {item['item_id']} panel {name}")
            if not panel.get("sha256") or hashlib.sha256(Path(panel["data_path"]).read_bytes()).hexdigest() != panel["sha256"]:
                raise ValueError("Original scientific array changed after selection")
            marks = _check_marks(item.get("marks"), name, data.shape) + _check_marks(panel.get("marks"), name, data.shape)
            arrays.append((data, marks))
        panels = [(apply_mapping(data, row["limits"], mapping), marks)
                  for mapping in (row["before_mapping"], row["after_mapping"]) for data, marks in arrays]
        rows.append(panels)

    width, height = metadata["width_mm"], metadata["height_mm"]
    fig = Figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH), dpi=DPI, facecolor="white")
    FigureCanvasAgg(fig)
    page = _axes(fig, (0, 0, width, height), width, height)
    page.set_xlim(0, width)
    page.set_ylim(height, 0)
    # 2 mm margins, 10 mm label gutter; pair gap > three times inner gap.
    side, inner_gap, group_gap = 37.5, 3., 10.
    xs = [12., 12. + side + inner_gap, 12. + 2 * side + inner_gap + group_gap,
          12. + 3 * side + 2 * inner_gap + group_gap]
    for i, label in enumerate(COMPARISON_GROUPS):
        page.text((xs[2*i] + xs[2*i+1] + side) / 2, 3.5, label, fontsize=10,
                  ha="center", va="center", weight="bold")
    for x, label in zip(xs, COMPARISON_COLUMNS):
        page.text(x + side / 2, 10.5, label, fontsize=10, ha="center", va="center")
    for i, panels in enumerate(rows):
        y = 16 + i * 41.5
        page.text(2, y + side / 2, f"E{i + 1}", fontsize=10, va="center", weight="bold")
        for x, (grays, marks) in zip(xs, panels):
            box = (x, y, side, side)
            ax = _axes(fig, box, width, height)
            nrows, ncols = grays.shape
            ax.imshow(grays, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal",
                      extent=(0, ncols, nrows, 0))
            ax.set_xlim(0, ncols)
            ax.set_ylim(nrows, 0)
            # Marks retain their geometry even when rectangular data is letterboxed.
            plot_w, plot_h = (side, side * nrows / ncols) if ncols >= nrows else (side * ncols / nrows, side)
            for mark in marks:
                _draw_box_mark(ax, mark, grays.shape, (x, y, plot_w, plot_h))
            page.add_patch(_frame(box))
    fig.canvas.draw()
    # No font shrinking: all text must fit at the declared printed size.
    renderer = fig.canvas.get_renderer()
    if any(not fig.bbox.contains(*point) for ax in fig.axes for text in ax.texts
           for point in text.get_window_extent(renderer).get_points()):
        raise ValueError("Comparison labels exceed the figure bounds")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = _new_path(out, inputs["session_id"])
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert("L").save(path, dpi=(DPI, DPI))
    fig.clear()
    return {"contract_version": 2, "status": "ok", "figure_path": str(path),
            "figure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **metadata}

