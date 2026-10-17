"""Comparable cell-pair history from immutable accepted scans, never display PNGs."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from app.print_tuning import readability as metric


def cycles_of(session):
    cycles = deepcopy(session.get("cycles", {}))
    cycles[str(session.get("active_cycle", 1))] = {"steps": session["rounds"]}
    return sorted(((int(key), value["steps"]) for key, value in cycles.items()), key=lambda item: item[0])


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize_cells(data, limits, measurements, mappings, mask, method=None):
    """One shared denominator AND noise floor over all measurable snapshots.

    Missing captures remain missing; they do not remove pairs from other captures.
    Adding a usable capture recomputes every summary from retained cell statistics.
    """
    settings = metric.method_settings(method)
    present = [m for m in measurements if m is not None]
    if not present:
        return [None] * len(measurements), {"reason": "No measurable captures"}
    common = np.logical_and.reduce([m["valid"] & np.isfinite(m["means"]) & np.isfinite(m["stds"]) for m in present])
    floor = max([settings["quantization_floor"], *[m["noise_floor"] for m in present]])
    outputs, diagnostics = [], {}
    for measured, mapping in zip(measurements, mappings):
        if measured is None:
            outputs.append(None)
            continue
        common_stats = {**measured, "valid": common, "noise_floor": floor}
        result = metric.compare_pairs(data, limits, common_stats, common_stats, [mapping, mapping], mask=mask, method=settings)
        diagnostics = {k: result.get(k) for k in ("eligible_pairs", "jointly_valid_pairs", "coverage", "pair_indices",
            "noise_floor", "clipped_source_cells", "nonfinite_source_cells", "masked_cells", "reason")}
        outputs.append(result["before"])
    return outputs, diagnostics


def measure_history(session, *, method=None):
    settings = metric.method_settings(method or session.get("metric_method"))
    metric.need(not session.get("metric_method") or settings == session["metric_method"], "Metric method is frozen across the run")
    snapshots = [("Baseline", 1, 1, False)]
    for cycle, steps in cycles_of(session):
        for step in (1, 2, 3):
            if steps[str(step)].get("winners"):
                snapshots.append((f"C{cycle}-S{step}", cycle, step, True))
    result = {"version": "metric-progress/1", "method": settings, "labels": [s[0] for s in snapshots],
              "experiments": [], "captures": {}, "source_panel": "Final",
              "note": "Separate captures; unknown acquisition settings do not establish scanner consistency. Paper/black normalization does not undo nonlinear scanner processing. Detail and CNR summarize the same pair evidence, not independent proof."}
    views = {cycle: {**session, "active_cycle": cycle, "rounds": steps} for cycle, steps in cycles_of(session)}
    cache = {}
    previous = session.get("metric_history") or {}
    for label, cycle, step, _ in snapshots:
        key = (cycle, step)
        if key not in cache:
            try:
                capture = metric.accepted_capture(views[cycle], step)
                if "cycle_id" in capture["manifest"]:
                    metric.need(capture["manifest"]["cycle_id"] == cycle, "Capture belongs to a different cycle")
                measured = {}
                # Reuse immutable cell measurements only after verifying the original
                # evaluation, manifest, scan and source hashes above/below. Thresholds
                # and source-pair definitions must also match the frozen method.
                reusable = previous.get("method") == settings
                if reusable:
                    for experiment, old_row in zip(session["selected_experiments"], previous.get("experiments", [])):
                        for winner in ((False, True) if key == (1, 1) else (True,)):
                            point_label = f"C{cycle}-S{step}" if winner else "Baseline"
                            indices = [i for i, p in enumerate(old_row["points"]) if p["label"] == point_label]
                            try:
                                metric.need(len(indices) == 1 and old_row["experiment_id"] == experiment["experiment_id"], "Cache identity mismatch")
                                index = indices[0]; old_point = old_row["points"][index]
                                metric.need(previous["captures"][point_label]["evaluation_sha256"] == capture["provenance"]["evaluation_sha256"], "Cache evaluation mismatch")
                                candidate = metric.candidate_evidence(views[cycle], capture, experiment, step, winner=winner)
                                evidence = old_point["evidence"]
                                metric.need(evidence["mapping"] == candidate["mapping"] and evidence["source"] == candidate["source"], "Cache source/mapping mismatch")
                                stats = {k: np.array(v, dtype=bool if k == "valid" else float) if k in ("means", "stds", "valid") else v
                                         for k, v in old_row["cell_measurements"][index].items()}
                                compact = {k: evidence[k] for k in ("calibration", "dpi", "bilevel", "noise_floor")}
                                compact["curve"] = np.array(evidence["curve"])
                                measured[experiment["experiment_id"], winner] = (candidate, stats, compact, np.array(old_row["mask"], dtype=bool), None)
                            except (KeyError, ValueError, TypeError, IndexError):
                                reusable = False
                    reusable = reusable and len(measured) == len(session["selected_experiments"]) * (2 if key == (1, 1) else 1)
                if reusable:
                    cache[key] = (capture, measured, None)
                    result["captures"][label] = capture["provenance"]
                    continue
                pages = metric.aligned_capture(capture)
                for experiment in session["selected_experiments"]:
                    for winner in ((False, True) if key == (1, 1) else (True,)):
                        try:
                            candidate = metric.candidate_evidence(views[cycle], capture, experiment, step, winner=winner)
                            page = pages[candidate["page_number"]]
                            measured[experiment["experiment_id"], winner] = (candidate, metric.cell_statistics(page, candidate),
                                {k: page[k] for k in ("curve", "calibration", "dpi", "bilevel", "noise_floor")},
                                metric.scan._cell_masks(candidate["plot"]["box"], candidate["data"].shape, candidate["marks"])[0], None)
                        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                            measured[experiment["experiment_id"], winner] = (None, None, None, None, metric.evidence_reason(error))
                cache[key] = (capture, measured, None)
                del pages
            except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                cache[key] = (None, None, metric.evidence_reason(error))
        capture, _, error = cache[key]
        result["captures"][label] = {"reason": error} if error else capture["provenance"]
    for index, experiment in enumerate(session["selected_experiments"]):
        cells, mappings, evidence, reasons, baseline_page, baseline_acquisition = [], [], [], [], None, None
        try:
            data = np.load(next(p["data_path"] for p in experiment["panels"] if p["name"] == "Final"), allow_pickle=False)
        except (OSError, ValueError, KeyError, StopIteration) as error:
            result["experiments"].append({"label": f"E{index + 1}", "experiment_id": experiment["experiment_id"],
                "coverage": {"reason": metric.evidence_reason(error)}, "cell_measurements": [None] * len(snapshots),
                "points": [{"label": label, "metrics": None, "reason": metric.evidence_reason(error)} for label, *_ in snapshots]})
            continue
        mask = np.ones(data.shape, dtype=bool)
        for label, cycle, step, winner in snapshots:
            capture, measured, error = cache[cycle, step]
            try:
                metric.need(not error, error)
                candidate, statistics, page, annotation_mask, error = measured[experiment["experiment_id"], winner]
                metric.need(not error, error)
                if baseline_page is None:
                    baseline_page, baseline_acquisition, mask = page, capture["provenance"]["acquisition_settings"], annotation_mask
                metric.need(np.array_equal(mask, annotation_mask), "Frozen annotation mask differs")
                acquisition = capture["provenance"]["acquisition_settings"]
                consistency = "different" if acquisition and baseline_acquisition and acquisition != baseline_acquisition else "unknown"
                mismatch = metric.comparability_reason([baseline_page, page], consistency, settings)
                metric.need(not mismatch, mismatch)
                cells.append(statistics)
                mappings.append(candidate["mapping"])
                evidence.append({k: deepcopy(candidate[k]) for k in ("candidate_id", "mapping", "source", "limits", "marks", "plot", "page_number")} |
                    {"calibration": page["calibration"], "dpi": page["dpi"], "bilevel": page["bilevel"], "noise_floor": page["noise_floor"], "curve": page["curve"]})
                reasons.append(None)
            except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                cells.append(None); mappings.append({}); evidence.append(None); reasons.append(metric.evidence_reason(error))
        summaries, coverage = summarize_cells(data, experiment["limits"], cells, mappings, mask, settings)
        points = [{"label": label, "metrics": summary, "reason": reason or (coverage.get("reason") if summary is None else None),
                   "evidence": item} for (label, *_), summary, reason, item in zip(snapshots, summaries, reasons, evidence)]
        result["experiments"].append({"label": f"E{index + 1}", "experiment_id": experiment["experiment_id"],
            "source_sha256": next(p["sha256"] for p in experiment["panels"] if p["name"] == "Final"),
            "limits": experiment["limits"], "mask": mask.tolist(), "coverage": coverage,
            "cell_measurements": cells, "points": points})
    return finite_json(result)


def describe_history(history):
    sentences = []
    trends = {}
    for experiment in history["experiments"]:
        points = experiment["points"]
        a, b = points[0]["metrics"], points[-1]["metrics"]
        if not a or not b:
            continue
        detail, cnr = b["detail_percent"] - a["detail_percent"], b["median_cnr"] - a["median_cnr"]
        ends = [p["metrics"] for p in points if p["label"].endswith("S3")]
        if len(ends) > 1 and ends[-2] and ends[-1]:
            d, c = ends[-1]["detail_percent"] - ends[-2]["detail_percent"], ends[-1]["median_cnr"] - ends[-2]["median_cnr"]
            trend = "plateaus" if abs(d) < .1 and abs(c) < .01 else "gains" if d >= 0 and c >= 0 else "regresses" if d <= 0 and c <= 0 else "trades detail against CNR"
            trends.setdefault(trend, []).append(experiment["label"])
        sentences.append(f"{experiment['label']}: detail {detail:+.1f} pp; median CNR {cnr:+.2f}.")
    for trend, labels in trends.items():
        sentences.append("Latest cycle " + trend + " for " + ", ".join(labels) + ".")
    if not sentences:
        reasons = list(dict.fromkeys(str(p.get("reason")) for e in history["experiments"] for p in (e["points"][0], e["points"][-1]) if p.get("reason")))
        return "Baseline/final readability comparison not available: " + "; ".join(reasons[:2]) + "."
    return " ".join(sentences)


def render_progress(history, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    charts = {}
    labels = history["labels"]
    for key, title, ylabel in (("detail_percent", "Detail preservation progress", "Detail preservation (%)"),
                               ("median_cnr", "Median CNR progress", "Median CNR (dimensionless)")):
        with plt.rc_context({"font.family": "serif", "font.size": 10}):
            fig, ax = plt.subplots(figsize=(7.08, 2.24), layout="constrained")
            for experiment, style, marker in zip(history["experiments"], ("-", "--", "-.", ":"), ("o", "s", "^", "D")):
                values = [p["metrics"][key] if p["metrics"] else np.nan for p in experiment["points"]]
                ax.plot(range(len(labels)), values, color="black", linestyle=style, marker=marker, markersize=4, label=experiment["label"])
            for i, label in enumerate(labels):
                if label.endswith("S1") and label != "C1-S1": ax.axvline(i - .5, color=".6", linewidth=.6)
            ticks = sorted(set([0, len(labels) - 1, *range(0, len(labels), max(1, int(np.ceil(len(labels) / 12))))]))
            ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=45 if len(labels) > 7 else 0, ha="right" if len(labels) > 7 else "center")
            if key == "detail_percent": ax.set_ylim(0, 100)
            else: ax.set_ylim(bottom=0)
            ax.set_title(title, pad=27); ax.set_ylabel(ylabel)
            ax.legend(ncol=4, fontsize=10, loc="lower center", bbox_to_anchor=(.5, 1.01), frameon=False)
            ax.grid(axis="y", color=".85", linewidth=.5)
            path = out / (key + "_progress.pdf")
            fig.savefig(path); plt.close(fig)
            charts[key] = {"path": str(path), "sha256": metric.sha(path), "labels": labels,
                           "missing_points": "line breaks; no interpolation", "width_mm": 180}
    return charts
