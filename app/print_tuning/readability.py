"""Offline, project-specific cell-pair print/scan measurements. Never writes evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from app.print_tuning import scan_evaluation as scan

METHOD = {"version": "cell-pairs/1", "source_difference": .05, "cnr_threshold": 3.,
          "minimum_coverage": .8, "interior_fraction": .5, "smoothing_mm": .3,
          "pixels_per_mm": scan.PX_PER_MM, "quantization_floor": 1 / 255,
          "minimum_calibration_range": .2, "maximum_patch_curve_difference": .2}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def method_settings(overrides=None):
    settings = METHOD | (overrides or {})
    adjustable = {"source_difference", "cnr_threshold", "minimum_coverage"}
    need(all(k in METHOD for k in settings), "Unknown metric setting")
    need(all(settings[k] == v for k, v in METHOD.items() if k not in adjustable),
         "Changing sampling/calibration requires a new implemented method version")
    need(all(isinstance(settings[k], (float, int)) and np.isfinite(settings[k]) for k in adjustable), "Nonfinite metric threshold")
    need(0 < settings["source_difference"] <= 1 and 0 < settings["minimum_coverage"] <= 1
         and settings["cnr_threshold"] > 0, "Invalid pair thresholds")
    return settings


def evidence_reason(error):
    if isinstance(error, StopIteration):
        return "Required Final panel or candidate evidence not available"
    if isinstance(error, FileNotFoundError):
        return "Evidence file missing: " + Path(error.filename or "unknown").name
    if isinstance(error, KeyError):
        return "Legacy evidence field not recorded: " + str(error).strip("'")
    return str(error)


def acquisition_metadata(files):
    metadata = []
    for file in files:
        path = Path(file["path"])
        if path.suffix.lower() == ".pdf":
            with pymupdf.open(path) as document:
                metadata.append({"path": str(path), "pdf_metadata": document.metadata,
                    "page_count": len(document), "note": "Producer/creator metadata does not confirm acquisition settings"})
        else:
            with Image.open(path) as image:
                metadata.append({"path": str(path), "mode": image.mode, "size": list(image.size),
                    "dpi": image.info.get("dpi"), "scan_provenance": image.info.get("scan_provenance"),
                    "exif_acquisition": {str(k): str(v) for k, v in image.getexif().items() if k in (271, 272, 305, 282, 283, 296)}})
    return metadata


def accepted_capture(session, number):
    """Select by the accepted review's explicit evaluation reference, never recency."""
    entry = session["rounds"][str(number)]
    review = entry.get("review") or {}
    need(review.get("assessment") == "usable", f"Round {number} has no accepted usable review")
    links = [r for r in entry.get("reviews", []) if r.get("response") == review]
    need(len(links) == 1, f"Round {number} accepted review has missing/ambiguous evidence linkage")
    link = links[0]
    records = [e for e in entry.get("evaluations", []) if e.get("record_id") == link["evaluation_record_id"]]
    need(len(records) == 1, "Accepted evaluation is missing/ambiguous")
    record = records[0]
    path = Path(record["result"]["evaluation_path"])
    need(record.get("evaluation_sha256") == sha(path), "Accepted evaluation hash changed")
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    need(evaluation["scan_quality"]["usable"] is True, "Accepted scan is unusable")
    need(evaluation["trial_id"] == link["trial_id"] and evaluation["round_id"] == number
         and evaluation["session_id"] == session["session_id"], "Evaluation identity mismatch")
    trial = entry.get("trial_result") or {}
    need(trial.get("trial_id") == link["trial_id"], "Accepted review belongs to superseded trial")
    manifest_path = Path(trial["manifest_path"])
    stored_trials = [t for t in entry.get("trials", []) if t["result"]["trial_id"] == trial["trial_id"]]
    need(len(stored_trials) == 1 and stored_trials[0].get("manifest_sha256") == sha(manifest_path),
         "Recorded trial manifest hash missing or changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    need(manifest["trial_id"] == evaluation["trial_id"] and manifest["round_id"] == number
         and manifest["calibration_session_id"] == session["session_id"], "Manifest identity mismatch")
    need(Path(evaluation["manifest_path"]).resolve() == manifest_path.resolve(), "Evaluation/manifest path mismatch")
    need([i["item_id"] for i in manifest["items"]] == [e["item_id"] for e in session["selected_experiments"]], "Manifest experiment order mismatch")
    need(evaluation["scan_files"] == record["result"]["scan_files"], "Recorded scan provenance differs from evaluation")
    need(manifest["pdf"]["sha256"] == trial["pdf_sha256"] == sha(trial["pdf_path"]), "Trial PDF changed")
    for file in evaluation["scan_files"]:
        need(sha(file["path"]) == file["sha256"], "Original scan hash changed")
    return {"evaluation": evaluation, "manifest": manifest, "review": review,
            "provenance": {"evaluation_path": str(path), "evaluation_sha256": sha(path),
                "manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path),
                "trial_id": link["trial_id"], "scan_files": deepcopy(evaluation["scan_files"]),
                "acquisition_metadata": acquisition_metadata(evaluation["scan_files"]),
                "scan_provenance": evaluation.get("scan_provenance", record["result"].get("scan_provenance",
                    [f.get("scan_provenance", "unknown") for f in evaluation["scan_files"]]
                    if any(f.get("scan_provenance") for f in evaluation["scan_files"]) else "unknown")),
                "acquisition_settings": evaluation.get("acquisition_settings")}}


def candidate_evidence(session, capture, experiment, number):
    """Bind experiment, Final source, limits, complete mapping and candidate location."""
    eid, item_id = experiment["experiment_id"], experiment["item_id"]
    manifest, evaluation = capture["manifest"], capture["evaluation"]
    items = [i for i in manifest["items"] if i["item_id"] == item_id]
    need(len(items) == 1, "Missing/duplicate experiment in manifest")
    item = items[0]
    need(item["limits"] == experiment["limits"], "Frozen normalization limits changed")
    source = next(p for p in experiment["panels"] if p["name"] == "Final")
    panel = next(p for p in item["panels"] if p["name"] == "Final")
    need(source["sha256"] == panel["sha256"] == sha(source["data_path"]), "Final source array hash mismatch")
    need(Path(source["data_path"]).resolve() == Path(panel["data_path"]).resolve(), "Final panel path mismatch")
    need(source.get("marks", []) == panel.get("marks", []), "Final panel annotation mask changed")
    variations = manifest["variations"][item_id]
    if number == 1:
        refs = [v for v in variations if v["role"] == "reference"]
        need(len(refs) == 1, "Missing/ambiguous starting reference")
        variation = refs[0]
        expected = session["rounds"]["1"]["reference_settings"][eid]
    else:
        winners = [w for w in capture["review"]["step_winners"] if w["experiment_id"] == eid]
        need(len(winners) == 1, "Missing/duplicate final steps winner")
        selected = [v for v in variations if v["variation_id"] == winners[0]["candidate_id"]]
        need(len(selected) == 1, "Winner does not belong to the recorded manifest")
        variation, expected = selected[0], session["final_settings"][eid]
    need(variation["mapping"] == expected, "Complete candidate mapping differs from accepted settings")
    matches = [c for c in evaluation["candidates"] if c["experiment_id"] == item_id
               and c["candidate_id"] == variation["variation_id"]]
    need(len(matches) == 1 and matches[0]["mapping"] == expected
         and matches[0]["role"] == variation["role"], "Evaluation candidate mapping/identity mismatch")
    need(sum(p["panel"] == "Final" for p in matches[0]["panels"]) == 1, "Missing/duplicate evaluated Final panel")
    plots = [(page, p) for page in manifest["pages"] for p in page["regions"]["plots"]
             if (p["item_id"], p["variation_id"], p["panel"]) == (item_id, variation["variation_id"], "Final")]
    need(len(plots) == 1, "Missing/duplicate Final plot geometry")
    page, plot = plots[0]
    need(page["page_number"] == matches[0]["page_number"], "Candidate page mismatch")
    data = np.load(source["data_path"], allow_pickle=False)
    need(tuple(plot["grid_shape"]) == data.shape == tuple(panel["grid_shape"]), "Source/manifest grid mismatch")
    return {"candidate_id": variation["variation_id"], "mapping": deepcopy(expected), "plot": plot,
            "page_number": page["page_number"], "marks": panel.get("marks", []), "source": source,
            "limits": experiment["limits"], "data": data}


def _smooth(image):
    # Fixed .3 mm square mean at 10 samples/mm, before cell and patch statistics.
    padded = np.pad(image, 1, mode="reflect")
    return sum(padded[r:r + image.shape[0], c:c + image.shape[1]] for r in range(3) for c in range(3)) / 9


def aligned_capture(capture):
    manifest = capture["manifest"]
    raw, files = scan._read_scan([f["path"] for f in capture["evaluation"]["scan_files"]], manifest["config"]["scan"]["accepted_formats"])
    need(all(dpi >= manifest["config"]["scan"]["min_dpi"] for _, _, dpi in raw), "Scan resolution below manifest minimum")
    pages = [scan._align_page(g, rgb, dpi, manifest["pages"], i) for i, (g, rgb, dpi) in enumerate(raw)]
    numbers = [p["manifest_page"]["page_number"] for p in pages]
    need(len(set(numbers)) == len(numbers) and set(numbers) == {p["page_number"] for p in manifest["pages"]}, "Missing/duplicate aligned pages")
    output = {}
    for page in pages:
        page["image"] = _smooth(page["image"])
        response = scan._printer_response(page, page["manifest_page"]["regions"])
        paper, black = response["paper"], response["black"]
        need(paper and black, "Paper/black calibration unavailable")
        span = paper["gray"] - black["gray"]
        need(span >= METHOD["minimum_calibration_range"], "Unusable paper-to-black calibration range")
        patches = response["gray_patches"]
        need(len(patches) >= 3 and all(p["std"] is not None for p in patches), "Gray patch noise unavailable")
        curve = np.array([(p["measured"] - black["gray"]) / span for p in sorted(patches, key=lambda p: p["level"])])
        need(np.all(np.diff(curve) >= -.05), "Gray patch response is materially nonmonotonic")
        page.update(calibration=response, span=span, black=black["gray"], curve=curve,
                    noise_floor=max(float(np.median([p["std"] for p in patches])) / span, 1 / (255 * span)),
                    bilevel=any(page["scan_index"] in f.get("bilevel_scan_indices", []) for f in files))
        output[page["manifest_page"]["page_number"]] = page
    return output


def cell_statistics(page, candidate):
    shape, box = candidate["data"].shape, candidate["plot"]["box"]
    valid, _ = scan._cell_masks(box, shape, candidate["marks"])
    means, stds = np.full(shape, np.nan), np.full(shape, np.nan)
    saturated, samples = 0, 0
    x, y, w, h = box
    for row, col in np.ndindex(shape):
        interior = [x + (col + .25) * w / shape[1], y + (row + .25) * h / shape[0],
                    .5 * w / shape[1], .5 * h / shape[0]]
        pixels = scan._region(page["image"], interior, 0)
        # Alignment rejects clipped plots; finite interior pixels are still checked.
        if min(pixels.shape, default=0) < 2 or not np.isfinite(pixels).all():
            valid[row, col] = False
            continue
        means[row, col] = (pixels.mean() - page["black"]) / page["span"]
        stds[row, col] = pixels.std() / page["span"]
        saturated += int(np.sum((pixels <= 0) | (pixels >= 1)))
        samples += pixels.size
    return {"means": means, "stds": stds, "valid": valid, "noise_floor": page["noise_floor"],
            "saturated_pixel_fraction": saturated / samples if samples else None,
            "missing_cells": int(np.sum(~np.isfinite(means)))}


def compare_pairs(data, limits, before, after, mappings, *, mask=None, method=None):
    settings = method_settings(method)
    need(np.isfinite(limits["lo"]) and np.isfinite(limits["hi"]) and limits["hi"] > limits["lo"], "Invalid frozen limits")
    source = (np.asarray(data, dtype=float) - limits["lo"]) / (limits["hi"] - limits["lo"])
    valid_source = np.isfinite(source) & (source >= 0) & (source <= 1)
    if mask is not None:
        valid_source &= mask
    pairs = []
    for r, c in np.ndindex(source.shape):
        for rr, cc in ((r, c + 1), (r + 1, c)):
            if rr < source.shape[0] and cc < source.shape[1] and valid_source[r, c] and valid_source[rr, cc] \
                    and abs(source[r, c] - source[rr, cc]) >= settings["source_difference"]:
                pairs.append(((r, c), (rr, cc)))
    joint = [(a, b) for a, b in pairs if all(v["valid"][a] and v["valid"][b]
             and all(np.isfinite(v[k][p]) for k in ("means", "stds") for p in (a, b))
             for v in (before, after))]
    coverage = len(joint) / len(pairs) if pairs else None
    result = {"eligible_pairs": len(pairs), "jointly_valid_pairs": len(joint), "coverage": coverage,
              "clipped_source_cells": int(np.sum(np.isfinite(source) & ((source < 0) | (source > 1)))),
              "nonfinite_source_cells": int(np.sum(~np.isfinite(source))),
              "masked_cells": int(np.sum(~mask)) if mask is not None else 0,
              "before": None, "after": None, "detail_change_pp": None, "cnr_change": None,
              "cnr_relative_change_percent": None, "reason": None}
    if not pairs or not joint or coverage < settings["minimum_coverage"]:
        result["reason"] = "No eligible source pairs" if not pairs else "Inadequate joint scan coverage"
        return result
    floor = max(before["noise_floor"], after["noise_floor"], settings["quantization_floor"])
    need(np.isfinite(floor) and floor > 0, "Invalid finite noise floor")
    result["noise_floor"] = floor
    result["pair_indices"] = [[list(a), list(b)] for a, b in joint]
    for label, capture, mapping in zip(("before", "after"), (before, after), mappings):
        cnrs, ordered, reversed_pairs, ties = [], [], [], []
        for a, b in joint:
            difference = capture["means"][b] - capture["means"][a]
            intended = (source[b] - source[a]) * (-1 if mapping.get("invert") else 1)
            ordered.append(bool(difference * intended > 0))
            reversed_pairs.append(bool(difference * intended < 0))
            ties.append(bool(difference == 0))
            pooled = np.sqrt((capture["stds"][a] ** 2 + capture["stds"][b] ** 2) / 2)
            cnrs.append(float(abs(difference) / max(pooled, floor)))
        result[label] = {"detail_percent": float(100 * np.mean(np.asarray(ordered) & (np.asarray(cnrs) >= settings["cnr_threshold"]))),
                         "median_cnr": float(np.median(cnrs)), "wrong_order_fraction": float(np.mean(reversed_pairs)),
                         "tied_brightness_fraction": float(np.mean(ties)),
                         "pair_cnr": cnrs, "pair_correct_order": ordered}
    result["detail_change_pp"] = result["after"]["detail_percent"] - result["before"]["detail_percent"]
    result["cnr_change"] = result["after"]["median_cnr"] - result["before"]["median_cnr"]
    if result["before"]["median_cnr"] > 0:
        result["cnr_relative_change_percent"] = 100 * result["cnr_change"] / result["before"]["median_cnr"]
    return result


def comparability_reason(pages, acquisition, settings=METHOD):
    if (pages[0]["bilevel"] != pages[1]["bilevel"]
            or pages[0]["curve"].shape != pages[1]["curve"].shape
            or np.max(np.abs(pages[0]["curve"] - pages[1]["curve"])) > settings["maximum_patch_curve_difference"]
            or acquisition == "different"):
        return "Material scanner calibration/processing mismatch"
    return None


def measure_session(session, *, method=None):
    settings = method_settings(method)
    result = {"method": settings, "experiments": [], "captures": {}, "acquisition_consistency": "unknown"}
    try:
        captures = [accepted_capture(session, n) for n in (1, 3)]
        result["captures"] = {k: c["provenance"] for k, c in zip(("before", "after"), captures)}
        acquisition = [c["provenance"]["acquisition_settings"] for c in captures]
        result["acquisition_consistency"] = "confirmed consistent" if acquisition[0] and acquisition[0] == acquisition[1] else "different" if all(acquisition) else "unknown"
        pages = [aligned_capture(c) for c in captures]
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        result["reason"] = evidence_reason(error)
        result["experiments"] = [{"experiment_id": e["experiment_id"], "before": None, "after": None,
                                  "reason": evidence_reason(error)} for e in session["selected_experiments"]]
        return result
    for experiment in session["selected_experiments"]:
        record = {"experiment_id": experiment["experiment_id"]}
        try:
            candidates = [candidate_evidence(session, c, experiment, n) for c, n in zip(captures, (1, 3))]
            selected_pages = [p[c["page_number"]] for p, c in zip(pages, candidates)]
            for label, candidate, page in zip(("before", "after"), candidates, selected_pages):
                record[label + "_evidence"] = {k: deepcopy(candidate[k]) for k in ("candidate_id", "mapping", "page_number", "source", "limits", "marks", "plot")}
                record[label + "_evidence"].update(calibration=page["calibration"], dpi=page["dpi"],
                    fit_error_mm=page["fit_error_mm"], bilevel=page["bilevel"], noise_floor=page["noise_floor"])
            stats = [cell_statistics(p, c) for p, c in zip(selected_pages, candidates)]
            masks = [scan._cell_masks(c["plot"]["box"], c["data"].shape, c["marks"])[0] for c in candidates]
            mask = masks[0] & masks[1]
            record.update(compare_pairs(candidates[0]["data"], experiment["limits"], *stats,
                                        [c["mapping"] for c in candidates], mask=mask, method=settings))
            record["mask"] = mask.tolist()
            record["cell_statistics"] = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()} for s in stats]
            mismatch = comparability_reason(selected_pages, result["acquisition_consistency"], settings)
            if mismatch:
                record["diagnostic_comparison"] = {k: record.get(k) for k in ("before", "after", "detail_change_pp", "cnr_change")}
                record.update(before=None, after=None, detail_change_pp=None, cnr_change=None,
                              cnr_relative_change_percent=None, reason=mismatch)
            record["measurement_note"] = "Separate scanner captures; acquisition settings " + result["acquisition_consistency"] + "." + (" Bilevel thresholding/halftoning limits tone evidence." if any(p["bilevel"] for p in selected_pages) else "")
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            record.update(before=None, after=None, reason=evidence_reason(error))
        result["experiments"].append(record)
    return result
