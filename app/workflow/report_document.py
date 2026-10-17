"""Reusable two-page report, deterministic offline regeneration, executor tool.

No model, experiment, printer or journal mutation occurs in the offline builder.
The executor tool alone installs a new report version in the active run.
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pymupdf as fitz

from app.report_policy import REPORT_VERSION
from app.print_tuning.final_figures import render_report_comparison
from app.print_tuning.readability import METHOD, measure_session, method_settings

MM = 72 / 25.4
GREEN = (.12, .28, .20)
MARGIN = 15 * MM
WIDTH, HEIGHT = 210 * MM, 297 * MM


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dependency_fingerprint(session_path, session, method, findings):
    """Hash recorded dependencies, including absent files; no recency heuristics."""
    paths = {str(Path(session_path).resolve())}
    from app.print_tuning import readability, final_figures
    from app import report_policy
    paths.update(str(Path(p).resolve()) for p in (__file__, readability.__file__, final_figures.__file__, report_policy.__file__))
    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and (key.endswith("path") or key in ("path", "results_json")):
                    paths.add(value)
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
    walk(session)
    # Manifest content determines plot geometry and may name additional sources.
    for saved in list(paths):
        path = Path(saved)
        if path.suffix == ".json" and path.is_file():
            walk(json.loads(path.read_text(encoding="utf-8")))
    dependencies = {p: digest(p) if Path(p).is_file() else None for p in sorted(paths)}
    payload = {"version": REPORT_VERSION, "method": method, "findings": findings, "dependencies": dependencies}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest(), dependencies


def scientific_inputs(session):
    selected = session["selected_experiments"]
    rows = []
    for index, item in enumerate(selected):
        source = item["source"]
        row = {"label": f"E{index + 1}", "experiment_id": item["experiment_id"], **source}
        if source.get("results_json") and Path(source["results_json"]).is_file():
            recorded = json.loads(Path(source["results_json"]).read_text(encoding="utf-8"))
            for key in ("experiment_id", "settings", "initial_enhancement", "final_enhancement"):
                if recorded.get(key) != row.get(key):
                    raise ValueError(f"Scientific result differs from frozen selection: {key}")
            row["recorded_result"] = recorded
            row["results_sha256"] = digest(source["results_json"])
            for key in ("wall_time_seconds", "camera_shots"):
                if key in recorded:
                    row[key] = recorded[key]
        rows.append(row)
    keys = set.intersection(*(set(r["settings"]) for r in rows))
    varied = sorted(k for k in keys if any(r["settings"][k] != rows[0]["settings"][k] for r in rows[1:]))
    fixed = {k: rows[0]["settings"][k] for k in sorted(keys) if k not in varied}
    return {"rows": rows, "varied": varied, "fixed": fixed, "chart_type": "categorical_grouped_bars"}


def _number(value, digits=3):
    return f"{value:.{digits}g}" if isinstance(value, (float, int)) and math.isfinite(value) else "not recorded"


def default_findings(science):
    rows = science["rows"]
    best, worst = max(rows, key=lambda r: r["final_enhancement"]), min(rows, key=lambda r: r["final_enhancement"])
    varied = science["varied"]
    values = [r["final_enhancement"] for r in rows]
    improved = sum(r["final_enhancement"] > r["initial_enhancement"] for r in rows)
    trend = "decreases" if all(a > b for a, b in zip(values, values[1:])) else "increases" if all(a < b for a, b in zip(values, values[1:])) else "does not change monotonically"
    interpretation = (
        f"{best['label']} has the largest final enhancement ({_number(best['final_enhancement'])}), compared with "
        f"{_number(worst['final_enhancement'])} for {worst['label']}; their absolute difference is "
        f"{_number(best['final_enhancement'] - worst['final_enhancement'])}. Final enhancement {trend} in the frozen E1-E4 order. "
        "The bars show separate recorded experiments; they do not interpolate between settings. "
        "The comparison supports a ranking of these measured outcomes under the recorded conditions. "
        f"{improved} of the four selected experiments finish above their own initial enhancement. "
        "Initial and final values describe the scientific focusing experiment. "
        "The print-rendering adjustments on the next page change its presentation and leave scientific enhancement unchanged.")
    conditions = "; ".join(f"{k.replace('_', ' ')}={v}" for k, v in science["fixed"].items()
                            if k in ("focus_size", "background_size", "iterations", "search_seed", "shots_per_measurement", "validation_shots"))
    return {"title": "Scientific experiment results",
        "question": "How do the selected settings compare in measured focusing enhancement?",
        "setup": "Varied: " + (", ".join(k.replace("_", " ") for k in varied) or "not recorded") + ". " +
                 (conditions + ". " if conditions else "Fixed conditions: not recorded. ") +
                 "Enhancement is focus mean divided by the mean of the four background-region means.",
        "interpretation": interpretation,
        "limitation_next": "One recorded realization per selected setting does not quantify repeatability." +
            (" Multiple conditions vary, so their individual effects cannot be isolated." if len(varied) > 1 else "") +
            " Repeat the selected settings with independent search/noise seeds, holding the measurement regions and budget fixed, to estimate variability."}


class Page:
    """Flowing serif text and measured tables; reject overflow instead of shrinking."""
    def __init__(self, document, number):
        self.page = document.new_page(width=WIDTH, height=HEIGHT)
        self.y = MARGIN
        self.bounds = []
        self.page.insert_text((WIDTH / 2 - 3, HEIGHT - 23), str(number), fontsize=10, fontname="tiro")

    def text(self, text, size=11, *, color=(0, 0, 0), gap=5, width=None, x=None):
        x = MARGIN if x is None else x
        width = WIDTH - MARGIN - x if width is None else width
        font = fitz.Font("tiro")
        lines = []
        for paragraph in text.split("\n"):
            line = ""
            for word in paragraph.split():
                test = (line + " " + word).strip()
                if font.text_length(test, fontsize=size) > width:
                    if not line:
                        raise ValueError("Unbreakable report text exceeds column width")
                    lines.append(line)
                    line = word
                else:
                    line = test
            lines.append(line)
        line_height = size * 1.18
        bottom = self.y + len(lines) * line_height
        if bottom > HEIGHT - MARGIN:
            raise ValueError("Report text overflows page; shorten the structured findings")
        for i, line in enumerate(lines):
            self.page.insert_text((x, self.y + size + i * line_height), line, fontname="tiro", fontsize=size, color=color)
        self.bounds.append([x, self.y, x + width, bottom])
        self.y = bottom + gap

    def heading(self, text):
        self.text(text, 14, color=GREEN, gap=7)

    def table(self, headers, rows, widths, *, row_height=18):
        x, top = MARGIN, self.y
        total = sum(widths)
        if total > WIDTH - 2 * MARGIN + .1 or top + row_height * (len(rows) + 1) > HEIGHT - MARGIN:
            raise ValueError("Report table exceeds page margins")
        for index, row in enumerate([headers, *rows]):
            x = MARGIN
            for value, width in zip(row, widths):
                text = str(value)
                if fitz.Font("tiro").text_length(text, fontsize=10) > width - 6:
                    raise ValueError(f"Table cell exceeds its column: {text}")
                self.page.insert_text((x + 3, top + index * row_height + 12), text, fontname="tiro", fontsize=10,
                                      color=GREEN if index == 0 else (0, 0, 0))
                x += width
            if index == 0 or index == len(rows):
                yy = top + (index + 1) * row_height
                self.page.draw_line((MARGIN, yy), (MARGIN + total, yy), color=GREEN, width=.4)
        self.bounds.append([MARGIN, top, MARGIN + total, top + row_height * (len(rows) + 1)])
        self.y = top + row_height * (len(rows) + 1) + 7


def _science_chart(page, rows, height=215):
    top = page.y + 7
    left, right = MARGIN + 40, WIDTH - MARGIN - 5
    maximum = max(r["final_enhancement"] for r in rows) * 1.15
    maximum = max(maximum, max(r["initial_enhancement"] for r in rows) * 1.15, 1)
    baseline = top + height - 25
    page.page.draw_line((left, baseline), (right, baseline), width=.5)
    for i, row in enumerate(rows):
        centre = left + (right - left) * (i + .5) / 4
        for offset, key, color in ((-12, "initial_enhancement", (.7, .7, .7)), (5, "final_enhancement", GREEN)):
            barheight = row[key] / maximum * (height - 50)
            page.page.draw_rect(fitz.Rect(centre + offset, baseline - barheight, centre + offset + 13, baseline), color=None, fill=color)
        page.page.insert_text((centre - 8, baseline + 15), row["label"], fontsize=11, fontname="tiro")
    for value in (0, maximum / 2, maximum):
        yy = baseline - value / maximum * (height - 50)
        page.page.insert_text((MARGIN, yy + 3), _number(value), fontsize=10, fontname="tiro")
    page.y = top + height
    page.text("Enhancement (dimensionless). Gray: scientific Initial; green: scientific Final.", 10)


def render_document(inputs, figure, target):
    science, findings, metrics = inputs["science"], inputs["findings"], inputs["metrics"]
    with fitz.open() as doc:
        p = Page(doc, 1)
        p.text(findings["title"], 21, color=GREEN, gap=10)
        p.text(findings["question"])
        p.heading("Experimental comparison")
        p.text(findings["setup"])
        _science_chart(p, science["rows"], inputs.get("presentation", {}).get("chart_height_pt", 215))
        varied = science["varied"]
        # Short parameter aliases preserve readable numeric tables. Full keys are saved in JSON.
        aliases = {"macropixel_size": "macropixel", "iterations": "iterations", "search_seed": "seed"}
        settings = [", ".join(f"{aliases.get(k, k)}={r['settings'][k]}" for k in varied) or "not recorded" for r in science["rows"]]
        cost_key = next((k for k in ("wall_time_seconds", "camera_shots") if all(k in r for r in science["rows"])), None)
        headers = ["ID", "Varied setting", "Initial", "Final"] + (["Time (s)" if cost_key == "wall_time_seconds" else "Shots"] if cost_key else [])
        rows = [[r["label"], setting, _number(r["initial_enhancement"]), _number(r["final_enhancement"])] +
                ([_number(r[cost_key])] if cost_key else []) for r, setting in zip(science["rows"], settings)]
        p.table(headers, rows, [30, 240, 80, 80, 80] if cost_key else [30, 280, 100, 100])
        p.heading("Evidence and interpretation")
        p.text(findings["interpretation"])
        p.heading("Limitation and next experiment")
        p.text(findings["limitation_next"])
        if findings.get("closing_note"):
            p.text(findings["closing_note"])

        p = Page(doc, 2)
        p.text("Hardware loop results", 21, color=GREEN, gap=7)
        p.text("Same Final camera arrays; starting and accepted tuned print mappings.", 10, gap=5)
        label_y = p.y
        for column, label in enumerate(("Starting print settings", "Tuned print settings")):
            x = MARGIN + (51 + column * 83) * MM - fitz.Font("tiro").text_length(label, fontsize=11) / 2
            p.page.insert_text((x, label_y + 10), label, fontsize=11, fontname="tiro")
        p.y += 15
        rect = fitz.Rect(MARGIN, p.y, WIDTH - MARGIN, p.y + 144 * MM)
        p.page.insert_image(rect, filename=figure["figure_path"], keep_proportion=True)
        for i in range(4):
            p.page.insert_text((MARGIN + 9, p.y + (i * 36 + 19) * MM), f"E{i + 1}", fontsize=11, fontname="tiro")
        p.y = rect.y1 + 3
        p.text("Original numerical data, frozen limits; independent column contrast is disabled.", 10, gap=3)
        mappings = []
        for row in figure["experiments"]:
            a, b = row["starting_mapping"], row["mapping"]
            arrow = lambda k: f"{_number(a[k])} to {_number(b[k])}"
            mappings.append([row["label"], arrow("gamma"), arrow("dark_level"), arrow("light_level"), arrow("steps")])
        p.table(["ID", "Gamma", "Dark", "Light", "Steps (0=smooth)"], mappings, [30, 105, 105, 105, 165], row_height=14)
        metric_rows = []
        for i, row in enumerate(metrics["experiments"]):
            a, b = row.get("before"), row.get("after")
            if a and b:
                metric_rows.append([f"E{i + 1}", f"{a['detail_percent']:.1f} to {b['detail_percent']:.1f}",
                    f"{row['detail_change_pp']:+.1f}", f"{a['median_cnr']:.2f} to {b['median_cnr']:.2f}", f"{row['cnr_change']:+.2f}"])
            else:
                metric_rows.append([f"E{i + 1}", "not available", "-", "not available", "-"])
        p.table(["ID", "Detail %: start to tuned", "pp", "Median CNR: start to tuned", "Delta"], metric_rows,
                [25, 145, 40, 230, 70], row_height=14)
        usable = [r for r in metrics["experiments"] if r.get("before") and r.get("after")]
        if usable:
            groups = {"increased": [], "decreased": [], "unchanged": [], "a trade-off": []}
            for index, record in enumerate(metrics["experiments"]):
                if record not in usable:
                    continue
                d, c = record["detail_change_pp"], record["cnr_change"]
                key = "increased" if d > 0 and c > 0 else "decreased" if d < 0 and c < 0 else "unchanged" if d == c == 0 else "a trade-off"
                groups[key].append(f"E{index + 1}")
            conclusion = "Detail/CNR: " + "; ".join(f"{', '.join(ids)} {key}" for key, ids in groups.items() if ids) + "."
        else:
            reasons = list(dict.fromkeys(r.get("reason", "Evidence missing") for r in metrics["experiments"]))
            conclusion = "Measurements not available: " + "; ".join(reasons) + "."
        notes = [r.get("measurement_note") for r in metrics["experiments"] if r.get("measurement_note")]
        note = notes[0] if notes else "Separate scanner captures; acquisition settings unknown."
        provenance = [str(c.get("scan_provenance", "unknown")) for c in metrics.get("captures", {}).values()]
        if any("synthetic" in p.lower() for p in provenance):
            note += " Synthetic scan evidence; no physical printer gains measured."
        else:
            note += " Scan origin " + ("explicitly recorded" if provenance and all(p != "unknown" for p in provenance) else "not recorded") + "."
        p.text(conclusion, 11, gap=3)
        coverage = ", ".join(f"E{i + 1} {r.get('jointly_valid_pairs', 0)}/{r.get('eligible_pairs', 0)}" for i, r in enumerate(metrics["experiments"]))
        p.text(note + " Joint/eligible pairs: " + coverage + ". Detail: correct ordering and CNR >= 3; engineering criterion, not a validated visibility score. Acquisition uncertainty prevents a causal printer-gain claim.", 10, gap=0)
        layout = [page.rect for page in doc]
        target = Path(target)
        doc.save(target)
    return {"pages": len(layout), "body_font_pt": 11, "minimum_table_caption_pt": 10}


def build_report(run_dir, output_dir, *, findings=None, method=None, figure=None, revision=None, presentation=None):
    """Read a completed run and write a separate reproducible report bundle."""
    root, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Offline output must be outside the run; original reports and history are read-only")
    session_path = root / "tuning_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if session.get("status") != "completed" or session.get("phase") != "completed" or not session.get("final_settings"):
        raise ValueError("Report requires completed tuning; do not restart tuning to regenerate a report")
    settings = method_settings(method)
    science = scientific_inputs(session)
    text = default_findings(science) | (findings or {})
    presentation = presentation or {}
    if set(presentation) - {"chart_height_pt"} or not 180 <= presentation.get("chart_height_pt", 215) <= 240:
        raise ValueError("Supported layout: chart_height_pt between 180 and 240; required font sizes/content stay fixed")
    fingerprint, dependencies = dependency_fingerprint(session_path, session, settings,
        {"findings": text, "revision": revision, "presentation": presentation})
    out.mkdir(parents=True, exist_ok=True)
    data_path, pdf = out / "report_inputs_v2.json", out / "report.pdf"
    if data_path.is_file() and pdf.is_file():
        cached = json.loads(data_path.read_text(encoding="utf-8"))
        if (cached.get("fingerprint") == fingerprint and cached.get("pdf_sha256") == digest(pdf)
                and Path(cached["figure"]["figure_path"]).is_file()
                and digest(cached["figure"]["figure_path"]) == cached["figure"]["figure_sha256"]
                and (figure is None or figure["figure_sha256"] == cached["figure"]["figure_sha256"])):
            return cached
        raise ValueError("Output bundle exists with different inputs; choose a new output directory")
    if pdf.exists() or data_path.exists():
        raise ValueError("Partial report bundle exists; choose a new output directory")
    comparison = figure or render_report_comparison(session_path, out)
    comparison = {**comparison, "inputs_sha256": fingerprint}
    inputs = {"schema_version": REPORT_VERSION, "fingerprint": fingerprint, "dependencies": dependencies,
              "science": science, "findings": text, "metrics": measure_session(session, method=settings),
              "figure": comparison, "report_path": str(pdf), "source_session": str(session_path)}
    inputs.update(revision=revision, presentation=presentation)
    inputs["layout"] = render_document(inputs, comparison, pdf)
    from app.workflow.reporting import inspect_tuned_report, require_report
    validation = inspect_tuned_report(pdf, comparison)
    require_report(validation)
    inputs["pdf_sha256"] = digest(pdf)
    inputs = safe_json(inputs)
    data_path.write_text(json.dumps(inputs, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return inputs


def current_report_inputs(root):
    """Use hash-bound inputs, including legacy builder paths recorded in state."""
    root = Path(root)
    candidates = [root / "report_inputs_v2.json"]
    state_path = root / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for record in reversed(state.get("execution_results", [])):
            result = record.get("result") or {}
            if record.get("tool_name") == "build_final_report" and result.get("report_inputs_path"):
                candidates.append(Path(result["report_inputs_path"]))
    target = root / "report.pdf"
    for path in candidates:
        if path.is_file() and target.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("pdf_sha256") == digest(target):
                return saved
    return {}


def create_report_tool(run_dir):
    from langchain_core.tools import tool

    @tool
    def build_final_report(findings: dict | None = None, presentation: dict | None = None,
                           revision_instructions: str = "", revision_id: str = "") -> dict:
        """Build the two-page final report with fixed layout and offline scan metrics.

        findings may override title, question, setup, interpretation, limitation_next,
        closing_note. For a revision, translate the exact pending instructions into
        concrete findings edits and/or presentation={"chart_height_pt":180..240}.
        Omitted fields retain the previous version. No-op edits fail; unsupported
        requests need clarification rather than printing the old PDF. The workflow
        supplies revision_instructions/revision_id; never invent them.
        Use concise, evidence-supported text. Never change results or mappings.
        This does not print. Use submit_print afterward to request human approval.
        """
        from app.workflow.reporting import load_record, save_record, print_state, complete_revision, inspect_tuned_report
        root = Path(run_dir).resolve()
        journal = load_record(root)
        if print_state(journal) in ("pending", "uncertain", "failed", "stopped"):
            return {"ok": False, "error": "Resolve the existing report submission before rebuilding"}
        figure = journal.get("figure")
        if not figure:
            return {"ok": False, "error": "Application must prepare the final comparison figure first"}
        target = root / "report.pdf"
        revision = journal.get("revision")
        if revision and (revision_id != revision["id"] or revision_instructions != revision["instructions"]):
            return {"ok": False, "error": "Exact pending revision instructions and ID are required"}
        if not revision and (revision_id or revision_instructions):
            return {"ok": False, "error": "No matching pending revision exists"}
        saved_inputs = root / "report_inputs_v2.json"
        previous = current_report_inputs(root)
        baseline = previous.get("findings") or default_findings(scientific_inputs(
            json.loads((root / "tuning_session.json").read_text(encoding="utf-8"))))
        if findings and (set(findings) - set(baseline) - {"closing_note"}
                         or any(not isinstance(value, str) for value in findings.values())):
            return {"ok": False, "error": "Unsupported findings fields or non-text edits"}
        text = baseline | (findings or {})
        baseline_layout = {"chart_height_pt": 215} | previous.get("presentation", {})
        layout = baseline_layout | (presentation or {})
        changed_fields = [key for key in text if text[key] != baseline.get(key, "")]
        if revision and not changed_fields and layout == baseline_layout:
            return {"ok": False, "error": "No content change was produced; revision remains pending. Supply concrete text/layout edits and retry."}
        # Archive BEFORE attempting regeneration, including failed attempts.
        if target.exists():
            archive = root / "report_versions"
            archive.mkdir(exist_ok=True)
            previous = archive / (digest(target) + ".pdf")
            if not previous.exists():
                shutil.copy2(target, previous)
        out = root.parent.parent / "output" / "reports" / root.name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        try:
            bundle = build_report(root, out, findings=text, figure=figure, revision=revision, presentation=layout)
            checks = {"text_fields": changed_fields, "presentation": layout,
                      "semantic_validation": "Human review required; instruction compliance is not proven"}
            if revision and bundle["pdf_sha256"] == revision["before_sha256"]:
                raise ValueError("No content change was produced; PDF hash is unchanged. Correct the edits and retry.")
            with fitz.open(bundle["report_path"]) as document:
                extracted = " ".join(page.get_text() for page in document)
            normalized = lambda value: " ".join(value.split())
            if any(normalized(text[key]) not in normalized(extracted) for key in changed_fields):
                raise ValueError("Requested text edit was not found in the rendered PDF")
            if revision and target.exists():
                with fitz.open(target) as old:
                    old_text = " ".join(page.get_text() for page in old)
                if normalized(old_text) == normalized(extracted) and layout == baseline_layout:
                    raise ValueError("No content change was produced; only PDF bytes differ. Correct the edits and retry.")
            shutil.copy2(bundle["report_path"], target)
            shutil.copy2(out / "report_inputs_v2.json", saved_inputs)
            if revision:
                complete_revision(root, revision_id, inspect_tuned_report(target, figure), checks)
            elif journal.get("pending"):
                journal["pending"] = None
                save_record(root, journal)
        except Exception as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}", "revision_id": revision_id}
        return {"ok": True, "report_path": str(target), "report_inputs_path": str(out / "report_inputs_v2.json"),
                "pdf_sha256": bundle["pdf_sha256"], "pages": 2, "revision_id": revision_id,
                "revision_checks": checks, "review_note": "Review the revised PDF; semantic compliance requires human review."}

    return build_final_report


def main():
    parser = argparse.ArgumentParser(description="Build a report offline without experiments, models or printing")
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New output directory outside the run")
    parser.add_argument("--source-difference", type=float, default=.05)
    args = parser.parse_args()
    result = build_report(args.run, args.out, method={"source_difference": args.source_difference})
    print(result["report_path"])


if __name__ == "__main__":
    main()
