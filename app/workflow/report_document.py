"""Reusable 2/3-page report, deterministic offline regeneration, executor tool.

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

from app.report_policy import REPORT_VERSION, REPORT_PAGES
from app.print_tuning.final_figures import render_report_comparison
from app.print_tuning.readability import METHOD, measure_session, method_settings
from app.print_tuning.progress import measure_history, render_progress, describe_history

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
    from app.print_tuning import progress
    from app import report_policy
    paths.update(str(Path(p).resolve()) for p in (__file__, readability.__file__, progress.__file__, final_figures.__file__, report_policy.__file__))
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
        p.text("Experiment images", 21, color=GREEN, gap=7)
        p.text("Scientific Initial and Final arrays; both columns use the final accepted mapping.", 10)
        for column, label in enumerate(("Initial", "Final")):
            p.page.insert_text((MARGIN + (47 + column * 76) * MM, p.y + 10), label, fontsize=11, fontname="tiro")
        p.y += 15
        image_height = 178 * MM if inputs.get("presentation", {}).get("pages", 3) == 3 else 135 * MM
        rect = fitz.Rect(MARGIN, p.y, WIDTH - MARGIN, p.y + image_height)
        p.page.insert_image(rect, filename=figure["figure_path"], keep_proportion=True)
        for i in range(4):
            p.page.insert_text((MARGIN + 5, p.y + (i + .5) * image_height / 4), f"E{i + 1}", fontsize=11, fontname="tiro")
        p.y = rect.y1 + 5
        mappings = [[row["label"], *[_number(row["mapping"][key]) for key in ("gamma", "dark_level", "light_level", "steps")]]
                    for row in figure["experiments"]]
        p.table(["ID", "Final gamma", "Dark", "Light", "Steps (0=smooth)"], mappings, [30, 105, 105, 105, 165], row_height=14)
        p.text("Frozen normalization and annotations preserve scientific differences. Print tuning does not change scientific enhancement.", 10)

        if inputs.get("presentation", {}).get("pages", 3) == 3:
            p = Page(doc, 3)
        p.heading("Hardware loop results")
        import pymupdf
        chart_height = 162  # Do not shrink chart labels to squeeze a two-page request.
        for key in ("detail_percent", "median_cnr"):
            with pymupdf.open(inputs["charts"][key]["path"]) as chart:
                p.page.show_pdf_page(fitz.Rect(MARGIN, p.y, WIDTH - MARGIN, p.y + chart_height), chart, 0)
            p.y += chart_height + 5
        p.text("Baseline uses the original reference; Cn-S1 = gamma, S2 = shared gray endpoints, S3 = steps. Missing points break lines; each point comes from one accepted scan.", 10)
        summary = []
        for row in inputs["history"]["experiments"]:
            a, b = row["points"][0]["metrics"], row["points"][-1]["metrics"]
            if a and b:
                summary.append([row["label"], f"{a['detail_percent']:.1f} to {b['detail_percent']:.1f}",
                    f"{b['detail_percent']-a['detail_percent']:+.1f}", f"{a['median_cnr']:.2f} to {b['median_cnr']:.2f}",
                    f"{b['median_cnr']-a['median_cnr']:+.2f}"])
            else:
                summary.append([row["label"], "not available", "-", "not available", "-"])
        p.table(["ID", "Detail %: base to final", "pp", "Median CNR: base to final", "Delta"], summary,
                [25, 145, 40, 230, 70], row_height=16)
        p.text(describe_history(inputs["history"]), 11)
        coverage = "; ".join(row["label"] + " " + str(row["coverage"].get("jointly_valid_pairs", 0)) + "/" +
            str(row["coverage"].get("eligible_pairs", 0)) for row in inputs["history"]["experiments"])
        settings = inputs["history"].get("method", METHOD)
        p.text("Common valid/eligible pairs: " + coverage + f". Source difference >={settings['source_difference']:g}; correct ordering and CNR >={settings['cnr_threshold']:g} define preservation. These engineering thresholds are not validated human-vision cutoffs.", 10)
        captures = inputs["history"]["captures"].values()
        origin = "Synthetic evidence; no physical printer gains measured. " if any("synthetic" in str(c.get("scan_provenance", "")).lower() for c in captures) else "Scan origin is recorded in the saved inputs. "
        p.text(origin + "Separate captures; acquisition settings may be unknown. Normalization does not undo nonlinear scanner correction; detail and CNR summarize the same evidence.", 10)
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
    if all("synthetic" in str(row.get("recorded_result", {}).get("provenance", "")).lower() for row in science["rows"]):
        if "synthetic" not in text["title"].lower():
            text["title"] = "Synthetic experiment results"
        if "synthetic" not in text["setup"].lower():
            text["setup"] += " Synthetic numerical fixtures; no scientific experiment or physical printer gain measured."
    presentation = presentation or {}
    if set(presentation) - {"chart_height_pt", "pages"} or not 180 <= presentation.get("chart_height_pt", 215) <= 240:
        raise ValueError("Supported layout: chart_height_pt between 180 and 240; required font sizes/content stay fixed")
    if presentation.get("pages", 3) not in (2, 3):
        raise ValueError("Report page count must be 2 or 3")
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
              "science": science, "findings": text,
              "figure": comparison, "report_path": str(pdf), "source_session": str(session_path)}
    inputs.update(revision=revision, presentation=presentation)
    inputs["history"] = measure_history(session, method=settings)
    inputs["metrics"] = {"method": settings, "experiments": [{**row["coverage"], "experiment_id": row["experiment_id"],
        "before": row["points"][0]["metrics"], "after": row["points"][-1]["metrics"],
        "reason": row["points"][0].get("reason") or row["points"][-1].get("reason") or row["coverage"].get("reason")}
        for row in inputs["history"]["experiments"]]}
    inputs["charts"] = render_progress(inputs["history"], out)
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
        """Build the 2/3-page final report with fixed layout and offline scan metrics.

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
        baseline_layout = {"chart_height_pt": 215, "pages": 3} | previous.get("presentation", {})
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
                "pdf_sha256": bundle["pdf_sha256"], "pages": bundle["layout"]["pages"], "revision_id": revision_id,
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
