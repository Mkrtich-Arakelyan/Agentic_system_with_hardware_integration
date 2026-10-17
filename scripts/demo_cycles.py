"""Generate a two-cycle synthetic demonstration; no models, experiments or printers.

Run from the repository root: python -m scripts.demo_cycles --out output/cycle_demo
"""
import argparse
from pathlib import Path
import json
import numpy as np
from app.workflow import tuning
from app.print_tuning.tuning_tools import create_tuning_tools
from app.print_tuning.fake_scan import fake_scan
from app.print_tuning.schemas import GammaProposalResponse, GammaReviewResponse, GrayLevelsReviewResponse, StepsReviewResponse
from app.workflow.report_document import build_report


def winner(e=None, candidate="reference"):
    result = {"candidate_id": candidate, "reason": "Synthetic demonstration choice", "confidence": "low", "tie_explanation": None}
    return result | ({"experiment_id": e} if e else {})


def generate(out):
    out = Path(out).resolve()
    if out.exists(): raise ValueError("Choose a new output directory; no existing artifact is overwritten")
    root = out / "synthetic_run"; root.mkdir(parents=True)
    results, selections = [], []
    ids = [f"synthetic_{i}" for i in range(1, 5)]
    for i, eid in enumerate(ids):
        folder = root / eid; folder.mkdir()
        paths = {"results_json": str(folder / "results.json")}
        y, x = np.indices((12, 12))
        for stage, amplitude in (("initial", .3), ("final", 1 + i * .15)):
            data = .15 + .03 * x + .02 * y + amplitude * np.exp(-((x - 6)**2 + (y - 6)**2) / 5)
            path = folder / (stage + ".npy"); np.save(path, data)
            paths[stage + "_intensity"] = str(path)
            paths[stage + "_camera"] = str(folder / (stage + ".png"))
        result = {"ok": True, "experiment_id": eid, "settings": {"iterations": 100 * (i + 1), "focus_size": 1},
            "initial_enhancement": 1., "final_enhancement": 2. + i * .4, "paths": paths,
            "provenance": "Synthetic fixture; no scientific experiment or physical printer gain measured"}
        (folder / "results.json").write_text(json.dumps(result))
        results.append(result)
        selections.append({"experiment_id": eid, "validity": "Synthetic fixture", "comparability": "Synthetic controls",
            "relevance": "Demo only", "information_gained": "Layout and workflow demonstration", "paths": paths})
    (root / "config.json").write_text(json.dumps({"bench": {"target_position": [6, 6]}}))
    tuning.initialize_tuning_session(root, {"selections": selections, "group_reason": "Clearly labeled synthetic fixtures"}, results, max_cycles=2)
    path = root / "tuning_session.json"
    make, read = create_tuning_tools(root)
    for cycle in (1, 2):
        session = tuning.load_tuning_session(path)
        proposals = []
        for eid in ids:
            gamma = session["accepted_settings"][eid]["gamma"]
            choices = [g for g in (.4, .6, .8, 1.2, 1.5, 2., 2.5) if g != gamma][:5]
            proposals.append({"experiment_id": eid, "tests": [{"candidate_id": f"test_{i+1}", "gamma": g,
                "reason": "Synthetic tone-curve comparison"} for i, g in enumerate(choices)]})
        tuning.accept_specialist_response(path, GammaProposalResponse(stage="propose_gamma", experiments=proposals), tuning.build_tuning_context(path, "propose_gamma"))
        for step, stage in enumerate(("review_gamma", "review_gray_levels", "review_steps"), 1):
            trial = make.invoke({"round_id": step}); assert trial["status"] == "ok", trial
            tuning.record_trial_result(path, trial)
            scans = fake_scan(trial["pdf_path"], root / "scans" / f"cycle_{cycle}" / f"step_{step}", noise=.004, seed=cycle*10+step)
            evaluated = read.invoke({"scan_paths": scans}); assert evaluated["status"] == "ok", evaluated
            tuning.record_scan_for_executor(path, evaluated)
            common = {"stage": stage, "assessment": "usable", "issues": []}
            if step == 1:
                current = tuning.load_tuning_session(path)["accepted_settings"][ids[0]]
                choices = [(d, l) for d, l in ((.01, 1.), (.03, .98), (.05, .95), (.07, .92), (.09, .9), (.11, .88))
                           if (d, l) != (current["dark_level"], current["light_level"])][:5]
                response = GammaReviewResponse(**common, gamma_winners=[winner(eid, "test_2") for eid in ids],
                    endpoint_tests=[{"candidate_id": f"test_{i+1}", "dark_level": d, "light_level": l, "reason": "Synthetic shared endpoints"} for i, (d, l) in enumerate(choices)])
            elif step == 2:
                current = tuning.load_tuning_session(path)["accepted_settings"]
                response = GrayLevelsReviewResponse(**common, endpoint_winner=winner(candidate="test_2"), step_tests=[
                    {"experiment_id": eid, "tests": [{"candidate_id": f"test_{i+1}", "steps": n, "reason": "Synthetic quantization comparison"}
                    for i, n in enumerate([n for n in (0, 2, 4, 8, 12, 16, 24, 32) if n != current[eid]["steps"]][:5])]} for eid in ids])
            else:
                response = StepsReviewResponse(**common, step_winners=[winner(eid, "test_4") for eid in ids])
            tuning.accept_specialist_response(path, response, tuning.build_tuning_context(path, stage), evaluation_record_id=evaluated["evaluation_path"])
        tuning.decide_cycle(path, "continue" if cycle == 1 else "finish")
    return build_report(root, out / "report", findings={"title": "Synthetic two-cycle tuning demonstration",
        "question": "How do these synthetic fixtures compare across iteration labels?"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", type=Path, required=True)
    print(generate(parser.parse_args().out)["report_path"])
