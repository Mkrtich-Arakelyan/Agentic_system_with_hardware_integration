"""Full offline HTTP → actual agents/graph/tools/journals, with scripted model replies."""
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from app.api.main import create_app
from app.api.session_manager import prepare
from app.print_tuning.fake_scan import fake_scan
from app.printing import connector
from app.workflow import runner
from tests.api.test_sessions import wait, send


class OfflineModel:
    profile = {"image_inputs": True}
    model_name = "offline"

    def __init__(self, root):
        self.root = root
        self.experiments = 0
        self.planned = False

    def invoke(self, messages):
        return AIMessage(content="Offline evidence summary")

    def results(self):
        return [json.loads(p.read_text()) for p in sorted(self.root.glob("experiment_*/results.json"))]

    def with_structured_output(self, schema, **kwargs):
        def invoke(messages):
            if schema.__name__ == "RoutingDecision":
                role = "scientist" if not self.planned or self.experiments >= 4 else "executor"
                return schema(next_role=role, instruction="Compare macropixel sizes with fixed controls", reason="Offline plan")
            if schema.__name__ == "ScientistResponse":
                self.planned = True
                results = self.results()
                ready = len(results) >= 4
                return schema(status="ready" if ready else "needs_more_experiments", analysis="Offline comparison plan",
                    selections=[dict(experiment_id=r["experiment_id"], validity="Recorded success", comparability="Fixed controls",
                                     relevance="Macropixel comparison", information_gained="Different spatial control") for r in results] if ready else [],
                    group_reason="Four controlled settings" if ready else "",
                    proposed_experiments=[] if ready else [dict(macropixel_size=m, focus_size=1, background_size=3,
                        iterations=2, search_seed=0, reason="Compare grid") for m in (1, 2, 4, 8)])
            session = json.loads((self.root / "tuning_session.json").read_text())
            ids = list(session["accepted_settings"])
            winner = lambda eid=None: dict(candidate_id="reference", reason="Retain reference for offline test", confidence="low",
                                          tie_explanation=None, **({"experiment_id": eid} if eid else {}))
            stage = schema.model_fields["stage"].annotation.__args__[0]
            if stage == "propose_gamma":
                value = dict(stage=stage, experiments=[dict(experiment_id=e, tests=[dict(candidate_id=f"test_{i}", gamma=g,
                    reason="Offline gamma test") for i, g in enumerate((.4, .6, 1.5, 2.2, 3.), 1)]) for e in ids])
            elif stage == "review_gamma":
                value = dict(stage=stage, assessment="usable", issues=[], gamma_winners=[winner(e) for e in ids],
                    endpoint_tests=[dict(candidate_id=f"test_{i}", dark_level=d, light_level=l, reason="Endpoint test")
                    for i, (d, l) in enumerate(((.02, 1.), (.04, 1.), (.06, .95), (.08, .9), (.1, .85)), 1)])
            elif stage == "review_gray_levels":
                value = dict(stage=stage, assessment="usable", issues=[], endpoint_winner=winner(), step_tests=[dict(experiment_id=e,
                    tests=[dict(candidate_id=f"test_{i}", steps=n, reason="Step test") for i, n in enumerate((2, 4, 8, 16, 32), 1)]) for e in ids])
            else:
                value = dict(stage=stage, assessment="usable", issues=[], step_winners=[winner(e) for e in ids])
            parsed = schema.model_validate(value)
            return {"parsed": parsed, "raw": None, "parsing_error": None} if kwargs.get("include_raw") else parsed
        return SimpleNamespace(invoke=invoke)

    def bind_tools(self, tools, **options):
        def invoke(messages):
            if options.get("tool_choice") == "none":
                return AIMessage(content="Offline tool result summary")
            names = {t.name for t in tools}
            if names == {"run_focusing_experiment"}:
                name = "run_focusing_experiment"
                args = dict(macropixel_size=(1, 2, 4, 8)[self.experiments], focus_size=1, background_size=3,
                            iterations=2, search_seed=0)
                self.experiments += 1
            elif str(messages[1].content).startswith("Tuning task:"):
                context = json.loads(messages[1].content.split("\n", 1)[1])
                name = context["requested_action"]
                args = {"make_trial_sheet": {"round_id": context["round_id"]},
                        "read_scan": {"scan_paths": context["scan_paths"]},
                        "printer_status": {"printer": context["printer"]},
                        "job_status": {"job_id": context["job_id"]},
                        "submit_trial_print": {"pdf_path": (context.get("trial") or {}).get("pdf_path"),
                                               "printer": context["printer"]}}[name]
            else:
                text = next(m.content for m in messages if isinstance(m.content, str) and "Report-and-print context:\n" in m.content)
                context = json.loads(text.split("Report-and-print context:\n", 1)[1])
                if context.get("report_revision_pending") or not (self.root / "report.pdf").exists():
                    name, args = "build_final_report", {}
                    if context.get("report_revision_pending"):
                        args["findings"] = {"closing_note": "Revised offline integration report."}
                elif context.get("active_job_id"):
                    name, args = "job_status", {"job_id": context["active_job_id"]}
                else:
                    name, args = "submit_print", {"pdf_path": str(self.root / "report.pdf"), "printer": "Office_Printer", "copies": 1}
            return AIMessage(content="", tool_calls=[{"id": "offline", "name": name, "args": args}])
        return SimpleNamespace(invoke=invoke)


def test_full_two_cycle_workflow_revision_and_fake_print(tmp_path, monkeypatch):
    import langchain_openai
    import app.printing.tools
    from app.printing import mcp_server
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setattr(connector, "_fake", lambda: True)
    monkeypatch.setattr(connector, "FAKE_JOBS_FILE", tmp_path / "fake_jobs.json")
    monkeypatch.setattr(connector, "FAKE_QUEUE_SECONDS", 0)
    monkeypatch.setattr(connector, "FAKE_PRINT_SECONDS", 0)
    tools = [StructuredTool.from_function(getattr(mcp_server, name), name=name) for name in
             ("list_printers", "printer_status", "job_status", "submit_trial_print", "submit_print", "cancel_job")]
    monkeypatch.setattr(app.printing.tools, "load_printer_tools", lambda env=None: tools)

    def offline_prepare(root, state, options, **kwargs):
        monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kw: OfflineModel(root))
        return prepare(root, state, options, **kwargs)

    with TestClient(create_app(tmp_path / "runs", prepare_fn=offline_prepare)) as client:
        sid = client.post("/api/sessions", json={"request": "Offline comparison", "fake_printer": True,
                                               "max_tuning_cycles": 2, "initial_decisions": 2}).json()["session_id"]
        scans, approvals, cycles, revised = [], [], [], False
        initial_hash = None
        for _ in range(35):
            session = wait(client, sid, timeout=180)
            p = session["pending"]
            kind = p["kind"]
            if kind == "budget":
                answer = "yes"
            elif kind == "tuning_printer":
                answer = {"action": "printer", "printer": "Office_Printer"}
            elif kind == "trial_print_approval":
                approvals.append((p["cycle_id"], p["step_id"]))
                answer = {"action": "approve"}
            elif kind == "tuning_scan":
                scans.append((p["cycle_id"], p["step_id"]))
                root = Path(session["run_dir"])
                saved = json.loads((root / "tuning_session.json").read_text())
                trial = saved["rounds"][str(p["step_id"])]["trial_result"]
                synthetic = fake_scan(trial["pdf_path"], tmp_path / f"capture_{p['cycle_id']}_{p['step_id']}", noise=.004, seed=42)
                upload = client.post(f"/api/sessions/{sid}/scans", data={"interrupt_token": p["interrupt_token"]},
                    files=[("files", (Path(f).name, Path(f).read_bytes(), "image/png")) for f in synthetic])
                assert upload.status_code == 200, upload.text
                answer = {"action": "scan", "scan_paths": upload.json()["paths"]}
            elif kind == "tuning_cycle":
                cycles.append(p["cycle_id"])
                answer = {"action": "continue" if p["cycle_id"] == 1 else "finish"}
            elif kind == "print_approval":
                if not revised:
                    initial_hash, revised = p["pdf_sha256"], True
                    answer = "Add a closing note saying Revised offline integration report."
                else:
                    assert p["pdf_sha256"] != initial_hash
                    assert list((Path(session["run_dir"]) / "report_versions").glob("*.pdf"))
                    assert send(client, sid, p, "yes").status_code == 200
                    break
            elif kind == "tuning_poll":
                answer = {"action": "check"}
            else:
                raise AssertionError(p)
            response = send(client, sid, p, answer)
            assert response.status_code == 200, response.text
        else:
            raise AssertionError("Did not reach report printing")
        final = wait(client, sid, "finished", timeout=180)
        assert len(scans) == len(approvals) == 6 and cycles == [1, 2]
        assert len([r for r in final["execution_results"] if r["tool_name"] == "run_focusing_experiment"]) == 4
        journal = json.loads((Path(final["run_dir"]) / "report_print.json").read_text())
        assert len(journal["attempts"]) == 1
        assert journal["attempts"][0]["readings"][-1]["state"] == "done"
        assert len(journal["revisions"]) == 1
