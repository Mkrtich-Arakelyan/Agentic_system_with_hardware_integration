"""Run Homework 3: hardware in the loop with an office printer."""

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from uuid import uuid4
from app.printing.settings import TRIAL_PRINT_REMINDER

BASE_DIR = Path(__file__).resolve().parents[2]
REQUIRED_DEPENDENCIES = {
    "dotenv": "python-dotenv",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "pydantic": "pydantic",
    "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai",
    "langgraph": "langgraph",
    "mcp": "mcp",
    "langchain_mcp_adapters": "langchain-mcp-adapters",
    "pypdf": "pypdf",
    "PIL": "Pillow",
    "pymupdf": "PyMuPDF",
}
FAKE_PRINTER_NAME = "Office_Printer"
DEFAULT_REQUEST = (
    "Compare at least four focusing experiments, changing one setting at a time while "
    "holding other settings fixed. Have the scientist select four experiments that "
    "answer the scientific question, requesting more runs if needed. "
    "Judge whether the comparison is fair, compare "
    "performance and computational cost, and suggest the next experiment. Then "
    "produce a 2/3-page report with one graph, its explanation and a verdict, and "
    "print it on the office printer after my approval."
)
# Paper color: asked once per new run; empty or one of these words means plain white (None).
PAPER_QUESTION = "Which paper color will you print on? Press Enter for the default (plain white): "
PAPER_COLOR_MAX = 40
DEFAULT_PAPER_WORDS = {"white", "plain white", "default"}
# Graph-internal keys of an interrupted visit; a resumed run starts at the orchestrator.
# paper_color is not one of them: a resumed run keeps it and is not asked again.
# Print tuning: tuning_session.json is the real record and the graph re-reads it on resume, so
# only per-visit routing keys are dropped (a pending action or approval, failure and poll
# counters). tuning_session_path and tuning_printer are kept, so the printer is not asked again.
RESUME_DROPPED_KEYS = (
    "pending_print", "pending_trial_print", "route", "hardware_passes", "last_printer_reading",
    "tuning_action", "tuning_failure", "tuning_failures", "tuning_polls",
    "report_request_passes", "report_workflow_failure", "approval", "trial_approval",
)
SCAN_ROUNDS = (1, 2, 3)
DEFAULT_SCAN_FORMATS = ("pdf", "png", "jpg", "jpeg")
TUNING_KINDS = ("tuning_printer", "trial_print_approval", "tuning_scan", "tuning_print_confirmation",
                "tuning_poll", "tuning_problem")


def write_experiments_table(state: dict, run_dir: Path) -> None:
    """Write only validated focusing evidence, marking the best final score."""
    from app.workflow.graph import _successful_focusing_results

    results = _successful_focusing_results(state)
    rows = [
        "| Experiment | macropixel_size | focus_size | background_size | iterations | search_seed | Initial enhancement | Final enhancement |",
        "|---|---|---|---|---|---|---|---|",
    ]
    best = max((result["final_enhancement"] for result in results), default=None)
    for number, result in enumerate(results, 1):
        settings = result["settings"]
        values = [str(settings.get(name, "-")) for name in (
            "macropixel_size", "focus_size", "background_size", "iterations", "search_seed",
        )]
        final = f"{result['final_enhancement']:.3f}"
        if result["final_enhancement"] == best:
            final = f"**{final}**"
        rows.append("| " + " | ".join([
            str(number), *values, f"{result['initial_enhancement']:.3f}", final,
        ]) + " |")
    (run_dir / "experiments_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def backup_run_json(run_dir: Path) -> Path:
    """Copy every existing JSON record before explicit recovery; never overwrite backups."""
    root = run_dir.resolve()
    records = [p for p in root.rglob("*.json") if "recovery_backups" not in p.relative_to(root).parts]
    backup = root / "recovery_backups" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    backup.mkdir(parents=True, exist_ok=False)
    for record in records:
        destination = backup / record.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record, destination)
    return backup


def load_resume(run_dir: Path, initial_decisions: int, *, resume_stopped_tuning: bool = False) -> tuple[dict, dict]:
    """Restore a saved run and discard an orchestrator action that never completed."""
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config.get("bench"), dict) or not isinstance(config.get("request"), str):
        raise ValueError("Saved config needs bench settings and the original request.")
    if not isinstance(state.get("dialogue"), list) or not isinstance(
        state.get("execution_results"), list
    ):
        raise ValueError("Saved state needs dialogue and execution_results lists.")
    if resume_stopped_tuning:
        from app.workflow.tuning import load_tuning_session, recover_stopped_tuning
        session_path = (run_dir / "tuning_session.json").resolve()
        if Path(state.get("tuning_session_path", session_path)).resolve() != session_path:
            raise ValueError("Saved tuning session must belong to the resumed run")
        session = load_tuning_session(session_path)
        if state.get("experiment_selection") != session["accepted_selection"]:
            raise ValueError("Saved scientific selection differs from the tuning session")
        backup = backup_run_json(run_dir)
        recover_stopped_tuning(session_path, backup_path=backup)
        state["tuning_session_path"] = str(session_path)
    # Failure and budget notices can follow the last unexecuted decision.
    for index in range(len(state["dialogue"]) - 1, -1, -1):
        turn = state["dialogue"][index]
        if turn["role"] == "system":
            continue
        if turn["role"] == "orchestrator":
            state["dialogue"].pop(index)
            # A progress summary that was never answered spent no decision.
            if not turn["text"].startswith("Progress summary for the human:"):
                state["decision_count"] = max(0, state.get("decision_count", 0) - 1)
                state.pop("decision", None)
        break
    for key in RESUME_DROPPED_KEYS:
        state.pop(key, None)
    # A run stopped at a budget check keeps its budget, so the question is asked again.
    state.setdefault("decision_budget", state.get("decision_count", 0) + initial_decisions)
    state["stop_reason"] = "running"
    return state, config


def human_stop_message(state: dict, run_dir: Path) -> str:
    """Distinguish the authoritative tuning stop from budget/report stops."""
    session_path = state.get("tuning_session_path")
    if session_path:
        from app.workflow.tuning import load_tuning_session
        session = load_tuning_session(session_path)
        if session["status"] == "stopped":
            return ("Print tuning was stopped by the human. Ordinary --resume keeps it stopped. "
                    "To explicitly reopen it: python main.py --resume "
                    f'"{run_dir}" --resume-stopped-tuning')
    if state.get("print_outcome") == "stopped":
        return "Final-report printing was stopped by the human; no printed completion is claimed."
    return ("Stopped by the human at a budget check. Continue with: "
            f'python main.py --resume "{run_dir}"')


def save_run(state: dict, run_dir: Path) -> None:
    """Save the latest graph state and a role-labelled transcript."""
    state_temp = run_dir / "state.json.tmp"
    state_temp.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    state_temp.replace(run_dir / "state.json")
    transcript = "\n\n".join(
        f"[{turn['role']}]\n{turn['text']}" for turn in state["dialogue"]
    )
    if state["execution_results"]:
        transcript += "\n\n[tool executions]\n" + json.dumps(
            state["execution_results"], indent=2, ensure_ascii=False, allow_nan=False
        )
    (run_dir / "dialogue.txt").write_text(transcript, encoding="utf-8")
    write_experiments_table(state, run_dir)


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def clean_paper_color(text: str | None) -> str | None:
    """The paper color with spaces tidied and at most PAPER_COLOR_MAX characters; None means plain white."""
    color = " ".join(str(text or "").split())[:PAPER_COLOR_MAX].strip()
    return None if not color or color.lower() in DEFAULT_PAPER_WORDS else color


def ask_paper_color() -> str | None:
    """Ask once for the paper color; without an interactive terminal use the default without asking."""
    if not sys.stdin.isatty():
        return None
    try:
        return clean_paper_color(input(PAPER_QUESTION))
    except EOFError:                   # Ctrl+D: the default, on a fresh line
        print(flush=True)
        return None


def paper_line(color: str | None) -> str:
    return f"Paper: {color}" if color else "Paper: plain white (default)"


def paper_context_line(color: str | None) -> str:
    """The paper line of a new run's context, so the agents know the paper too."""
    if color:
        return f"Paper: {color} (named by the user). Everything is printed in black toner on this paper.\n"
    return "Paper: plain white (default).\n"


def check_tuning_tools(printer_tools, tuning_tools) -> None:
    """Fail early if the graph's tuning loop would lack a tool (it builds tuning_map from both lists)."""
    from app.agents.tuning_executor import TUNING_TOOLS

    missing = TUNING_TOOLS - {tool.name for tool in [*printer_tools, *tuning_tools]}
    if missing:
        raise RuntimeError("print tuning needs these tools, which are missing: " + ", ".join(sorted(missing)))


def scan_folders(run_dir: Path, max_cycles=None) -> list[Path]:
    """Create run_dir/scans/round_1 .. round_3 (the folders the tuning pauses suggest) and return them."""
    folders = ([Path(run_dir) / "scans" / f"cycle_{cycle}" / f"step_{step}"
                for cycle in range(1, max_cycles + 1) for step in SCAN_ROUNDS] if max_cycles else
               [Path(run_dir) / "scans" / f"round_{number}" for number in SCAN_ROUNDS])
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def scan_formats() -> tuple[str, ...]:
    """File types read_scan accepts: the tuning config's scan.accepted_formats (as trial manifests copy it)."""
    try:
        from app.print_tuning.grayscale_render import load_config

        return tuple(str(name).lower() for name in load_config()["scan"]["accepted_formats"])
    except Exception:
        return DEFAULT_SCAN_FORMATS


def _ask_choice(question: str, choices) -> tuple[str, str]:
    """(choice, rest of the line): re-ask until the first word is one of choices."""
    choices = [str(choice).lower() for choice in choices]
    while True:
        reply = input(question + "\n> ").strip()
        if not reply:
            continue
        first, _, rest = reply.partition(" ")
        word = first.lower().strip(".,!:;")
        if word in choices:
            return word, rest.strip()
        print("Please answer with one of: " + ", ".join(choices), flush=True)


def resolve_scan_paths(text: str, folder: Path, formats=None) -> tuple[list[str], str]:
    """Validate explicit scan paths, one per line, preserving their supplied order.

    Spaces and backslashes are literal; matching surrounding single/double quotes
    are optional. No shell parsing, directory expansion, or scan discovery occurs.
    Empty input and directories are rejected.
    A relative path is looked up in the current folder, then in the suggested scan folder.
    """
    formats = tuple(formats or scan_formats())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return [], "give explicit scan file paths, one per line"
    paths, problems = [], []
    for word in lines:
        if word[0] in ("'", '"'):
            if len(word) < 2 or word[-1] != word[0]:
                problems.append("cannot read the path: surrounding quotes must match")
                continue
            word = word[1:-1]
        if not word.strip():
            problems.append("empty scan file path")
            continue
        path = Path(word).expanduser()
        if not path.is_absolute() and not path.exists() and (folder / path).exists():
            path = folder / path
        path = path.resolve()
        if path.is_dir():
            problems.append(f"directory given: {path}; name the specific scan files instead")
        elif not path.is_file():
            problems.append(f"file not found: {path}")
        elif path.suffix.lower().lstrip(".") not in formats:
            problems.append(f"{path.name} is not a scan file read_scan accepts ({', '.join(formats)})")
        else:
            paths.append(path)
    unique = list(dict.fromkeys(str(path) for path in paths))
    if problems or not unique:
        return [], "; ".join(problems) or "no scan files given"
    return unique, ""


def _ask_scan_files(folder: Path) -> dict:
    """Collect and confirm explicit files; keep local edits out of graph state."""
    while True:
        print('Enter one scan file path per line (one multipage PDF or the page images).\n'
              'Paths may contain spaces; surrounding quotes are optional.\n'
              'Finish with a blank line, or type "stop" to end tuning.', flush=True)
        lines = []
        while True:
            line = input("Scan file: ").strip()
            if line.lower() == "stop":
                return {"action": "stop"}
            if not line:
                break
            lines.append(line)
        paths, problem = resolve_scan_paths("\n".join(lines), folder)
        if problem:
            print(f"No usable scan files: {problem}. Enter the corrected selection.", flush=True)
            continue
        print("Selected scan files:\n  " + "\n  ".join(paths), flush=True)
        choice, _ = _ask_choice('Use these files? Type "confirm", "correct" to enter a new selection, or "stop".',
                                ("confirm", "correct", "stop"))
        if choice == "stop":
            return {"action": "stop"}
        if choice == "confirm":
            return {"action": "scan", "scan_paths": paths}


def _ask_tuning(payload: dict, default_printer: str | None) -> dict:
    """Show one print-tuning pause and return its answer as the dict the graph expects."""
    kind = payload["kind"]
    choices = payload.get("choices") or []
    question = payload.get("question", "Your answer:")
    if kind == "tuning_printer":
        print("Print tuning: which printer prints the trial sheets?")
        if default_printer:
            print(f"Default printer: {default_printer}")
        hint = (f'Type "printer" to use {default_printer}, ' if default_printer else "") + \
            'type "printer <exact name>", or "stop".'
        while True:
            choice, rest = _ask_choice(question + "\n" + hint, choices)
            if choice == "stop":
                return {"action": "stop"}
            if rest or default_printer:
                return {"action": "printer", "printer": rest or default_printer}
            print('Give the printer name, for example: printer Office_Printer', flush=True)
    if kind == "trial_print_approval":
        print(f"Trial sheet for round {payload.get('round_id')} (trial {payload.get('trial_id')})\n"
              f"PDF: {payload.get('pdf_path')}\nPages: {payload.get('page_count')}\n"
              f"Printer: {payload.get('printer')}\nPDF sha256: {payload.get('pdf_sha256')}\n"
              + (f"Effective settings: {json.dumps(payload['print_settings'], ensure_ascii=False)}\n"
                 if payload.get("print_settings") else "") + TRIAL_PRINT_REMINDER)
        return {"action": _ask_choice(question, choices)[0]}
    if kind == "tuning_scan":
        folder = Path(payload.get("suggested_scan_directory") or Path.cwd())
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Round {payload.get('round_id')} (trial {payload.get('trial_id')}): scan BOTH sides of the printed sheets as separate pages.\n"
              f"Save the scans in: {folder}\n"
              'Type "scan", then enter each file path on its own line and finish with a blank line. '
              'You will confirm the selected files before they are used. '
              '"reprint" to print the sheet again, or "stop".')
        while True:
            choice, rest = _ask_choice(question, choices)
            if choice != "scan":
                return {"action": choice}
            if rest:
                print('Type "scan" alone, then enter one path per line.', flush=True)
                continue
            return _ask_scan_files(folder)
    if kind == "tuning_print_confirmation":
        print("The trial print could not be confirmed: the printer did not report that it printed.\n"
              "Please check the printer tray. Did the pages come out? "
              '"printed" = yes, "not_printed" = no, "stop" = end tuning.')
    elif kind == "tuning_poll":
        print("The trial print is still waiting or printing.")
    elif kind == "tuning_problem" and payload.get("message"):
        print("Problem: " + str(payload["message"]))
    return {"action": _ask_choice(question, choices)[0]}


def ask_human(payload: dict, default_printer: str | None = None) -> str | dict:
    """Show one graph pause and return a valid answer; re-ask on empty or invalid input.

    Legacy report approval, printer-problem and budget pauses return text. Tuning
    pauses and other pauses with choices (including report_print_confirmation)
    return {"action": ...} dicts using the payload's allowed choices.
    """
    kind = payload.get("kind")
    print("\n" + "=" * 60, flush=True)
    if kind in TUNING_KINDS:
        return _ask_tuning(payload, default_printer)
    if kind == "report_workflow_problem" and "correct" in payload.get("choices", []):
        print(payload.get("message", ""), flush=True)
        choice, _ = _ask_choice(payload["question"], payload["choices"])
        if choice == "correct":
            return {"action": "correct", "instructions": input("Replacement revision instructions:\n> ")}
        return {"action": choice}
    if kind == "print_approval":
        print(f"Report: {payload.get('pdf_path')}\nPages: {payload.get('pages')}\n"
              f"Printer: {payload.get('printer') or 'the default printer'}\n"
              f"Copies: {payload.get('copies')}\nPDF sha256: {payload.get('pdf_sha256')}")
        allowed = None
    elif kind == "printer_problem":
        print("Printer reading:\n" + json.dumps(payload.get("reading"), indent=2, ensure_ascii=False))
        allowed = ("retry", "skip")
    elif kind == "budget":
        print(f"Progress summary:\n{payload.get('summary')}\n"
              f"Orchestrator decisions used: {payload.get('decisions_used')}")
        allowed = ("yes", "y", "no", "n")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if payload.get("choices"):
            return {"action": _ask_choice(payload.get("question", "Your answer:"), payload["choices"])[0]}
        allowed = None
    while True:
        reply = input(payload.get("question", "Your answer:") + "\n> ")
        if not reply.strip():
            continue
        first_word = reply.split()[0].lower().strip(".,!:;")
        if allowed is None or first_word in allowed:
            return reply
        print("Please answer with one of: " + ", ".join(allowed), flush=True)


def request_saved_report_revision(run_dir, state, instructions):
    """Explicit CLI revision, without rerunning scientific/tuning work."""
    from app.workflow.tuning import load_tuning_session
    from app.workflow.reporting import request_revision
    session = load_tuning_session(Path(run_dir) / "tuning_session.json")
    if session["status"] != "completed" or session["phase"] != "completed" or not session.get("final_settings"):
        raise ValueError("Report revision requires completed tuning; experiments/tuning will not be rerun")
    revision = request_revision(run_dir, instructions)
    return {**state, "report_revision_pending": True, "report_revision": revision, "pending_print": {}, "approval": {}}


def create_run(root, request, initial_decisions=10, paper_color=None):
    run_dir = Path(root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    state = {"dialogue": [{"role": "user", "text": request}], "execution_results": [],
             "decision_count": 0, "decision_budget": initial_decisions,
             "stop_reason": "running", "paper_color": clean_paper_color(paper_color)}
    return run_dir, state


def prepare_workflow(run_dir, state, *, request, model, specialist_model=None,
                     initial_decisions=10, extra_decisions=5, fake_printer=False,
                     printer_name=None, cycle_limit=3, saved_config=None, api_key=None,
                     fake_jobs_file=None):
    """Shared CLI/web setup. No terminal input, argument parsing, or graph execution."""
    from app.agents.common import MIN_EXPERIMENTS

    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from app.agents.report_and_print import create_information_tool
    from app.experiments.bench import SimulatedBench, SEARCH_SHOTS, VALIDATION_SHOTS
    from app.workflow.graph import build_graph
    from app.experiments.tools import create_coding_tool, create_focusing_tool, RUN_TIMEOUT_SECONDS
    from app.print_tuning.tuning_tools import create_tuning_tools

    if saved_config:
        bench_options = dict(saved_config["bench"])
        for name in ("dmd_shape", "camera_shape", "target_position"):
            if name in bench_options:
                bench_options[name] = tuple(bench_options[name])
        bench = SimulatedBench(**bench_options)
        print("Resuming in the same folder with the same transmission matrix and target. "
              "The measurement-noise sequence restarts from its saved seed.", flush=True)
    else:
        bench = SimulatedBench(
            dmd_shape=(16, 16), camera_shape=(32, 32),
            transmission_seed=42, noise_seed=43, noise_std=0.01,
        )
    bench_settings = {
        "dmd_shape": bench.dmd_shape,
        "camera_shape": bench.camera_shape,
        "target_position": bench.target_position,
        "transmission_seed": bench.transmission_seed,
        "noise_seed": bench.noise_seed,
        "noise_std": bench.noise_std,
    }
    context = (
        "Homework 3: hardware in the loop.\n"
        f"Bench settings: {json.dumps(bench_settings)}\n"
        "These dimensions and noise settings are simulation choices, not hardware specifications.\n"
        "The DMD controls binary amplitudes 0 or 1; an illuminated pattern cannot be all off.\n"
        "A fixed complex transmission matrix maps 256 DMD pixels to 1024 camera pixels.\n"
        "Camera intensity is I = abs(T @ pattern.ravel()) ** 2, followed by fresh "
        "additive Gaussian noise and clipping negative measurements to zero. "
        "Intensities are relative simulation units.\n"
        "One shared bench is used for the whole run: its matrix and target stay fixed, "
        "and its independent noise generator advances between captures.\n"
        "Use run_focusing_experiment for all optical experiments. Choose its parameters "
        "autonomously; do not recreate the bench or implement another search.\n"
        "Valid macropixel sizes are 1, 2, 4, and 8: each divides both DMD dimensions "
        "and leaves at least two controllable blocks. Each iteration flips one block.\n"
        "Focus and background sizes must be positive odd integers, with background "
        "larger than focus. Both centred squares must fit around the fixed target; "
        "the four background regions must be nonempty. Coordinates are (row, column).\n"
        "Background is the larger square outside the focus square, divided into "
        "top-left, top-right, bottom-left, and bottom-right regions. The target row "
        "belongs to bottom regions and the target column to right regions.\n"
        "Enhancement = focus mean / max(mean of the four background-region means, 1e-12). "
        "The four means have equal weight, regardless of region pixel counts.\n"
        f"Average {SEARCH_SHOTS} fresh captures for each search measurement and "
        f"{VALIDATION_SHOTS} fresh captures for final validation. The returned final "
        "enhancement comes from validation, not the highest noisy search score.\n"
        "Use 1 to 1,000 iterations per tool call. Improvement is not guaranteed.\n"
        "Keeping focus and background regions fixed makes comparisons between "
        "macropixel sizes or iteration budgets easier to interpret. Changing the "
        "regions changes the metric; acknowledge that when comparing results.\n"
        f"The run compares at least {MIN_EXPERIMENTS} successful focusing experiments "
        "that change one setting.\n"
        "Computational cost is compared by iterations and camera shots per measurement "
        "(and by wall_time_seconds and camera_shots if the results include them).\n"
        "The final product is report.pdf in the output directory: 2 or 3 A4 portrait pages with one "
        "graph, its explanation, a fairness judgement, performance vs cost, a suggested "
        "next experiment and a verdict on page 1; Hardware loop results on page 2. "
        "The executor calls build_final_report, which uses a reusable layout, frozen "
        "scientific evidence and existing accepted scans. No additional tuning round is needed.\n"
        "The office printer is reached only through the executor's printer tools in the "
        "report-and-print turn. The graph asks the human before any print. "
        "Printer states: idle, printing, stopped, offline, error, unknown, out_of_paper, "
        "paper_jam. Job states: waiting, printing, done, failed, uncertain.\n"
        "Print tuning: after the four experiments are chosen, the graph tunes how their heatmaps "
        "print in three-step cycles: gamma, then shared dark/light gray levels, then gray steps. "
        "After each accepted cycle the human chooses Continue, Finish or Stop; continuation is not print approval.\n"
        "Every trial sheet print needs human approval of that exact PDF and printer. The human "
        f"scans both faces into {run_dir / 'scans'}/cycle_N/step_N. Legacy round_N paths remain accepted.\n"
        + (
            f"The office printer is named {printer_name}; pass printer=\"{printer_name}\" "
            "to printer_status and submit_print.\n"
            if printer_name else
            "Call list_printers first and pass the office printer's exact name as printer "
            "to printer_status and submit_print.\n"
        ) +
        paper_context_line(state.get("paper_color")) +
        f"Output directory: {run_dir}\n"
        "The focusing tool saves initial/final camera images, the final pattern, "
        "results.json, and an experiments.csv table. Do not duplicate these outputs.\n"
        "Base all conclusions on recorded tool results, never proposals or invented "
        "measurements."
    )
    if not saved_config:
        state["dialogue"].insert(0, {"role": "system", "text": context})
    run_config = dict(saved_config) if saved_config else {
        "provider": "openai",
        "api": "responses",
        "reasoning_effort": "medium",
        "python": sys.version,
        "bench": bench_settings,
        "max_tuning_cycles": cycle_limit,
        "request": request,
        "script_timeout_seconds": RUN_TIMEOUT_SECONDS,
    }
    run_config.pop("max_orchestrator_decisions", None)
    run_config.pop("printer_backend", None)
    run_config.update(
        model=model, min_experiments=MIN_EXPERIMENTS,
        initial_decisions=initial_decisions, extra_decisions=extra_decisions,
        fake_printer=fake_printer, printer_name=printer_name, specialist_model=specialist_model,
    )
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_run(state, run_dir)

    # Only variables the printer files read; FAKE_PRINTER_PROBLEM comes from the shell.
    printer_env = {"PRINTER_MODE": "fake" if fake_printer else "real"}
    if fake_jobs_file is not None:
        printer_env["FAKE_PRINTER_FILE"] = str(fake_jobs_file)
    if printer_name:
        printer_env["PRINTER_ALLOWED"] = printer_name
    try:
        from app.printing.tools import load_printer_tools

        printer_tools = load_printer_tools(printer_env)
    except Exception as error:
        print(f"Could not start the printer MCP server: {error}", file=sys.stderr, flush=True)
        raise

    def chat_model(model):
        return ChatOpenAI(
            model=model, api_key=api_key, max_tokens=16384,
            timeout=120, max_retries=2,
            use_responses_api=True, output_version="responses/v1",
            reasoning={"effort": "medium"},
            store=False, include=["reasoning.encrypted_content"],
        )

    llm = chat_model(model)
    coding_tool = create_coding_tool(run_dir)
    focusing_tool = create_focusing_tool(run_dir, bench)
    tuning_tools = create_tuning_tools(run_dir)
    check_tuning_tools(printer_tools, tuning_tools)
    workflow = build_graph(
        llm, focusing_tool, coding_tool, printer_tools, create_information_tool(),
        run_dir, extra_decisions, InMemorySaver(), tuning_tools=tuning_tools, max_cycles=cycle_limit,
        **({"specialist_llm": chat_model(specialist_model)} if specialist_model else {}),
    )
    return workflow


def main() -> int:
    # Model replies can contain symbols unsupported by redirected Windows consoles.
    # Keep saved evidence in UTF-8 and prevent display errors from stopping a run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Override the configured model.")
    parser.add_argument("--request", help="Override the autonomous request for a new run.")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR", help="Continue a saved run in the same folder.")
    parser.add_argument("--report-revision", metavar="TEXT",
                        help="With --resume, request report edits without rerunning experiments or tuning.")
    parser.add_argument("--max-tuning-cycles", type=positive_int, default=None,
                        help="Maximum tuning cycles for a new run (default 3, maximum 20); saved sessions retain their limit.")
    parser.add_argument("--resume-stopped-tuning", action="store_true",
                        help="With --resume, back up JSON records and explicitly reopen human-stopped tuning.")
    parser.add_argument("--initial-decisions", type=positive_int, metavar="N",
                        help="Orchestrator decisions before the first check-in (default 10).")
    parser.add_argument("--extra-decisions", type=positive_int, metavar="N",
                        help="Decisions added each time the human says yes (default 5).")
    parser.add_argument("--fake-printer", action=argparse.BooleanOptionalAction, default=None,
                        help=f"Use the fake printer {FAKE_PRINTER_NAME} (PRINTER_MODE=fake).")
    parser.add_argument("--printer", metavar="NAME", help="Exact name of the office printer.")
    parser.add_argument("--specialist-model", metavar="MODEL",
                        help="Separate image-capable model for the photo quality specialist "
                             "(default: OPENAI_SPECIALIST_MODEL, else the main model).")
    parser.add_argument("--paper-color", metavar="COLOR",
                        help='Paper color to print on, e.g. "light yellow" (default: plain white; '
                             "asked at the start of a new run if not given).")
    args = parser.parse_args()
    if args.max_tuning_cycles is not None and (args.max_tuning_cycles > 20 or args.resume):
        parser.error("--max-tuning-cycles is 1..20 and applies only to new runs")
    if args.resume_stopped_tuning and not args.resume:
        parser.error("--resume-stopped-tuning requires --resume RUN_DIR.")
    if args.report_revision is not None and (not args.resume or not args.report_revision.strip()):
        parser.error("--report-revision requires --resume RUN_DIR and nonempty instructions.")
    if args.resume and args.request is not None:
        parser.error("--request cannot change the request of a resumed run.")
    if args.resume and args.paper_color is not None:
        parser.error("--paper-color cannot change the paper of a resumed run.")
    args.request = args.request if args.request is not None else DEFAULT_REQUEST
    if not args.request.strip():
        parser.error("--request must not be blank.")
    missing = [
        package for module, package in REQUIRED_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        parser.error(
            "Missing required dependencies: " + ", ".join(missing)
            + f'. Install them with "{sys.executable}" -m pip install -r '
            + f'"{BASE_DIR / "requirements.txt"}"'
        )
    from dotenv import load_dotenv
    from app.agents.common import DEFAULT_INITIAL_DECISIONS, MIN_EXPERIMENTS
    from app.agents.orchestrator import DEFAULT_EXTRA_DECISIONS

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        parser.error("Set OPENAI_API_KEY in your environment or the .env beside main.py.")

    # Command-line values win; a resumed run otherwise keeps its saved values.
    saved = {}
    if args.resume:
        try:
            saved = json.loads((args.resume.resolve() / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            parser.error(f"Cannot resume {args.resume.resolve()}: {error}")
    initial_decisions = args.initial_decisions or saved.get("initial_decisions") or DEFAULT_INITIAL_DECISIONS
    extra_decisions = args.extra_decisions or saved.get("extra_decisions") or DEFAULT_EXTRA_DECISIONS
    fake_printer = (
        args.fake_printer if args.fake_printer is not None
        else bool(saved.get("fake_printer", False))
    )
    printer_name = (
        args.printer or saved.get("printer_name")
        or (FAKE_PRINTER_NAME if fake_printer else None)
    )

    # Fail before models, approval, run creation, or stopped-session recovery.
    from app.printing.connector import PrinterError, trial_print_preflight
    try:
        trial_print_preflight(fake_printer=fake_printer)
    except PrinterError as error:
        parser.error(str(error))

    saved_config = None
    if args.resume:
        run_dir = args.resume.resolve()
        try:
            state, saved_config = load_resume(run_dir, initial_decisions,
                                              resume_stopped_tuning=args.resume_stopped_tuning)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"Cannot resume {run_dir}: {error}")
        args.request = saved_config["request"]
        if args.report_revision is not None:
            try:
                state = request_saved_report_revision(run_dir, state, args.report_revision)
            except (OSError, ValueError) as error:
                parser.error(str(error))
    else:
        # Asked before the run folder exists, so stopping at the question leaves no empty folder.
        paper_color = clean_paper_color(args.paper_color) if args.paper_color is not None else ask_paper_color()
        run_dir, state = create_run(BASE_DIR / "runs", args.request, initial_decisions, paper_color)
    args.model = args.model or (saved_config or {}).get("model") or os.getenv("OPENAI_MODEL", "gpt-5-nano")
    specialist_model = (args.specialist_model or (saved_config or {}).get("specialist_model")
                        or os.getenv("OPENAI_SPECIALIST_MODEL") or None)
    if specialist_model == args.model:
        specialist_model = None
    print(f"Model: {args.model}\nResults: {run_dir}\nRequest: {args.request}\n"
          f"Printer: {printer_name or 'found with list_printers'}"
          + (" (fake)" if fake_printer else "") + "\n" + paper_line(state.get("paper_color")), flush=True)
    if specialist_model:
        print(f"Specialist model: {specialist_model}", flush=True)
    cycle_limit = args.max_tuning_cycles or saved.get("max_tuning_cycles", 3)
    folders = scan_folders(run_dir, cycle_limit if not saved or saved.get("max_tuning_cycles") else None)
    print(f"Trial scan directories: {folders[0]} ... {folders[-1]}", flush=True)

    try:
        from langgraph.types import Command
        workflow = prepare_workflow(
            run_dir, state, request=args.request, model=args.model, specialist_model=specialist_model,
            initial_decisions=initial_decisions, extra_decisions=extra_decisions,
            fake_printer=fake_printer, printer_name=printer_name, cycle_limit=cycle_limit,
            saved_config=saved_config, api_key=api_key,
        )
        # The report-and-print self-loop needs more than the default 25 steps.
        config = {"configurable": {"thread_id": run_dir.name}, "recursion_limit": 200}
        printed_dialogue = len(state["dialogue"])
        printed_results = len(state["execution_results"])
        graph_input = state
        while True:
            for snapshot in workflow.stream(graph_input, config, stream_mode="values"):
                # An interrupt arrives as the state plus an "__interrupt__" entry.
                state = {key: value for key, value in snapshot.items() if key != "__interrupt__"}
                save_run(state, run_dir)
                for turn in state["dialogue"][printed_dialogue:]:
                    print(f"\n[{turn['role']}]\n{turn['text']}", flush=True)
                for record in state["execution_results"][printed_results:]:
                    print("\n[tool execution]\n" + json.dumps(
                        record, indent=2, ensure_ascii=False
                    ), flush=True)
                printed_dialogue = len(state["dialogue"])
                printed_results = len(state["execution_results"])
            pending = workflow.get_state(config)
            if not pending.next:
                break
            reply = ask_human(pending.tasks[0].interrupts[0].value, default_printer=printer_name)
            graph_input = Command(resume=reply)
    except (Exception, KeyboardInterrupt) as error:
        failure_trace = traceback.format_exc()
        state["stop_reason"] = "execution_failure"
        message = f"Execution/API failure ({type(error).__name__}); see error.txt."
        state["dialogue"].append({"role": "system", "text": message})
        try:
            (run_dir / "error.txt").write_text(failure_trace, encoding="utf-8")
            save_run(state, run_dir)
        except Exception as save_error:
            print(f"Could not save the latest failure evidence: {save_error}", file=sys.stderr)
        print(f"\n{message}\nLatest available evidence: {run_dir}", file=sys.stderr)
        print(f"Continue with: python main.py --resume {run_dir}", file=sys.stderr)
        return 1

    print(f"\nStopped: {state['stop_reason']}")
    if state["stop_reason"] == "finished":
        print("Finished: the report was made and printed, or printing was skipped by the human.")
        print(f"Report: {(state.get('report') or {}).get('path', run_dir / 'report.pdf')}")
        print(f"Print outcome: {state.get('print_outcome')}")
    elif state["stop_reason"] == "stopped_by_human":
        print(human_stop_message(state, run_dir))
    else:
        print("The workflow ended without an evidence-backed finish.")

    from app.workflow.graph import _successful_focusing_results
    successful = _successful_focusing_results(state)
    if successful:
        best = max(successful, key=lambda result: result["final_enhancement"])
        print(f"Highest recorded final enhancement: {best['final_enhancement']:.6g}")
        print(f"Experiment: {best['experiment_id']}")
        print("Settings: " + json.dumps(best["settings"]))
        print("Saved outputs: " + json.dumps(best["paths"]))
    print(f"Dialogue and execution evidence: {run_dir}")
    return 0 if state["stop_reason"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
