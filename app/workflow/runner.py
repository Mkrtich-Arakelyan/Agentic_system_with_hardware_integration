"""Run Homework 3: hardware in the loop with an office printer."""

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

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
    "produce a one-page report with one graph, its explanation and a verdict, and "
    "print it on the office printer after my approval."
)
# Paper color: asked once per new run; empty or one of these words means plain white (None).
PAPER_QUESTION = "Which paper color will you print on? Press Enter for the default (plain white): "
PAPER_COLOR_MAX = 40
DEFAULT_PAPER_WORDS = {"white", "plain white", "default"}
# Graph-internal keys of an interrupted visit; a resumed run starts at the orchestrator.
# paper_color is not one of them: a resumed run keeps it and is not asked again.
RESUME_DROPPED_KEYS = (
    "pending_print", "pending_trial_print", "route", "hardware_passes", "last_printer_reading",
)


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
        values = [str(settings.get(name, "—")) for name in (
            "macropixel_size", "focus_size", "background_size", "iterations", "search_seed",
        )]
        final = f"{result['final_enhancement']:.3f}"
        if result["final_enhancement"] == best:
            final = f"**{final}**"
        rows.append("| " + " | ".join([
            str(number), *values, f"{result['initial_enhancement']:.3f}", final,
        ]) + " |")
    (run_dir / "experiments_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def load_resume(run_dir: Path, initial_decisions: int) -> tuple[dict, dict]:
    """Restore a saved run and discard an orchestrator action that never completed."""
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config.get("bench"), dict) or not isinstance(config.get("request"), str):
        raise ValueError("Saved config needs bench settings and the original request.")
    if not isinstance(state.get("dialogue"), list) or not isinstance(
        state.get("execution_results"), list
    ):
        raise ValueError("Saved state needs dialogue and execution_results lists.")
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


def save_run(state: dict, run_dir: Path) -> None:
    """Save the latest graph state and a role-labelled transcript."""
    (run_dir / "state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
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


def ask_human(payload: dict) -> str:
    """Show one graph pause and return a valid answer; re-ask on empty or invalid input."""
    kind = payload.get("kind")
    print("\n" + "=" * 60, flush=True)
    if kind == "print_approval":
        print(f"Report: {payload.get('pdf_path')}\nPages: {payload.get('pages')}\n"
              f"Printer: {payload.get('printer') or 'the default printer'}\n"
              f"Copies: {payload.get('copies')}")
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
        allowed = None
    while True:
        reply = input(payload.get("question", "Your answer:") + "\n> ").strip()
        if not reply:
            continue
        first_word = reply.split()[0].lower().strip(".,!:;")
        if allowed is None or first_word in allowed:
            return reply
        print("Please answer with one of: " + ", ".join(allowed), flush=True)


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
    parser.add_argument("--initial-decisions", type=positive_int, metavar="N",
                        help="Orchestrator decisions before the first check-in (default 10).")
    parser.add_argument("--extra-decisions", type=positive_int, metavar="N",
                        help="Decisions added each time the human says yes (default 5).")
    parser.add_argument("--fake-printer", action=argparse.BooleanOptionalAction, default=None,
                        help=f"Use the fake printer {FAKE_PRINTER_NAME} (PRINTER_MODE=fake).")
    parser.add_argument("--printer", metavar="NAME", help="Exact name of the office printer.")
    parser.add_argument("--paper-color", metavar="COLOR",
                        help='Paper color to print on, e.g. "light yellow" (default: plain white; '
                             "asked at the start of a new run if not given).")
    args = parser.parse_args()
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

    saved_config = None
    if args.resume:
        run_dir = args.resume.resolve()
        try:
            state, saved_config = load_resume(run_dir, initial_decisions)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"Cannot resume {run_dir}: {error}")
        args.request = saved_config["request"]
    else:
        # Asked before the run folder exists, so stopping at the question leaves no empty folder.
        paper_color = clean_paper_color(args.paper_color) if args.paper_color is not None else ask_paper_color()
        run_dir = BASE_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "dialogue": [{"role": "user", "text": args.request}],
            "execution_results": [],
            "decision_count": 0,
            "decision_budget": initial_decisions,
            "stop_reason": "running",
            "paper_color": paper_color,
        }
    args.model = args.model or (saved_config or {}).get("model") or os.getenv("OPENAI_MODEL", "gpt-5-nano")
    print(f"Model: {args.model}\nResults: {run_dir}\nRequest: {args.request}\n"
          f"Printer: {printer_name or 'found with list_printers'}"
          + (" (fake)" if fake_printer else "") + "\n" + paper_line(state.get("paper_color")), flush=True)

    try:
        from langchain_openai import ChatOpenAI
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command
        from app.agents.report_and_print import create_information_tool
        from app.experiments.bench import SimulatedBench, SEARCH_SHOTS, VALIDATION_SHOTS
        from app.workflow.graph import build_graph
        from app.experiments.tools import create_coding_tool, create_focusing_tool, RUN_TIMEOUT_SECONDS

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
            "The final product is report.pdf in the output directory: one A4 page with one "
            "graph, its explanation, a fairness judgement, performance vs cost, a suggested "
            "next experiment and a verdict. The executor builds it with write_and_run from "
            "the saved files experiments.csv and each experiment's results.json; it must "
            f"not type numbers into the script. Scripts must finish within {RUN_TIMEOUT_SECONDS} seconds.\n"
            "The office printer is reached only through the executor's printer tools in the "
            "report-and-print turn. The graph asks the human before any print. "
            "Printer states: idle, printing, stopped, offline, error, unknown, out_of_paper, "
            "paper_jam. Job states: waiting, printing, done, failed, unknown.\n"
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
            "request": args.request,
            "script_timeout_seconds": RUN_TIMEOUT_SECONDS,
        }
        run_config.pop("max_orchestrator_decisions", None)
        run_config.pop("printer_backend", None)
        run_config.update(
            model=args.model, min_experiments=MIN_EXPERIMENTS,
            initial_decisions=initial_decisions, extra_decisions=extra_decisions,
            fake_printer=fake_printer, printer_name=printer_name,
        )
        (run_dir / "config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        save_run(state, run_dir)

        # Only variables the printer files read; FAKE_PRINTER_PROBLEM comes from the shell.
        printer_env = {}
        if fake_printer:
            printer_env["PRINTER_MODE"] = "fake"
        if printer_name:
            printer_env["PRINTER_ALLOWED"] = printer_name
        try:
            from app.printing.tools import load_printer_tools

            printer_tools = load_printer_tools(printer_env)
        except Exception as error:
            print(f"Could not start the printer MCP server: {error}", file=sys.stderr, flush=True)
            raise

        llm = ChatOpenAI(
            model=args.model, api_key=api_key, max_tokens=16384,
            timeout=120, max_retries=2,
            use_responses_api=True, output_version="responses/v1",
            reasoning={"effort": "medium"},
            store=False, include=["reasoning.encrypted_content"],
        )
        coding_tool = create_coding_tool(run_dir)
        focusing_tool = create_focusing_tool(run_dir, bench)
        workflow = build_graph(
            llm, focusing_tool, coding_tool, printer_tools, create_information_tool(),
            run_dir, extra_decisions, InMemorySaver(),
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
            reply = ask_human(pending.tasks[0].interrupts[0].value)
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
        print("Stopped by the human at a budget check. Continue with: "
              f"python main.py --resume {run_dir}")
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
