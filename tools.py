import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
from uuid import uuid4

import numpy as np
from langchain_core.tools import BaseTool, tool

from bench import SimulatedBench, run_focusing_search


RUN_TIMEOUT_SECONDS = 300


def _text(output: bytes | None) -> str:
    """Decode captured output, including partial output after a timeout."""
    return (output or b"").decode("utf-8", errors="replace")


def create_coding_tool(work_dir: str | Path) -> BaseTool:
    """Create a write_and_run tool tied to this run's working directory."""
    directory = Path(work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    @tool
    def write_and_run(filename: str, code: str) -> dict:
        """Write a new Python file and execute it with a five-minute timeout.

        filename: A fresh filename such as attempt_1.py. Use only letters,
            digits, underscores, or hyphens before .py; do not supply a path.
        code: The complete Python source to write and run.

        Returns exit_code, stdout, stderr, and timed_out. An exit_code of None
        means there was no normal process exit code, such as on a timeout or
        file-writing failure. The script runs in the configured run directory
        using the same Python environment as the agent system.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.py", filename):
            return {
                "exit_code": None, "stdout": "", "timed_out": False,
                "stderr": "Invalid filename. Use a simple name such as attempt_1.py, without a path.",
            }

        script = directory / filename
        try:
            # Exclusive creation preserves earlier attempts and prevents overwriting.
            with script.open("x", encoding="utf-8") as file:
                file.write(code)

            run = subprocess.run(
                [sys.executable, "-X", "utf8", "-u", str(script)],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=RUN_TIMEOUT_SECONDS,
            )
            return {
                "exit_code": run.returncode,
                "stdout": _text(run.stdout),
                "stderr": _text(run.stderr),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as error:
            return {
                "exit_code": None,
                "stdout": _text(error.stdout),
                "stderr": _text(error.stderr)
                + f"\nExecution timed out after {RUN_TIMEOUT_SECONDS} seconds; the script process was stopped.",
                "timed_out": True,
            }
        except OSError:
            return {
                "exit_code": None, "stdout": "", "stderr": traceback.format_exc(),
                "timed_out": False,
            }

    return write_and_run


def _draw_camera_image(axes, image, target, focus_size, enhancement, title, vmax):
    """Use the same scale, focus outline and labels for every camera panel."""
    from matplotlib.patches import Rectangle

    camera = axes.imshow(image, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
    row, column = target
    half = focus_size // 2
    axes.add_patch(Rectangle(
        (column - half - 0.5, row - half - 0.5), focus_size, focus_size,
        fill=False, edgecolor="red", linewidth=1.5,
    ))
    axes.set_title(f"{title} (enhancement = {enhancement:.4f})")
    axes.set_xlabel("Camera column")
    axes.set_ylabel("Camera row")
    return camera


def _save_camera_image(image, path, target, focus_size, enhancement, title, vmax):
    """Save a camera image with the shared colour scale and focus outline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 5), layout="constrained")
    try:
        camera = _draw_camera_image(
            axes, image, target, focus_size, enhancement, title, vmax,
        )
        figure.colorbar(camera, ax=axes, label="Relative intensity")
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def _save_summary_figure(result, path, target, focus_size, vmax):
    """Compare both camera images and the search history in one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    try:
        for axis, stage in zip(axes[:2], ("initial", "final")):
            camera = _draw_camera_image(
                axis, result[f"{stage}_image"], target, focus_size,
                result[f"{stage}_enhancement"], stage.capitalize(), vmax,
            )
        figure.colorbar(camera, ax=list(axes[:2]), label="Relative intensity")
        history = result["history"]
        axes[2].plot(
            [entry["iteration"] for entry in history],
            [entry["current_enhancement"] for entry in history],
            label="Search measurement",
        )
        final = result["final_enhancement"]
        axes[2].axhline(
            final, linestyle="--", color="tab:orange",
            label=f"Final validated: {final:.4f}",
        )
        axes[2].set_title("Enhancement during search")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Enhancement")
        axes[2].grid(alpha=0.3)
        axes[2].legend()
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def create_focusing_tool(work_dir: str | Path, bench: SimulatedBench) -> BaseTool:
    """Create a focusing tool that reuses the supplied bench for every call."""
    directory = Path(work_dir).resolve()

    @tool
    def run_focusing_experiment(
        macropixel_size: int,
        focus_size: int,
        background_size: int,
        iterations: int,
        search_seed: int = 0,
    ) -> dict:
        """Perform a real numerical search on the simulated optical bench.

        Macropixel size must divide both DMD dimensions and leave at least
        two controllable macropixels. Focus and background sizes must be
        positive odd integers, with background larger than focus; both
        squares must fit around the fixed target. Enhancement is focus mean
        intensity divided by the mean of the four background-region means.
        Measurements include noise, so improvement is not guaranteed.
        Use 1 to 1,000 iterations. Output paths are generated internally.
        """
        try:
            if iterations > 1000:
                raise ValueError("iterations must not exceed 1,000 per call.")
            grid_shape = bench.macropixel_shape(macropixel_size)
            if grid_shape[0] * grid_shape[1] < 2:
                raise ValueError("macropixel_size must leave at least two controllable blocks.")
            result = run_focusing_search(
                bench, macropixel_size, focus_size, background_size,
                iterations, search_seed=search_seed,
            )

            experiment_id = f"experiment_{uuid4().hex}"
            experiment_dir = directory / experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=False)
            paths = {
                "initial_camera": str(experiment_dir / "initial_camera.png"),
                "final_camera": str(experiment_dir / "final_camera.png"),
                "final_pattern": str(experiment_dir / "final_pattern.npy"),
                "results_json": str(experiment_dir / "results.json"),
                "experiments_csv": str(directory / "experiments.csv"),
                "summary_figure": str(experiment_dir / "summary.png"),
            }
            vmax = float(max(result["initial_image"].max(), result["final_image"].max()))
            for stage in ("initial", "final"):
                _save_camera_image(
                    result[f"{stage}_image"], paths[f"{stage}_camera"],
                    bench.target_position, focus_size, result[f"{stage}_enhancement"],
                    f"{stage.capitalize()} camera", vmax,
                )
            _save_summary_figure(
                result, paths["summary_figure"], bench.target_position, focus_size, vmax,
            )
            np.save(paths["final_pattern"], result["final_pattern"], allow_pickle=False)

            summary = {
                "experiment_id": experiment_id,
                "settings": result["settings"],
                "initial_enhancement": result["initial_enhancement"],
                "final_enhancement": result["final_enhancement"],
                "focus_mean": result["focus_mean"],
                "background_means": result["background_means"],
                "background_reference": result["background_reference"],
                "paths": paths,
            }
            record = {**summary, "history": result["history"]}
            Path(paths["results_json"]).write_text(
                json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
            )

            row = {
                "experiment_id": experiment_id,
                **{name: result["settings"][name] for name in (
                    "macropixel_size", "focus_size", "background_size",
                    "iterations", "search_seed",
                )},
                "initial_enhancement": result["initial_enhancement"],
                "final_enhancement": result["final_enhancement"],
                "focus_mean": result["focus_mean"],
                **{f"background_{name}_mean": value
                   for name, value in result["background_means"].items()},
                "background_reference": result["background_reference"],
                "results_json": paths["results_json"],
            }
            table = Path(paths["experiments_csv"])
            needs_header = not table.exists() or table.stat().st_size == 0
            with table.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
            return {"ok": True, **summary}
        except Exception as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    return run_focusing_experiment
