# Agentic system with hardware integration

Homework 2 Task 2: a four-role LangGraph system that runs focusing experiments on a simulated optical bench. The current implementation uses simulated hardware.

## Run

Use Python 3.13 and install the dependencies:

```shell
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your own OpenAI API key. Then run:

```shell
python main.py
```

Continue an interrupted run in its existing folder:

```shell
python main.py --resume runs/<folder>
```

The transmission matrix is recreated identically on resume, while the measurement-noise sequence restarts.

## Files

- `agents.py`: role prompts, structured routing, tool dispatch, and workflow limits.
- `graph.py`: graph routing and evidence-based completion checks.
- `bench.py`: the simulated DMD, transmission matrix, camera, and focusing search.
- `tools.py`: experiment output generation and supporting Python execution.
- `main.py`: command-line entry point, saved state, resume, and experiment tables.
- `runs/`: saved experiment evidence and figures.
- `videos/`: recorded demonstration.
- `SNLS80mV.mat`: the included Silverbox dataset from earlier homework work.

Run artifacts preserve the original local paths as recorded evidence. Local credentials and Python caches are excluded.
