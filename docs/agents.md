# Agent modules

| Module | Existing responsibility |
| --- | --- |
| `app/agents/orchestrator.py` | Routing prompt/schema, orchestrator call, progress summary prompt/call, extra-decision budget constant |
| `app/agents/scientist.py` | Scientific prompt, experiment selection/proposal/response schemas, scientist call |
| `app/agents/executor.py` | Experiment-execution prompt and executor turn |
| `app/agents/report_and_print.py` | Report-and-print executor mode, information-request tool factory, report inspection, approval and job-tracking helpers, report-pass limits |
| `app/agents/general.py` | General explanation prompt and call |
| `app/agents/common.py` | Shared message construction, tool-result summary helper, initial-budget and experiment-minimum constants |
| `app/agents/photo_quality_specialist.py` | Existing standalone image-message preparation and validated tuning decisions; preserved during the split |

`report_and_print` remains a specialized executor mode. This module split changes
neither tool distribution nor the graph's existing nodes, routing, approval gates,
budgets, retries, execution order, or side effects.

Existing public imports remain available:

```python
from app.agents import ScientistResponse, scientist, orchestrator, report_and_print
```

For internal application code, import functions explicitly from their modules:

```python
from app.agents.scientist import scientist as run_scientist
from app.agents.orchestrator import orchestrator as run_orchestrator
from app.agents.report_and_print import report_and_print as run_report_and_print
```

The graph uses these callable aliases so a same-named submodule cannot change
what it dispatches. Use `importlib.import_module("app.agents.scientist")` when a
module object itself is needed; `from app.agents import scientist` intentionally
returns the public callable. Imports create no model or tool instances.
