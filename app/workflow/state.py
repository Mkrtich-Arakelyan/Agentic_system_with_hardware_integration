"""Existing graph state and additive message/evidence reducers."""

from operator import add
from typing import Annotated, TypedDict


class State(TypedDict, total=False):
    dialogue: Annotated[list[dict], add]
    execution_results: Annotated[list[dict], add]
    decision: dict
    decision_count: int
    decision_budget: int
    stop_reason: str
    route: str
    pending_print: dict
    approval: dict
    report: dict
    print_outcome: str
    active_job_id: str
    hardware_passes: int
    budget_summary: str
    last_printer_reading: dict
    scientist_response: dict
    experiment_selection: dict
