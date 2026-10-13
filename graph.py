import math
from operator import add
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

import agents
from agents import MAX_DECISIONS, MIN_EXPERIMENTS


DECISION_LIMIT_MESSAGE = (
    f"The {MAX_DECISIONS}-orchestrator-decision budget was reached. "
    "The workflow stopped without an evidence-backed finish."
)


class State(TypedDict, total=False):
    dialogue: Annotated[list[dict], add]
    execution_results: Annotated[list[dict], add]
    decision: dict
    decision_count: int
    stop_reason: str


def _successful_focusing_results(state: dict) -> list[dict]:
    """Check recorded tool payloads, not scores mentioned in dialogue text."""
    successful = []
    for record in state.get("execution_results", []):
        if (
            not isinstance(record, dict)
            or record.get("tool_name") != "run_focusing_experiment"
            or record.get("validation_error")
        ):
            continue
        result = record.get("result")
        if not isinstance(result, dict) or result.get("ok") is not True:
            continue
        if not isinstance(result.get("experiment_id"), str) or not result["experiment_id"]:
            continue
        if not isinstance(result.get("settings"), dict) or not result["settings"]:
            continue
        scores = (result.get("initial_enhancement"), result.get("final_enhancement"))
        try:
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in scores
            ):
                continue
        except OverflowError:
            continue
        paths = result.get("paths")
        if isinstance(paths, dict) and all(
            isinstance(paths.get(name), str) and paths[name]
            for name in ("initial_camera", "final_camera", "results_json")
        ):
            successful.append(result)
    return successful


def _count_successful_focusing_results(state: dict) -> int:
    """Count valid focusing-tool evidence using the shared validation checks."""
    return len(_successful_focusing_results(state))


def build_graph(
    llm: BaseChatModel,
    coding_tool: BaseTool,
    focusing_tool: BaseTool,
):
    """Route the four roles; only the executor receives both tools."""

    def orchestrator_node(state: State) -> dict:
        # Let the final selected action complete, then stop before another LLM call.
        if state.get("decision_count", 0) >= MAX_DECISIONS:
            return {
                "stop_reason": "decision_limit",
                "dialogue": [{"role": "system", "text": DECISION_LIMIT_MESSAGE}],
            }

        update = agents.orchestrator(state, llm)
        next_role = update["decision"]["next_role"]
        update["stop_reason"] = "running"

        if next_role == "finish":
            count = _count_successful_focusing_results(state)
            if count >= MIN_EXPERIMENTS:
                update["stop_reason"] = "finished"
            else:
                has_plan = any(turn["role"] == "scientist" for turn in state["dialogue"])
                redirected_role = "executor" if has_plan else "scientist"
                update["decision"]["next_role"] = redirected_role
                update["decision"]["instruction"] = (
                    "Run another planned focusing experiment with settings not tried yet; "
                    "preserve the planned comparison controls and report its enhancement."
                    if has_plan else
                    f"Plan at least {MIN_EXPERIMENTS} focusing experiments with different "
                    "settings before the executor calls the tool."
                )
                update["dialogue"][0]["text"] = agents.RoutingDecision.model_validate(
                    update["decision"]
                ).model_dump_json()
                update["dialogue"].append({
                    "role": "system",
                    "text": (
                        f"Finish refused: only {count} successful focusing experiments "
                        f"are recorded; at least {MIN_EXPERIMENTS} are required. "
                        f"This decision was redirected to {redirected_role}."
                    ),
                })
        return update

    def route(state: State) -> str:
        if state["stop_reason"] != "running":
            return END
        return state["decision"]["next_role"]

    builder = StateGraph(State)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("scientist", lambda state: agents.scientist(state, llm))
    builder.add_node("general", lambda state: agents.general(state, llm))
    builder.add_node(
        "executor", lambda state: agents.executor(state, llm, coding_tool, focusing_tool)
    )

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route, {
        "scientist": "scientist",
        "general": "general", "executor": "executor", END: END,
    })
    for role in ("scientist", "general", "executor"):
        builder.add_edge(role, "orchestrator")

    return builder.compile()
