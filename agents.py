import json
import math
import traceback
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


MAX_DECISIONS = 10
MIN_EXPERIMENTS = 2


ORCHESTRATOR_PROMPT = """
You coordinate autonomous focusing at the fixed camera target.
Read the full dialogue and tool evidence. Return next_role, a short instruction,
and a one-sentence reason. Choose the role needed now, not a fixed sequence:
- scientist: optical theory and experiment planning;
- general: supporting concepts;
- executor: experiments and supporting code;
- finish: report completed work.

Get a scientist plan before experiments and reuse it unless revision is needed.
Only the executor runs tools; you write no theory or code. Choose parameters
through the agents without asking the user for settings or approval.
For fair comparisons, change one setting and hold the others fixed, including
search_seed, focus_size, background_size and iterations when varying macropixels.

Use the supplied decision budget. Each orchestrator choice consumes one decision;
specialist replies and tool calls do not. Remaining decisions include this turn.
Leave a decision for finish. If only one remains and the experiment minimum is
met, finish. Never finish before the supplied minimum successful experiments
with different settings: require ok=true, finite enhancements, saved paths and
no validation_error. Proposed experiments and supporting code do not count.

Use final_enhancement, already validated by the tool; do not run another search
just for validation. Report the best experiment's ID, settings, enhancement and
output files. If measurement regions differ, explain that the metrics differ.
Base decisions and conclusions on recorded results, never invented measurements.
"""

SCIENTIST_PROMPT = """
You own optical theory and experiment design. Follow the latest instruction and
use the dialogue's bench dimensions, target, noise and tool constraints.
Explain transmission, binary-amplitude DMD patterns, macropixels and noise as needed.
Enhancement = focus_mean / max((TL_mean + TR_mean + BL_mean + BR_mean) / 4, 1e-12).
The four background means have equal weight; measurements are noisy.

Give a short numbered plan that meets the supplied experiment minimum and leaves
a decision for finish. State shared settings once, then each experiment's single
changed setting and reason. Use macropixel_size, focus_size, background_size,
iterations and search_seed. Correct proposals that change several settings at once.
For macropixel comparisons keep iterations, search_seed and measurement regions
fixed across experiments. Budget about 20 iterations per controllable macropixel
or more, capped at 1,000; use a common budget sufficient for the finer grid where
possible and explain any shortfall caused by the cap.

The tool averages three shots per measurement and validates with five fresh shots.
Use its final_enhancement; no extra validation experiment or configurable shot counts.
Reuse existing plans and actual evidence; describe only needed revisions.
You have no tools and do not invent results. If asked to execute, give ready-to-run
settings and direct the orchestrator to the executor. Do not ask for approval.
"""

GENERAL_PROMPT = """
Follow the orchestrator's instruction. Explain supporting optical or programming
concepts briefly and clearly. Leave experiment design to the scientist and code
and execution to the executor. Use recorded evidence; never invent measurements.
"""

EXECUTOR_PROMPT = """
You are the only role with tools. Follow the latest instruction and existing
scientist plan; if no plan exists, request one without running an experiment.
Make exactly one tool call per turn:
- run_focusing_experiment: use planned settings exactly, including search_seed;
- write_and_run: only explicitly requested supporting code, with a complete script
  and fresh filename. Do not recreate the bench or implement another search.

The focusing tool averages three shots and validates with five fresh shots.
Its final_enhancement is ready to report; another call starts a new search,
not an extra validation. Follow the metric supplied in the bench context.
After the tool call, briefly report measured enhancements and saved paths, or
its actual error/traceback/timeout. A validation_error means invalid evidence.
Do not invent results, repeat the plan or request a second tool call.
"""


class RoutingDecision(BaseModel):
    """Required output of each orchestrator call."""

    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(min_length=1, description="Instruction for the chosen role")
    reason: str = Field(min_length=1, description="One-sentence reason for this choice")


def _messages(state: dict, prompt: str, *, include_budget: bool = False) -> list:
    """Keep agent names intact without pretending they are API message roles."""
    dialogue = json.dumps(state["dialogue"], ensure_ascii=False)
    evidence = json.dumps(state.get("execution_results", []), ensure_ascii=False)
    content = "Dialogue:\n" + dialogue + "\n\nRecorded tool results:\n" + evidence
    if include_budget:
        used = state.get("decision_count", 0)
        budget = {
            "orchestrator_decision_limit": MAX_DECISIONS,
            "orchestrator_decisions_used": used,
            "orchestrator_decisions_remaining": max(0, MAX_DECISIONS - used),
            "minimum_successful_focusing_experiments": MIN_EXPERIMENTS,
        }
        content = "Decision budget:\n" + json.dumps(budget) + "\n\n" + content
    return [SystemMessage(content=prompt), HumanMessage(content=content)]


def orchestrator(state: dict, llm: BaseChatModel) -> dict:
    decision = llm.with_structured_output(
        RoutingDecision, method="json_schema"
    ).invoke(_messages(state, ORCHESTRATOR_PROMPT, include_budget=True))
    return {
        "dialogue": [{"role": "orchestrator", "text": decision.model_dump_json()}],
        "decision": decision.model_dump(),
        "decision_count": state.get("decision_count", 0) + 1,
    }


def scientist(state: dict, llm: BaseChatModel) -> dict:
    response = llm.invoke(_messages(state, SCIENTIST_PROMPT, include_budget=True))
    return {"dialogue": [{"role": "scientist", "text": response.text}]}


def general(state: dict, llm: BaseChatModel) -> dict:
    response = llm.invoke(_messages(state, GENERAL_PROMPT))
    return {"dialogue": [{"role": "general", "text": response.text}]}


def executor(
    state: dict,
    llm: BaseChatModel,
    coding_tool: BaseTool,
    focusing_tool: BaseTool,
) -> dict:
    """Dispatch one requested tool call and retain its result as evidence."""
    tool_map = {coding_tool.name: coding_tool, focusing_tool.name: focusing_tool}
    available_tools = list(tool_map.values())
    messages = _messages(state, EXECUTOR_PROMPT)
    request = llm.bind_tools(available_tools, tool_choice="any").invoke(messages)
    if len(request.tool_calls) != 1:
        return {"dialogue": [{
            "role": "executor",
            "text": "No tool ran: expected exactly one tool call. Please retry."
        }]}

    call = request.tool_calls[0]
    if call["name"] not in tool_map:
        return {"dialogue": [{
            "role": "executor", "text": "No tool ran: the requested tool is unavailable."
        }]}
    is_focusing = call["name"] == focusing_tool.name
    if is_focusing and not any(turn["role"] == "scientist" for turn in state["dialogue"]):
        return {"dialogue": [{
            "role": "executor", "text": "No tool ran: I need a scientist plan before focusing."
        }]}

    try:
        result = tool_map[call["name"]].invoke(call["args"])
    except Exception as error:
        if is_focusing:
            result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        else:
            result = {"exit_code": None, "stdout": "", "stderr": traceback.format_exc()}

    validation_error = None
    try:
        if not isinstance(result, dict):
            raise ValueError("The tool must return a dictionary.")
        if not is_focusing:
            if not {"exit_code", "stdout", "stderr"} <= result.keys():
                raise ValueError("Coding tool must return exit_code, stdout, and stderr.")
        else:
            if not isinstance(result.get("ok"), bool):
                raise ValueError("Focusing tool must return a boolean ok.")
            if result["ok"]:
                required = {
                    "experiment_id", "settings", "initial_enhancement",
                    "final_enhancement", "paths",
                }
                if not required <= result.keys():
                    raise ValueError("Successful focusing result is missing required fields.")
                if not isinstance(result["experiment_id"], str) or not result["experiment_id"]:
                    raise ValueError("A successful experiment needs an experiment ID.")
                if not isinstance(result["settings"], dict) or not result["settings"]:
                    raise ValueError("A successful experiment needs its settings.")
                paths = result["paths"]
                if (
                    not isinstance(paths, dict)
                    or not {"initial_camera", "final_camera", "final_pattern",
                            "results_json", "experiments_csv"} <= paths.keys()
                    or not all(isinstance(path, str) and path for path in paths.values())
                ):
                    raise ValueError("A successful experiment needs saved output paths.")
                for name in ("initial_enhancement", "final_enhancement"):
                    value = result[name]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                    ):
                        raise ValueError(f"{name} must be a finite number.")
            elif not isinstance(result.get("error"), str) or not result["error"]:
                raise ValueError("A failed focusing result needs an error message.")
    except (ValueError, OverflowError) as error:
        validation_error = str(error)

    recorded = {"tool_name": call["name"], "tool_args": call["args"], "result": result}
    if validation_error:
        recorded["validation_error"] = validation_error
    evidence = json.dumps(result, ensure_ascii=False)
    # Preserve the original AIMessage, including its encrypted reasoning content.
    tool_message = evidence
    if validation_error:
        tool_message += "\nResult validation error: " + validation_error
    messages.extend([request, ToolMessage(content=tool_message, tool_call_id=call["id"])])
    try:
        # Keep both tool declarations, but disable further calls.
        response = llm.bind_tools(available_tools, tool_choice="none").invoke(messages)
        summary = response.text
    except Exception as error:
        # A summary failure must not discard a completed tool result.
        summary = f"Model summary unavailable ({type(error).__name__}); see the raw tool result."
    if validation_error:
        summary += "\nResult validation error: " + validation_error
    return {
        "dialogue": [{
            "role": "executor",
            "text": summary + f"\n\nTool: {call['name']}\nRaw tool result:\n" + evidence,
        }],
        "execution_results": [recorded],
    }
