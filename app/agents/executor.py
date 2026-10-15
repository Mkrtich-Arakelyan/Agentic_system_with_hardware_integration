"""Existing experiment-execution turn and its focusing tool binding."""

import json
import math

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from app.agents.common import _messages, _summarize


EXECUTOR_PROMPT = """
You run the focusing experiments. Follow the latest instruction and existing
scientist plan; if no plan exists, request one without running an experiment.
Make exactly one run_focusing_experiment call per turn and use the planned
settings exactly, including search_seed.

The focusing tool averages three shots and validates with five fresh shots.
Its final_enhancement is ready to report; another call starts a new search,
not an extra validation. Follow the metric supplied in the bench context.
After the tool call, briefly report measured enhancements and saved paths
(and wall_time_seconds and camera_shots when the result has them), or its
actual error. A validation_error means invalid evidence.
Do not invent results, repeat the plan or request a second tool call.
"""


def executor(state: dict, llm: BaseChatModel, focusing_tool: BaseTool) -> dict:
    """Run one focusing experiment and retain its result as evidence."""
    messages = _messages(state, EXECUTOR_PROMPT)
    request = llm.bind_tools([focusing_tool], tool_choice="any").invoke(messages)
    if len(request.tool_calls) != 1:
        return {"dialogue": [{
            "role": "executor",
            "text": "No tool ran: expected exactly one tool call. Please retry."
        }]}

    call = request.tool_calls[0]
    if call["name"] != focusing_tool.name:
        return {"dialogue": [{
            "role": "executor", "text": "No tool ran: the requested tool is unavailable."
        }]}
    if not any(turn["role"] == "scientist" for turn in state["dialogue"]):
        return {"dialogue": [{
            "role": "executor", "text": "No tool ran: I need a scientist plan before focusing."
        }]}

    try:
        result = focusing_tool.invoke(call["args"])
    except Exception as error:
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"}

    validation_error = None
    try:
        if not isinstance(result, dict):
            raise ValueError("The tool must return a dictionary.")
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
    summary = _summarize(llm, [focusing_tool], messages)
    if validation_error:
        summary += "\nResult validation error: " + validation_error
    return {
        "dialogue": [{
            "role": "executor",
            "text": summary + f"\n\nTool: {call['name']}\nRaw tool result:\n" + evidence,
        }],
        "execution_results": [recorded],
    }
