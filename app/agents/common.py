"""Shared agent message construction, tool-result summaries, and budget context."""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


MIN_EXPERIMENTS = 4


DEFAULT_INITIAL_DECISIONS = 10


def _messages(
    state: dict, prompt: str, *, include_budget: bool = False, extra: str = ""
) -> list:
    """Keep agent names intact without pretending they are API message roles."""
    dialogue = json.dumps(state["dialogue"], ensure_ascii=False)
    evidence = json.dumps(state.get("execution_results", []), ensure_ascii=False)
    content = "Dialogue:\n" + dialogue + "\n\nRecorded tool results:\n" + evidence
    content += "\n\nScientific review and frozen selection:\n" + json.dumps({
        "scientist_response": state.get("scientist_response"),
        "experiment_selection": state.get("experiment_selection"),
    }, ensure_ascii=False)
    if include_budget:
        used = state.get("decision_count", 0)
        budget_limit = state.get("decision_budget", DEFAULT_INITIAL_DECISIONS)
        budget = {
            "orchestrator_decisions_used": used,
            "current_decision_budget": budget_limit,
            "decisions_remaining_in_this_block": max(0, budget_limit - used),
            "note": (
                "When this block is used up the human is shown a summary and may "
                "grant more decisions; more are not guaranteed."
            ),
            "minimum_successful_focusing_experiments": MIN_EXPERIMENTS,
        }
        content = "Decision budget:\n" + json.dumps(budget) + "\n\n" + content
    if extra:
        content += "\n\n" + extra
    return [SystemMessage(content=prompt), HumanMessage(content=content)]


def _summarize(llm: BaseChatModel, tools: list, messages: list) -> str:
    """Ask for a short summary with tools declared but disabled."""
    try:
        # Keep the tool declarations, but disable further calls.
        return llm.bind_tools(tools, tool_choice="none").invoke(messages).text
    except Exception as error:
        # A summary failure must not discard a completed tool result.
        return f"Model summary unavailable ({type(error).__name__}); see the raw tool result."
