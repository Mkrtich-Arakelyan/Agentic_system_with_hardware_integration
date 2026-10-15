"""Existing orchestration decisions and progress summaries."""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.agents.common import _messages


DEFAULT_EXTRA_DECISIONS = 5


ORCHESTRATOR_PROMPT = """
You coordinate autonomous focusing at the fixed camera target and the printed
one-page report of the results. Read the full dialogue and tool evidence. Return
next_role, a short instruction, and a one-sentence reason. Choose the role
needed now, not a fixed sequence:
- scientist: optical theory, experiment planning, analysis and the report spec;
- general: plain explanations, including the explanation of the report graph;
- executor: runs focusing experiments;
- report_and_print: the executor builds the report PDF and prints it after
  human approval.

The typical path is: scientist plan, at least the supplied minimum of successful
focusing experiments, scientist analysis with a REPORT SPEC, general explanation
of the chosen graph, then report_and_print. Get a scientist plan before
experiments and reuse it unless revision is needed. Only the executor runs
tools; you write no theory or code.

For fair comparisons, change one setting and hold the others fixed, including
search_seed, focus_size, background_size and iterations when varying macropixels.
A successful experiment has ok=true, finite enhancements, saved paths and no
validation_error; proposed experiments and supporting code do not count.

There is no finish choice. The run ends inside report_and_print, after the
report is printed or the human skips printing. If report_and_print came back
with "Needs information", read what is missing and the human's request, route
to the role that can supply it (scientist for analysis or verdict, executor for
a new experiment), then route back to report_and_print. If a system turn says
finishing was refused, route to the role that can supply what it names, then
back to report_and_print.

Human approval of printing is handled by the graph. Do not ask the human for
settings or approval yourself. Each orchestrator choice consumes one decision
from the current block; specialist replies and tool calls do not. Use the
remaining decisions wisely and get the report printed rather than counting on
extra decisions. Base decisions and conclusions on recorded results, never
invented measurements.

Use the user's scientific question to guide planning and reporting. Before
report_and_print, obtain a ready scientist selection of four experiments.
If the scientist requests more experiments, route to the executor using the
proposed settings, then back to the scientist for review. Four successes are
only a minimum, not proof that the evidence is sufficient. Respect the existing
decision budget. Once experiment_selection is frozen, reuse it for printing.
"""


SUMMARY_PROMPT = """
You are the orchestrator. The current block of orchestrator decisions is used up
and the human decides whether to grant more. Write at most 8 short lines for the
human: what has been done (experiments with their final enhancements, report
state, print state), what you plan next, and why you need more decisions.
Use only recorded results; never invent measurements.
"""


class RoutingDecision(BaseModel):
    """Required output of each orchestrator call."""

    next_role: Literal["scientist", "general", "executor", "report_and_print"]
    instruction: str = Field(min_length=1, description="Instruction for the chosen role")
    reason: str = Field(min_length=1, description="One-sentence reason for this choice")


def orchestrator(state: dict, llm: BaseChatModel) -> dict:
    decision = llm.with_structured_output(
        RoutingDecision, method="json_schema"
    ).invoke(_messages(state, ORCHESTRATOR_PROMPT, include_budget=True))
    return {
        "dialogue": [{"role": "orchestrator", "text": decision.model_dump_json()}],
        "decision": decision.model_dump(),
        "decision_count": state.get("decision_count", 0) + 1,
    }


def summarize_progress(state: dict, llm: BaseChatModel) -> str:
    """Progress summary for the budget review; does not spend a decision."""
    return llm.invoke(_messages(state, SUMMARY_PROMPT, include_budget=True)).text
