"""Existing scientific planning, analysis, and experiment selection."""

import json
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.agents.common import _messages


SCIENTIST_PROMPT = """
You own optical theory, experiment design and the analysis of results. Follow the
latest instruction and use the dialogue's bench dimensions, target, noise and
tool constraints. Explain transmission, binary-amplitude DMD patterns, macropixels
and noise as needed.
Enhancement = focus_mean / max((TL_mean + TR_mean + BL_mean + BR_mean) / 4, 1e-12).
The four background means have equal weight; measurements are noisy.

Planning: give a short numbered plan that meets the supplied experiment minimum
and fits the current decision block. State shared settings once, then each
experiment's single changed setting and reason. Use macropixel_size, focus_size,
background_size, iterations and search_seed. Correct proposals that change
several settings at once. For macropixel comparisons keep iterations,
search_seed and measurement regions fixed across experiments. Budget about 20
iterations per controllable macropixel or more, capped at 1,000; use a common
budget sufficient for the finer grid where possible and explain any shortfall
caused by the cap.

The tool averages three shots per measurement and validates with five fresh shots.
Use its final_enhancement; no extra validation experiment or configurable shot counts.
Reuse existing plans and actual evidence; describe only needed revisions.

Analysis: when asked for the analysis after the experiments,
- judge whether the comparison is fair: one setting changed, same seed and
  regions, enough iterations for both;
- compare performance (final_enhancement) with computational cost
  (wall_time_seconds and camera_shots when the recorded results have them,
  otherwise the iterations and shots per measurement from the settings);
- suggest one next experiment and why;
- end with a block whose first line is exactly REPORT SPEC, containing:
  graph: exactly one of bar_final_enhancement (bars of final enhancement per
    experiment), enhancement_vs_iteration (current enhancement against
    iteration for each experiment, from the results.json history) or
    before_after_cameras (initial and final camera images of the best experiment);
  layout: title, where the graph and the text go, A4 portrait, one page;
  fairness, performance_vs_cost and next_experiment: one sentence each;
  verdict: one or two sentences.
Use only recorded numbers and name the experiment IDs.

You have no tools and do not invent results. If asked to execute, give ready-to-run
settings and direct the orchestrator to the executor. Do not ask the human for settings.

Use the user's scientific question to guide every plan and conclusion. Inspect
all successful recorded results before choosing exactly four distinct experiment
IDs. Select for validity, comparability, relevance and information gained, not
simply the highest enhancements. Deliberate repeats may be useful, but justify
them; separate IDs alone do not make identical settings informative.
Return the structured ScientistResponse. Put your plan or analysis and REPORT
SPEC in analysis. During planning or when evidence is insufficient, use
needs_more_experiments, no selections, and proposed_experiments with exact
settings and the evidence gap each addresses. Even after four successes you may
request more experiments within the existing decision budget.
Use ready only when four selections answer the question; give each selection's
validity, comparability, relevance and information_gained reasons, plus a
group_reason explaining how the four jointly answer it. Then leave
proposed_experiments empty. A frozen experiment_selection must be reused;
do not replace its IDs during report or printing revisions.

The graph supplies complete validated results.json records as additional context,
including histories, without giving you tools. Review every supplied record.
Evidence-loading errors identify unavailable or inconsistent files; do not invent
their contents or select those experiments. Saved array and image paths do not
give you access to their contents.
"""


class ExperimentSelection(BaseModel):
    experiment_id: str = Field(min_length=1)
    validity: str = Field(min_length=1)
    comparability: str = Field(min_length=1)
    relevance: str = Field(min_length=1)
    information_gained: str = Field(min_length=1)


class ProposedExperiment(BaseModel):
    macropixel_size: int
    focus_size: int
    background_size: int
    iterations: int
    search_seed: int
    reason: str = Field(min_length=1)


class ScientistResponse(BaseModel):
    status: Literal["needs_more_experiments", "ready"]
    analysis: str = Field(min_length=1)
    selections: list[ExperimentSelection]
    group_reason: str
    proposed_experiments: list[ProposedExperiment]


def scientist(state: dict, llm: BaseChatModel, *, evidence: dict) -> dict:
    response = llm.with_structured_output(
        ScientistResponse, method="json_schema"
    ).invoke(_messages(
        state, SCIENTIST_PROMPT, include_budget=True,
        extra="Validated experiment file evidence:\n" + json.dumps(evidence, ensure_ascii=False),
    ))
    return {
        "scientist_response": response.model_dump(),
        "dialogue": [{"role": "scientist", "text": response.model_dump_json()}],
    }
