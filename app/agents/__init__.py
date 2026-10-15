"""Public agent callables, prompts, schemas, and executor-mode helpers.

Import callables directly from their owning modules inside application code to
avoid confusing a same-named submodule with a function.
"""

from app.agents.common import DEFAULT_INITIAL_DECISIONS, MIN_EXPERIMENTS
from app.agents.orchestrator import (
    DEFAULT_EXTRA_DECISIONS, ORCHESTRATOR_PROMPT, SUMMARY_PROMPT,
    RoutingDecision, orchestrator, summarize_progress,
)
from app.agents.scientist import (
    SCIENTIST_PROMPT, ExperimentSelection, ProposedExperiment, ScientistResponse,
    scientist,
)
from app.agents.executor import EXECUTOR_PROMPT, executor
from app.agents.report_and_print import (
    LOST_JOB_STATES, MAX_HARDWARE_PASSES, REPORT_AND_PRINT_PROMPT,
    create_information_tool, inspect_report, report_and_print,
)
from app.agents.general import GENERAL_PROMPT, general

__all__ = [
    "RoutingDecision", "ExperimentSelection", "ProposedExperiment", "ScientistResponse",
    "orchestrator", "scientist", "general", "summarize_progress", "executor",
    "create_information_tool", "inspect_report", "report_and_print",
    "MIN_EXPERIMENTS", "MAX_HARDWARE_PASSES", "DEFAULT_INITIAL_DECISIONS",
    "DEFAULT_EXTRA_DECISIONS", "LOST_JOB_STATES", "ORCHESTRATOR_PROMPT",
    "SCIENTIST_PROMPT", "GENERAL_PROMPT", "EXECUTOR_PROMPT",
    "REPORT_AND_PRINT_PROMPT", "SUMMARY_PROMPT",
]

from app.agents.photo_quality_specialist import (
    CandidateMeasurement,
    ImageAttachment,
    ImageLabel,
    MissingImage,
    PhotoQualitySpecialistError,
    SpecialistEvidence,
    SpecialistInputError,
    photo_quality_specialist,
    prepare_specialist_messages,
)

__all__ += [
    "CandidateMeasurement", "ImageAttachment", "ImageLabel", "MissingImage",
    "PhotoQualitySpecialistError", "SpecialistEvidence", "SpecialistInputError",
    "photo_quality_specialist", "prepare_specialist_messages",
]
