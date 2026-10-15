"""Existing agent definitions and prompts."""

from app.agents.definitions import (
    RoutingDecision,
    ExperimentSelection,
    ProposedExperiment,
    ScientistResponse,
    orchestrator,
    scientist,
    general,
    summarize_progress,
    executor,
    create_information_tool,
    inspect_report,
    report_and_print,
    MIN_EXPERIMENTS,
    MAX_HARDWARE_PASSES,
    DEFAULT_INITIAL_DECISIONS,
    DEFAULT_EXTRA_DECISIONS,
    LOST_JOB_STATES,
    ORCHESTRATOR_PROMPT,
    SCIENTIST_PROMPT,
    GENERAL_PROMPT,
    EXECUTOR_PROMPT,
    REPORT_AND_PRINT_PROMPT,
    SUMMARY_PROMPT,
)

__all__ = ['RoutingDecision', 'ExperimentSelection', 'ProposedExperiment', 'ScientistResponse', 'orchestrator', 'scientist', 'general', 'summarize_progress', 'executor', 'create_information_tool', 'inspect_report', 'report_and_print', 'MIN_EXPERIMENTS', 'MAX_HARDWARE_PASSES', 'DEFAULT_INITIAL_DECISIONS', 'DEFAULT_EXTRA_DECISIONS', 'LOST_JOB_STATES', 'ORCHESTRATOR_PROMPT', 'SCIENTIST_PROMPT', 'GENERAL_PROMPT', 'EXECUTOR_PROMPT', 'REPORT_AND_PRINT_PROMPT', 'SUMMARY_PROMPT']
