"""Structured specialist responses for the four print-tuning decision stages.

Future contextual validation must check the four frozen experiment IDs, active
stage/round, five correctly identified tests, configuration ranges/allowed values,
duplicate tests and references, endpoint ordering, shared endpoint consistency,
and winner membership in the reviewed manifest/round. It must also enforce the
assessment conditions below and the absence of proposals after the final review.
No contextual checks or cross-field validation are implemented here.

For usable reviews, supply all applicable winners and next-round proposals.
For needs_rescan or insufficient_contrast, issues must explain the problem,
all winner/proposal lists must be empty, and endpoint_winner must be None.
An effective tie is usable: reference may win with a tie_explanation.
Nullable fields are required; callers must explicitly supply null when unused.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt


TestCandidateId = Literal["test_1", "test_2", "test_3", "test_4", "test_5"]
WinnerCandidateId = Literal["reference", "test_1", "test_2", "test_3", "test_4", "test_5"]
Assessment = Literal["usable", "needs_rescan", "insufficient_contrast"]
Confidence = Literal["low", "medium", "high"]
NonblankText = Annotated[str, Field(min_length=1, pattern=r"\S")]
FiniteNumber = Annotated[FiniteFloat, Field(strict=True)]


class TuningResponseModel(BaseModel):
    """Common closed-object policy for responses and their nested objects."""

    model_config = ConfigDict(extra="forbid")


class GammaTest(TuningResponseModel):
    candidate_id: TestCandidateId
    gamma: FiniteNumber
    reason: NonblankText


class EndpointTest(TuningResponseModel):
    """Renderer endpoints: dimensionless gray, 0 black and 1 white.

    Configuration bounds and minimum gap are checked contextually later.
    """

    candidate_id: TestCandidateId
    dark_level: FiniteNumber
    light_level: FiniteNumber
    reason: NonblankText


class StepsTest(TuningResponseModel):
    """Discrete gray-band count; the renderer uses 0 for smooth rendering."""

    candidate_id: TestCandidateId
    steps: StrictInt
    reason: NonblankText


class ExperimentGammaTests(TuningResponseModel):
    experiment_id: NonblankText
    tests: list[GammaTest]


class ExperimentStepsTests(TuningResponseModel):
    experiment_id: NonblankText
    tests: list[StepsTest]


class ExperimentWinner(TuningResponseModel):
    experiment_id: NonblankText
    candidate_id: WinnerCandidateId
    reason: NonblankText
    confidence: Confidence
    tie_explanation: str | None


class SharedEndpointWinner(TuningResponseModel):
    candidate_id: WinnerCandidateId
    reason: NonblankText
    confidence: Confidence
    tie_explanation: str | None


class GammaProposalResponse(TuningResponseModel):
    stage: Literal["propose_gamma"]
    experiments: list[ExperimentGammaTests]


class GammaReviewResponse(TuningResponseModel):
    """Usable: gamma winners and shared endpoint tests are required in workflow.

    Other assessments: explain issues and leave both lists empty.
    """

    stage: Literal["review_gamma"]
    assessment: Assessment
    issues: list[NonblankText]
    gamma_winners: list[ExperimentWinner]
    endpoint_tests: list[EndpointTest]


class GrayLevelsReviewResponse(TuningResponseModel):
    """Usable: supply the shared endpoint winner and per-experiment step tests.

    Other assessments: explain issues, use null winner and empty step_tests.
    """

    stage: Literal["review_gray_levels"]
    assessment: Assessment
    issues: list[NonblankText]
    endpoint_winner: SharedEndpointWinner | None
    step_tests: list[ExperimentStepsTests]


class StepsReviewResponse(TuningResponseModel):
    """Final decision, with no further proposals.

    Usable: supply per-experiment winners. Other assessments: explain issues
    and leave step_winners empty.
    """

    stage: Literal["review_steps"]
    assessment: Assessment
    issues: list[NonblankText]
    step_winners: list[ExperimentWinner]
