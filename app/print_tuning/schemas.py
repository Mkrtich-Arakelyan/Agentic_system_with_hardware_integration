"""Structured specialist responses for the three print-tuning decision stages.

Only gamma is tuned, in all three rounds: propose_gamma (before round 1),
review_gamma (after the round 1 and round 2 scans: winners plus next tests) and
review_gamma_final (after the round 3 scan: final winners only). Gray levels and
steps are fixed and never appear here.

Contextual checks live in app.print_tuning.validation: frozen experiment IDs,
active stage/round, candidate counts, configuration ranges, duplicate tests and
references, winner membership, and assessment rules. No contextual checks or
cross-field validation are implemented in these schemas.

For usable reviews, supply all applicable winners and next-round proposals.
For needs_rescan or insufficient_contrast, issues must explain the problem and
all winner/proposal lists must be empty.
An effective tie is usable: reference may win with a tie_explanation.
Nullable fields are required; callers must explicitly supply null when unused.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


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


class ExperimentGammaTests(TuningResponseModel):
    experiment_id: NonblankText
    tests: list[GammaTest]


class ExperimentWinner(TuningResponseModel):
    experiment_id: NonblankText
    candidate_id: WinnerCandidateId
    reason: NonblankText
    confidence: Confidence
    tie_explanation: str | None


class GammaProposalResponse(TuningResponseModel):
    stage: Literal["propose_gamma"]
    experiments: list[ExperimentGammaTests]


class GammaReviewResponse(TuningResponseModel):
    """After the round 1 and round 2 scans.

    Usable: one gamma winner per experiment and 5 next gamma tests per experiment.
    Other assessments: explain issues and leave both lists empty.
    """

    stage: Literal["review_gamma"]
    assessment: Assessment
    issues: list[NonblankText]
    gamma_winners: list[ExperimentWinner]
    gamma_tests: list[ExperimentGammaTests]


class GammaFinalReviewResponse(TuningResponseModel):
    """After the round 3 scan: the final decision, with no further proposals.

    Usable: one final gamma winner per experiment. Other assessments: explain
    issues and leave gamma_winners empty.
    """

    stage: Literal["review_gamma_final"]
    assessment: Assessment
    issues: list[NonblankText]
    gamma_winners: list[ExperimentWinner]
