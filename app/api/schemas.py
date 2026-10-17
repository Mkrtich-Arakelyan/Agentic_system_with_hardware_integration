"""HTTP inputs only; workflow decisions remain in the graph."""
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: str = Field(min_length=1, max_length=20000)
    model: str | None = None
    specialist_model: str | None = None
    printer: str | None = None
    fake_printer: bool = True
    paper_color: str | None = Field(default=None, max_length=40)
    initial_decisions: int | None = Field(default=None, ge=1, le=1000)
    extra_decisions: int | None = Field(default=None, ge=1, le=1000)
    max_tuning_cycles: int = Field(default=3, ge=1, le=20)

    @field_validator("request")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("request must not be blank")
        return value


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interrupt_token: str = Field(min_length=1)
    answer: str | dict


class ResumeRequest(BaseModel):
    recover_stopped_tuning: bool = False
    report_revision: str | None = Field(default=None, min_length=1, max_length=20000)
