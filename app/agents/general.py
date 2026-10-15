"""Existing general explanation turn."""

from langchain_core.language_models.chat_models import BaseChatModel

from app.agents.common import _messages


GENERAL_PROMPT = """
Follow the orchestrator's instruction. Explain supporting optical or programming
concepts briefly and clearly. Leave experiment design to the scientist and code
and execution to the executor. Use recorded evidence; never invent measurements.
When asked for the report explanation, write 2-3 plain sentences that explain
what the graph in the latest REPORT SPEC shows and how to read it, for a reader
who is not an optics expert. Add no new numbers.
"""


def general(state: dict, llm: BaseChatModel) -> dict:
    response = llm.invoke(_messages(state, GENERAL_PROMPT))
    return {"dialogue": [{"role": "general", "text": response.text}]}
