from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages


class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int
