from langgraph.graph import StateGraph, START

from .state import State
from .nodes import (
    writer_node,
    tool_node,
    extract_draft_node,
    reviewer_node,
    should_use_tool,
    should_stop_looping,
)


def build_graph():
    graph = StateGraph(State)

    graph.add_node("writer", writer_node)
    graph.add_node("tools", tool_node)
    graph.add_node("extract_draft", extract_draft_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "writer")
    graph.add_conditional_edges("writer", should_use_tool)
    graph.add_edge("tools", "writer")  # loop back so writer can use tool results
    graph.add_edge("extract_draft", "reviewer")
    graph.add_conditional_edges("reviewer", should_stop_looping)

    return graph.compile()
