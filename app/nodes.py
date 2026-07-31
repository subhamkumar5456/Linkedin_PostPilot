import time

from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from .state import State
from .llms import writer_llm_with_tools, reviewer_llm, tools
from .prompts import WRITER_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT
from .config import MAX_ATTEMPTS
from tracking.metrics import get_current_run, get_usage, looks_like_refusal

tool_node = ToolNode(tools)


def writer_node(state: State) -> dict:
    """Writes (or rewrites) the draft, or continues after a tool call."""
    metrics = get_current_run()
    messages_so_far = state.get("messages", [])

    # Continuing a tool-calling turn: the last message is a tool result,
    # so we hand the whole running history back to the LLM instead of
    # re-building a fresh prompt (fix for the original tools->reviewer
    # dead-end, which skipped extract_draft entirely).
    if messages_so_far and getattr(messages_so_far[-1], "type", None) == "tool":
        t0 = time.perf_counter()
        response = writer_llm_with_tools.invoke(
            [("system", WRITER_SYSTEM_PROMPT)] + messages_so_far
        )
        latency = time.perf_counter() - t0
        metrics.log_call(
            "writer_after_tools", state.get("attempt", 1), "gemini-writer",
            latency, get_usage(response),
        )
        if looks_like_refusal(response.content):
            metrics.refusal_flagged = True
        return {"messages": [response]}

    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state.get("review_feedback", "")

    if attempt == 1:
        user_message = (
            f"Write a linkedin Post on this topic : {topic} "
            f"if you need the current info search the web first"
        )
    else:
        user_message = (
            f"your previous post was rejected "
            f"here is the reviewer's feedback \n\n {previous_feedback}"
            f"write a new, improvised draft that fixes every issue mentioned "
            f"do not repeat the same mistake"
        )

    messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]

    t0 = time.perf_counter()
    response = writer_llm_with_tools.invoke(messages)
    latency = time.perf_counter() - t0
    metrics.log_call("writer", attempt, "gemini-writer", latency, get_usage(response))
    if looks_like_refusal(response.content):
        metrics.refusal_flagged = True

    return {
        "messages": [("human", user_message), response],
        "attempt": attempt,
    }


def extract_draft_node(state: State) -> dict:
    """After the writer llm finishes tool calls, pulls the final text out."""
    last_message = state["messages"][-1]
    draft = last_message.content
    print(f"\n\nGenerated Post:\n{draft}\n")
    return {"draft": draft}


def reviewer_node(state: State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    metrics = get_current_run()
    draft = state["draft"]

    prompt = f"review this linkedin post draft\n{draft}\ngive your reviews"

    t0 = time.perf_counter()
    response = reviewer_llm.invoke(
        [("system", REVIEWER_SYSTEM_PROMPT), ("human", prompt)]
    )
    latency = time.perf_counter() - t0
    usage = get_usage(response)

    review_text = response.content.strip()
    is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

    if "FEEDBACK" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"

    metrics.log_call(
        "reviewer", state.get("attempt", 1), "groq-reviewer", latency, usage,
        extra={"verdict": verdict},
    )
    if looks_like_refusal(feedback):
        metrics.refusal_flagged = True

    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved,
    }


def should_use_tool(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "extract_draft"


def should_stop_looping(state: State):
    if state["is_approved"]:
        print("Post has been approved")
        return END
    if state["attempt"] >= MAX_ATTEMPTS:
        print("Reached max attempts")
        return END
    return "writer"
