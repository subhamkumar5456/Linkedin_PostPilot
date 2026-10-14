from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from .state import State
from .llms import writer_llm_with_tools, reviewer_llm, tools
from .prompts import WRITER_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT
from .config import MAX_ATTEMPTS
from tracking.metrics import get_current_run
from tracking.db import db_cursor

tool_node = ToolNode(tools)


def writer_node(state: State) -> dict:
    """Writes (or rewrites) the draft and can call tavily tool for web search first."""
    messages_so_far = state.get("messages", [])

    # Continuing after a tool call: hand the full running history back to
    # the LLM so it can write the post using the search results it fetched.
    if messages_so_far and getattr(messages_so_far[-1], "type", None) == "tool":
        response = writer_llm_with_tools.invoke(
            [("system", WRITER_SYSTEM_PROMPT)] + messages_so_far,
            _node="writer_after_tools",
            _attempt=state.get("attempt", 1),
        )
        return {"messages": [response]}

    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state.get("review_feedback", "")

    if attempt == 1:
        user_message = (
            f"Write a linkedin Post on this topic : {topic}"
            f"if you need the current info search the web first"
        )
    else:
        user_message = (
            f"your previous post was rejected "
            f"here is the reviewer's feedback \n\n {previous_feedback}"
            f"write a new, impovised draft that fixes every issue mentioned "
            f"do not repeat the same mistake"
        )

    messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]

    response = writer_llm_with_tools.invoke(
        messages,
        _node="writer",
        _attempt=attempt,
    )

    return {
        "messages": [("human", user_message), response],
        "attempt": attempt,
    }


def extract_draft_node(state: State) -> dict:
    """After the writer llm finishes tool calls, pulls the final text out of graph."""
    last_message = state["messages"][-1]
    draft = last_message.content
    print(f"\n\n generated Post : \n{draft}\n")
    return {"draft": draft}


def reviewer_node(state: State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft = state["draft"]

    prompt = (
        f"review this linkedin post draft"
        f"{draft}\n"
        f"give your reviews"
    )

    response = reviewer_llm.invoke(
        [("system", REVIEWER_SYSTEM_PROMPT), ("human", prompt)],
        _attempt=state.get("attempt", 1),
    )

    review_text = response.content.strip()
    is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

    if "FEEDBACK" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"

    # Back-fill verdict into the call entry that TrackedLLM just recorded.
    # The verdict isn't known until after the response, so we patch the last
    # recorded call in-memory and update the DB row in one targeted write.
    try:
        run = get_current_run()
        if run.calls:
            run.calls[-1]["verdict"] = verdict
            last_call = run.calls[-1]
            with db_cursor() as cur:
                # SQLite doesn't support ORDER BY/LIMIT in UPDATE directly;
                # use a subquery to target only the most recent matching row.
                cur.execute(
                    """
                    UPDATE llm_calls SET verdict = ?
                    WHERE id = (
                        SELECT id FROM llm_calls
                        WHERE run_id = ? AND node = ? AND attempt = ?
                        ORDER BY id DESC LIMIT 1
                    )
                    """,
                    (verdict, last_call["run_id"], last_call["node"], last_call["attempt"]),
                )
    except RuntimeError:
        pass  # no active run — non-fatal

    print(f"[Verdict:{verdict}]")
    print(f"[Feedback:{feedback}]")

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
        print("Post has been ✅ approved")
        return END
    if state["attempt"] >= MAX_ATTEMPTS:
        print("Reached max attempts")
        return END
    return "writer"
