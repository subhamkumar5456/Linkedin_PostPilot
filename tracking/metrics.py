"""
tracking/metrics.py

Latency / token / cost / approval / refusal tracking, persisted to the
SQLite tables defined in tracking/db.py.

Usage from a node function:

    from tracking.metrics import get_current_run, get_usage, looks_like_refusal

    t0 = time.perf_counter()
    response = some_llm.invoke(messages)
    latency = time.perf_counter() - t0

    get_current_run().log_call(
        node="writer", attempt=1, model_key="gemini-writer",
        latency_s=latency, usage=get_usage(response),
    )

Usage from the entrypoint (once per run):

    from tracking.metrics import RunMetrics, set_current_run

    run_metrics = RunMetrics()
    set_current_run(run_metrics)
    ... invoke the graph ...
    run_metrics.save(topic=..., attempts_used=..., approved=...)
"""

import contextvars
from datetime import datetime

from .db import db_cursor, fetch_recent_runs, fetch_calls_for_run, overall_stats  # noqa: F401

# PLACEHOLDER pricing — replace with the current published $/1M-token rates
# for whichever models you're actually billed for before trusting the cost
# numbers. Keys are your own labels, matched by whatever model_key you pass
# to log_call().
PRICING_PER_MILLION_TOKENS = {
    "gemini-writer": {"input": 0.0, "output": 0.0},   # <- fill in real numbers
    "groq-reviewer": {"input": 0.0, "output": 0.0},   # <- fill in real numbers
}

REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i'm sorry, but", "i am unable",
    "as an ai", "i won't be able to", "i do not feel comfortable",
    "i'm not able to",
]

# ---------------------------------------------------------------------------
# Active-run context — lets any node function reach the current run's
# RunMetrics without passing it through LangGraph state. contextvars (rather
# than a plain module-level global) keeps this safe if you ever run graphs
# concurrently (e.g. an async API serving multiple requests at once).
# ---------------------------------------------------------------------------

_current_run: contextvars.ContextVar = contextvars.ContextVar(
    "current_run", default=None
)


def set_current_run(run_metrics: "RunMetrics") -> None:
    _current_run.set(run_metrics)


def get_current_run() -> "RunMetrics":
    m = _current_run.get()
    if m is None:
        raise RuntimeError(
            "No active RunMetrics — call set_current_run(RunMetrics()) "
            "before invoking the graph."
        )
    return m


def looks_like_refusal(text) -> bool:
    """Cheap keyword heuristic — catches obvious refusals/non-answers,
    not a real classifier.

    Handles both plain str and list-of-parts (returned by models when
    they emit tool calls alongside text content).
    """
    if not text:
        return False
    # When the model makes a tool call, content is a list of dicts:
    # [{'type': 'text', 'text': '...'}, {'type': 'tool_use', ...}]
    if isinstance(text, list):
        text = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in text
        )
    lowered = text.lower()
    return any(p in lowered for p in REFUSAL_PATTERNS)


def get_usage(response) -> dict:
    """Works across providers — most LangChain chat models (including
    ChatGoogleGenerativeAI and ChatGroq) populate usage_metadata on the
    returned AIMessage."""
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def estimate_cost(model_key: str, usage: dict) -> float:
    rates = PRICING_PER_MILLION_TOKENS.get(model_key, {"input": 0, "output": 0})
    return (
        usage["input_tokens"] / 1_000_000 * rates["input"]
        + usage["output_tokens"] / 1_000_000 * rates["output"]
    )


class RunMetrics:
    """One instance per agent run (one graph.invoke() call)."""

    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.calls: list = []
        self.refusal_flagged = False

    def log_call(self, node, attempt, model_key, latency_s, usage, extra=None):
        extra = extra or {}
        error = extra.get("error")
        entry = {
            "run_id": self.run_id,
            "node": node,
            "attempt": attempt,
            "model": model_key,
            "latency_s": round(latency_s, 3),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "est_cost_usd": round(estimate_cost(model_key, usage), 6),
            "verdict": extra.get("verdict"),
            "error": (str(error)[:500] if error else None),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.calls.append(entry)

        # Write immediately, not just at save() — so a long/streaming run
        # is inspectable in the DB before it finishes, and so a call that
        # fails and is never retried again still leaves a record.
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO llm_calls
                    (run_id, node, attempt, model, latency_s, input_tokens,
                     output_tokens, total_tokens, est_cost_usd, verdict, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["run_id"], entry["node"], entry["attempt"],
                    entry["model"], entry["latency_s"], entry["input_tokens"],
                    entry["output_tokens"], entry["total_tokens"],
                    entry["est_cost_usd"], entry["verdict"], entry["error"],
                    entry["timestamp"],
                ),
            )
        return entry

    def summary(self, topic, attempts_used, approved) -> dict:
        total_latency = sum(c["latency_s"] for c in self.calls)
        total_tokens = sum(c["total_tokens"] for c in self.calls)
        total_cost = sum(c["est_cost_usd"] for c in self.calls)
        return {
            "run_id": self.run_id,
            "topic": topic,
            "attempts_used": attempts_used,
            "approved": bool(approved),
            "one_shot_approval": bool(approved and attempts_used == 1),
            "refusal_flagged": self.refusal_flagged,
            "total_latency_s": round(total_latency, 3),
            "total_tokens": total_tokens,
            "est_total_cost_usd": round(total_cost, 6),
            "num_llm_calls": len(self.calls),
        }

    def save(self, topic, attempts_used, approved) -> dict:
        summary = self.summary(topic, attempts_used, approved)
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs
                    (run_id, topic, attempts_used, approved, one_shot_approval,
                     refusal_flagged, total_latency_s, total_tokens,
                     est_total_cost_usd, num_llm_calls, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    attempts_used = excluded.attempts_used,
                    approved = excluded.approved,
                    one_shot_approval = excluded.one_shot_approval,
                    refusal_flagged = excluded.refusal_flagged,
                    total_latency_s = excluded.total_latency_s,
                    total_tokens = excluded.total_tokens,
                    est_total_cost_usd = excluded.est_total_cost_usd,
                    num_llm_calls = excluded.num_llm_calls
                """,
                (
                    summary["run_id"], summary["topic"], summary["attempts_used"],
                    int(summary["approved"]), int(summary["one_shot_approval"]),
                    int(summary["refusal_flagged"]), summary["total_latency_s"],
                    summary["total_tokens"], summary["est_total_cost_usd"],
                    summary["num_llm_calls"], datetime.now().isoformat(timespec="seconds"),
                ),
            )
        return summary

    def print_summary(self, topic, attempts_used, approved) -> dict:
        summary = self.summary(topic, attempts_used, approved)
        print("\n" + "=" * 55)
        print("RUN METRICS")
        print("=" * 55)
        for k, v in summary.items():
            print(f"{k}: {v}")
        return summary
