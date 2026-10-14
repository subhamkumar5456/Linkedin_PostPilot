"""
tracking/llm_tracker.py

A transparent wrapper around any LangChain chat model that automatically
tracks every .invoke() call — latency, token usage, estimated cost, and
refusal detection — without a single line of boilerplate in your node functions.

Usage in llms.py:

    from tracking.llm_tracker import TrackedLLM

    writer_llm = TrackedLLM(
        ChatGoogleGenerativeAI(model=..., google_api_key=...),
        model_key="gemini-writer",
        node="writer",
    )
    writer_llm_with_tools = writer_llm.bind_tools(tools)

    reviewer_llm = TrackedLLM(
        ChatGroq(model=..., temperature=0.2),
        model_key="groq-reviewer",
        node="reviewer",
    )

Usage in node functions — just call .invoke() normally, optionally
overriding node name or attempt number for that specific call:

    response = writer_llm_with_tools.invoke(messages)
    response = writer_llm_with_tools.invoke(messages, _node="writer_after_tools")
    response = reviewer_llm.invoke(messages, _attempt=state["attempt"])

The wrapper never passes `_node`, `_attempt`, or `_extra` through to the
underlying model, so existing LangChain call-site signatures are unaffected.
"""

import time
import logging

from .metrics import get_current_run, get_usage, looks_like_refusal

logger = logging.getLogger(__name__)


class TrackedLLM:
    """
    Wraps any LangChain BaseChatModel (or bound-model variant) and
    intercepts .invoke() to auto-record metrics via the active RunMetrics.

    Attributes:
        _llm        : The underlying LangChain model / bound-model.
        model_key   : Key used for cost lookup (matches PRICING_PER_MILLION_TOKENS).
        node        : Default node label recorded in llm_calls (e.g. "writer").
    """

    def __init__(self, llm, model_key: str, node: str):
        self._llm = llm
        self.model_key = model_key
        self.node = node

    # ------------------------------------------------------------------
    # Core interception
    # ------------------------------------------------------------------

    def invoke(self, input, *, _node: str = None, _attempt: int = None,
               _extra: dict = None, **kwargs):
        """
        Drop-in replacement for llm.invoke().

        Extra keyword args (prefixed with _) are consumed here and never
        forwarded to the underlying model:
            _node    : Override the node label for this specific call.
            _attempt : Override the attempt number for this call.
            _extra   : Extra dict forwarded to RunMetrics.log_call() (e.g. {"verdict": "APPROVED"}).

        All other **kwargs are passed through to the underlying .invoke().
        """
        node = _node or self.node
        attempt = _attempt if _attempt is not None else 1
        extra = _extra or {}

        # --- call the real model and time it ---
        t0 = time.perf_counter()
        try:
            response = self._llm.invoke(input, **kwargs)
        except Exception as exc:
            latency = time.perf_counter() - t0
            # Still log the failed call so errors are visible in the DB
            self._safe_log(node, attempt, latency,
                           {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                           extra={**extra, "error": exc})
            raise
        latency = time.perf_counter() - t0

        # --- extract usage & check for refusals ---
        usage = get_usage(response)

        if looks_like_refusal(response.content):
            try:
                get_current_run().refusal_flagged = True
            except RuntimeError:
                pass  # no active run — non-fatal

        # --- record to the active RunMetrics ---
        self._safe_log(node, attempt, latency, usage, extra=extra)

        return response

    # ------------------------------------------------------------------
    # Tool binding — returns another TrackedLLM so the bound model is
    # also intercepted (writer_llm_with_tools stays tracked).
    # ------------------------------------------------------------------

    def bind_tools(self, tools, **kwargs):
        bound = self._llm.bind_tools(tools, **kwargs)
        return TrackedLLM(bound, model_key=self.model_key, node=self.node)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_log(self, node, attempt, latency_s, usage, extra=None):
        """Log to RunMetrics if one is active; warn and skip otherwise."""
        try:
            run = get_current_run()
        except RuntimeError:
            logger.warning(
                "[TrackedLLM] No active RunMetrics — skipping metric log "
                "(node=%s, model=%s). Call set_current_run() before invoking.",
                node, self.model_key,
            )
            return
        run.log_call(node, attempt, self.model_key, latency_s, usage, extra=extra)

    # ------------------------------------------------------------------
    # Transparent attribute proxy — lets callers access any attribute
    # on the underlying model (e.g. .model_name, .temperature) as if
    # they were calling the real object.
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        # Only called when the attribute isn't found on TrackedLLM itself.
        return getattr(self._llm, name)

    def __repr__(self):
        return (
            f"TrackedLLM(model_key={self.model_key!r}, "
            f"node={self.node!r}, llm={self._llm!r})"
        )
