"""Agentic fraud investigation: a LangGraph StateGraph workflow with auditable trace.

The agent orchestrates four tools — score, customer history, policy retrieval (RAG), and
LLM disposition — and records each step. Each tool is a node in the StateGraph; edges
follow the fixed investigation order. A deterministic fallback (no LangGraph import) is
kept so ``make copilot`` runs without extra dependencies.

Public API is unchanged: ``investigate(txn, score_fn, history_df, retriever)`` returns an
``Investigation`` dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .llm import draft_disposition
from .retriever import get_retriever

# ---------------------------------------------------------------------------
# Output contract (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    tool: str
    output: str


@dataclass
class Investigation:
    transaction_id: str
    probability: float
    recommendation: str
    rationale: str
    citations: list[str]
    confidence: float
    trace: list[TraceStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "probability": self.probability,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "citations": self.citations,
            "confidence": self.confidence,
            "trace": [step.__dict__ for step in self.trace],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _typology_query(txn: dict) -> str:
    """Turn a transaction into a retrieval query describing its risk signals."""
    parts = [str(txn.get("category", ""))]
    if txn.get("amt", 0) > 300:
        parts.append("high value single purchase")
    parts.append("velocity" if txn.get("_velocity", 0) > 3 else "")
    parts.append("geographic distance new merchant")
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# LangGraph StateGraph implementation
# ---------------------------------------------------------------------------

def _build_graph(
    txn: dict,
    score_fn: Any,
    history_df: pd.DataFrame | None,
    retriever: Any,
):
    """Build and compile the LangGraph investigation graph.

    Returns the compiled graph ready to invoke, or None if langgraph is
    unavailable (triggering the deterministic fallback).
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None

    from typing import TypedDict

    class InvestigationState(TypedDict, total=False):
        txn: dict
        probability: float
        history_summary: str
        retrieved: list
        disposition: dict
        trace: list

    # --- Node functions ---

    def score_node(state: InvestigationState) -> dict:
        prob = float(score_fn(state["txn"]))
        trace = list(state.get("trace", []))
        trace.append({"tool": "score_transaction", "output": f"probability={prob:.3f}"})
        return {"probability": prob, "trace": trace}

    def history_node(state: InvestigationState) -> dict:
        if history_df is not None and len(history_df):
            summary = (
                f"{len(history_df)} recent txns; "
                f"median amt ${history_df['amt'].median():.0f}"
            )
        else:
            summary = "no prior history available"
        trace = list(state.get("trace", []))
        trace.append({"tool": "get_customer_history", "output": summary})
        return {"history_summary": summary, "trace": trace}

    def retrieve_node(state: InvestigationState) -> dict:
        query = _typology_query(state["txn"])
        results = retriever.retrieve(query, k=3)
        cite_ids = ", ".join(r.chunk.id for r in results) or "none"
        trace = list(state.get("trace", []))
        trace.append({"tool": "retrieve_policy", "output": cite_ids})
        return {"retrieved": results, "trace": trace}

    def disposition_node(state: InvestigationState) -> dict:
        disp = draft_disposition(
            state["probability"],
            state["history_summary"],
            state["retrieved"],
        )
        trace = list(state.get("trace", []))
        trace.append({
            "tool": "draft_disposition",
            "output": f"{disp['recommendation']} ({disp['source']})",
        })
        return {"disposition": disp, "trace": trace}

    # --- Build graph ---
    builder = StateGraph(InvestigationState)
    builder.add_node("score_transaction", score_node)
    builder.add_node("get_customer_history", history_node)
    builder.add_node("retrieve_policy", retrieve_node)
    builder.add_node("draft_disposition", disposition_node)

    builder.set_entry_point("score_transaction")
    builder.add_edge("score_transaction", "get_customer_history")
    builder.add_edge("get_customer_history", "retrieve_policy")
    builder.add_edge("retrieve_policy", "draft_disposition")
    builder.add_edge("draft_disposition", END)

    return builder.compile()


def _fallback_investigate(
    txn: dict,
    score_fn: Any,
    history_df: pd.DataFrame | None,
    retriever: Any,
) -> Investigation:
    """Plain-Python loop fallback when LangGraph is not installed."""
    trace: list[TraceStep] = []

    # Tool 1: score
    prob = float(score_fn(txn))
    trace.append(TraceStep("score_transaction", f"probability={prob:.3f}"))

    # Tool 2: customer history
    if history_df is not None and len(history_df):
        hist_summary = (
            f"{len(history_df)} recent txns; "
            f"median amt ${history_df['amt'].median():.0f}"
        )
    else:
        hist_summary = "no prior history available"
    trace.append(TraceStep("get_customer_history", hist_summary))

    # Tool 3: retrieve policy (RAG)
    retrieved = retriever.retrieve(_typology_query(txn), k=3)
    trace.append(TraceStep(
        "retrieve_policy",
        ", ".join(r.chunk.id for r in retrieved) or "none",
    ))

    # Tool 4: draft disposition (LLM, grounded + cited)
    disp = draft_disposition(prob, hist_summary, retrieved)
    trace.append(TraceStep(
        "draft_disposition",
        f"{disp['recommendation']} ({disp['source']})",
    ))

    return Investigation(
        transaction_id=str(txn.get("trans_num", "unknown")),
        probability=prob,
        recommendation=disp["recommendation"],
        rationale=disp["rationale"],
        citations=disp["citations"],
        confidence=disp["confidence"],
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Public API (unchanged signature)
# ---------------------------------------------------------------------------

def investigate(
    txn: dict,
    score_fn: Any,
    history_df: pd.DataFrame | None = None,
    retriever: Any = None,
) -> Investigation:
    """Run the investigation workflow for one transaction.

    Parameters
    ----------
    txn : dict
        The transaction under review.
    score_fn : callable
        ``score_fn(txn) -> float`` fraud probability (inject the trained model or stub).
    history_df : DataFrame, optional
        Recent transactions for the card, for the history tool.
    retriever : optional
        Any object with ``retrieve(query, k)``; defaults to the TF-IDF retriever.

    Returns
    -------
    Investigation
        Dataclass with recommendation, rationale, citations, confidence, and trace.
    """
    retriever = retriever or get_retriever()

    # Attempt LangGraph execution; fall back to plain loop if unavailable
    graph = _build_graph(txn, score_fn, history_df, retriever)

    if graph is None:
        return _fallback_investigate(txn, score_fn, history_df, retriever)

    # Run the compiled graph
    initial_state = {"txn": txn, "trace": []}
    result = graph.invoke(initial_state)

    # Convert raw trace dicts back to TraceStep dataclasses
    trace = [TraceStep(**step) for step in result["trace"]]
    disp = result["disposition"]

    return Investigation(
        transaction_id=str(txn.get("trans_num", "unknown")),
        probability=result["probability"],
        recommendation=disp["recommendation"],
        rationale=disp["rationale"],
        citations=disp["citations"],
        confidence=disp["confidence"],
        trace=trace,
    )
