"""Agentic fraud investigation: a tool-using workflow with an auditable trace.

The agent orchestrates four tools — score, customer history, policy retrieval (RAG), and
LLM disposition — and records each step. This maps directly onto a LangGraph StateGraph
(each tool is a node; edges follow the order below); the plain-Python loop here keeps it
runnable and testable without extra dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .llm import draft_disposition
from .retriever import get_retriever


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


def _typology_query(txn: dict) -> str:
    """Turn a transaction into a retrieval query describing its risk signals."""
    parts = [str(txn.get("category", ""))]
    if txn.get("amt", 0) > 300:
        parts.append("high value single purchase")
    parts.append("velocity" if txn.get("_velocity", 0) > 3 else "")
    parts.append("geographic distance new merchant")
    return " ".join(p for p in parts if p)


def investigate(txn: dict, score_fn, history_df: pd.DataFrame | None = None,
                retriever=None) -> Investigation:
    """Run the investigation workflow for one transaction.

    Parameters
    ----------
    txn : dict
        The transaction under review.
    score_fn : callable
        ``score_fn(txn) -> float`` fraud probability (inject the trained model or a stub).
    history_df : DataFrame, optional
        Recent transactions for the card, for the history tool.
    retriever : optional
        Any object with ``retrieve(query, k)``; defaults to the TF-IDF retriever.
    """
    retriever = retriever or get_retriever()
    trace: list[TraceStep] = []

    # Tool 1: score
    prob = float(score_fn(txn))
    trace.append(TraceStep("score_transaction", f"probability={prob:.3f}"))

    # Tool 2: customer history
    if history_df is not None and len(history_df):
        hist_summary = (f"{len(history_df)} recent txns; "
                        f"median amt ${history_df['amt'].median():.0f}")
    else:
        hist_summary = "no prior history available"
    trace.append(TraceStep("get_customer_history", hist_summary))

    # Tool 3: retrieve policy (RAG)
    retrieved = retriever.retrieve(_typology_query(txn), k=3)
    trace.append(TraceStep("retrieve_policy",
                           ", ".join(r.chunk.id for r in retrieved) or "none"))

    # Tool 4: draft disposition (LLM, grounded + cited)
    disp = draft_disposition(prob, hist_summary, retrieved)
    trace.append(TraceStep("draft_disposition",
                           f"{disp['recommendation']} ({disp['source']})"))

    return Investigation(
        transaction_id=str(txn.get("trans_num", "unknown")),
        probability=prob,
        recommendation=disp["recommendation"],
        rationale=disp["rationale"],
        citations=disp["citations"],
        confidence=disp["confidence"],
        trace=trace,
    )
