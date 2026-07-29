"""Pluggable LLM adapter for disposition synthesis.

Uses Anthropic if ``ANTHROPIC_API_KEY`` is set; otherwise a deterministic, grounded
fallback so the copilot runs in tests/CI without keys. Both paths return the same
structured, cited disposition contract.
"""
from __future__ import annotations

import os

DISPOSITION_SCHEMA = ("Return recommendation (escalate|clear|request_info), a grounded "
                      "rationale, citations (chunk ids), and confidence in [0,1].")


def _fallback(prob, history_summary, retrieved) -> dict:
    """Rule-grounded disposition when no LLM is configured."""
    citations = [r.chunk.id for r in retrieved]
    top = retrieved[0].chunk.heading if retrieved else "no matching policy"
    if prob >= 0.8 and retrieved:
        rec, conf = "escalate", 0.8
        why = (f"High model probability ({prob:.2f}) and the pattern matches '{top}'. "
               f"History: {history_summary}. Escalation warranted per playbook.")
    elif prob >= 0.4:
        rec, conf = "request_info", 0.55
        why = (f"Moderate probability ({prob:.2f}); signals are mixed given history "
               f"({history_summary}). Verify with customer before adverse action.")
    else:
        rec, conf = "clear", 0.7
        why = (f"Low probability ({prob:.2f}) and consistent with history "
               f"({history_summary}). No typology strongly matches.")
    return {"recommendation": rec, "rationale": why, "citations": citations,
            "confidence": conf, "source": "fallback"}


def draft_disposition(prob, history_summary, retrieved) -> dict:
    """Synthesize a cited disposition. Prefers Anthropic; falls back deterministically."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _fallback(prob, history_summary, retrieved)
    try:
        import anthropic

        context = "\n\n".join(f"[{r.chunk.id}] {r.chunk.heading}: {r.chunk.text}"
                              for r in retrieved)
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=("You are a fraud investigation copilot. Ground every claim in the "
                    "provided policy context and cite chunk ids. " + DISPOSITION_SCHEMA),
            messages=[{"role": "user", "content":
                       f"Model fraud probability: {prob:.2f}\n"
                       f"Customer history: {history_summary}\n\n"
                       f"Policy context:\n{context}\n\n"
                       "Produce the disposition as JSON."}],
        )
        import json

        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        out = json.loads(text)
        out["source"] = "anthropic"
        return out
    except Exception:  # noqa: BLE001 — never let synthesis break the pipeline
        return _fallback(prob, history_summary, retrieved)
