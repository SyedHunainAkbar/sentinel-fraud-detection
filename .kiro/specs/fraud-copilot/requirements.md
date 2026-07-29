# Requirements — Fraud Investigation Copilot (RAG + Agentic)

## Introduction
A retrieval-augmented, tool-using agent that turns a flagged transaction into an
auditable, cited investigation note. Human-in-the-loop: it recommends, an analyst decides.

## R1 — Policy corpus & retrieval (RAG)
- The system SHALL chunk the policy corpus in `data/policies/` into section chunks with
  stable ids.
- WHEN given a query, the retriever SHALL return the top-k relevant chunks with ids.
- The retriever SHALL be pluggable: TF-IDF offline; embeddings (Bedrock/Anthropic) in
  production, behind one interface.

## R2 — Investigation agent (tools + trace)
- The agent SHALL orchestrate four tools in order: score_transaction, get_customer_history,
  retrieve_policy, draft_disposition.
- The agent SHALL record a decision trace (tool + output) for every step, for audit.
- The workflow SHALL map to a LangGraph StateGraph, documented in design.md.

## R3 — Grounded, cited disposition
- draft_disposition SHALL return recommendation (escalate|clear|request_info), a grounded
  rationale, citation chunk ids, and a confidence score.
- Every claim SHALL be grounded in retrieved policy or computed evidence; IF retrieval is
  empty, THEN the agent SHALL request review rather than fabricate.
- The LLM SHALL be swappable with a deterministic fallback so CI runs without keys.

## R4 — Output & audit
- The demo SHALL persist investigations (with traces) to `reports/investigations.json`.
