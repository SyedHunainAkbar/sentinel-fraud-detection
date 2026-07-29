---
inclusion: always
---

# Agentic AI & RAG Steering (Fraud Investigation Copilot)

The copilot turns a flagged transaction into an auditable, cited investigation note using
retrieval-augmented generation and a tool-using agent. Apply these principles.

## Grounding & citations (non-negotiable in a regulated domain)
- Every claim in a generated disposition MUST be grounded in either retrieved policy text
  or computed evidence (a score, a feature, a history fact). Cite the source chunk id.
- If retrieval returns nothing relevant, the agent says so and requests review rather than
  fabricating. No ungrounded assertions.

## RAG design
- Corpus: fraud typologies, investigation playbook, and regulatory notes in `data/policies/`.
- Retrieval is pluggable: a TF-IDF retriever works offline for tests/CI; an embedding
  retriever (Bedrock/OpenAI/Anthropic) is the production path. Same interface either way.
- Chunk by section; return top-k chunks with ids so outputs can cite them.

## Agentic design (human-in-the-loop)
The investigation agent is a tool-using workflow, not an autonomous decision-maker. It
recommends; a human dispositions. Tools:
1. `score_transaction` — model probability of fraud.
2. `get_customer_history` — recent transactions for the card.
3. `retrieve_policy` — top-k relevant policy/typology chunks (RAG).
4. `draft_disposition` — LLM synthesis of a cited recommendation.

The agent records a **decision trace** (each tool call + result) so the reasoning is
auditable. Map cleanly to a LangGraph StateGraph (nodes = tools, edges = the workflow);
document that mapping. Keep the LLM adapter swappable with a deterministic fallback so the
pipeline runs without API keys.

## Output contract
`draft_disposition` returns: `recommendation` (escalate | clear | request_info),
`rationale` (grounded prose), `citations` (chunk ids), and `confidence`. Persist the full
trace for audit.
