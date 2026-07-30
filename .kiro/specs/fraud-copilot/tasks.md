# Tasks — Fraud Investigation Copilot

- [x] 1. Corpus loader + chunker (`copilot/corpus.py`)
- [x] 2. TF-IDF retriever with pluggable seam (`copilot/retriever.py`)
- [x] 3. Pluggable LLM adapter + deterministic fallback (`copilot/llm.py`)
- [x] 4. Agent orchestration with decision trace (`copilot/investigate.py`)
- [x] 5. Demo writing `reports/investigations.json`
- [x] 6. Tests: retrieval relevance + grounded/traced disposition
- [x] 7. Swap in an embedding retriever (Bedrock/Anthropic) behind `get_retriever()`
- [x] 8. Port orchestration to a LangGraph StateGraph (keep the output contract)
- [x] 9. Add a Streamlit "analyst" tab: enter a txn, see disposition + citations + trace
- [x] 10. Evaluate copilot: agreement with labels on a held-out set of flagged txns
