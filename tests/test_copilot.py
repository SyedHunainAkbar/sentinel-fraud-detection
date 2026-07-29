"""Tests for the RAG retriever and investigation agent."""
from sentinel.copilot.investigate import investigate
from sentinel.copilot.retriever import TfidfRetriever


def test_retriever_finds_relevant_policy():
    r = TfidfRetriever()
    hits = r.retrieve("many small online transactions testing a stolen card", k=2)
    assert hits, "retriever returned nothing"
    # the typologies doc should surface for a card-testing query
    assert any("typolog" in h.chunk.source for h in hits)


def test_investigation_is_grounded_and_traced():
    txn = {"trans_num": "t1", "category": "shopping_net", "amt": 950.0}
    inv = investigate(txn, score_fn=lambda t: 0.92)
    assert inv.recommendation in {"escalate", "clear", "request_info"}
    assert inv.citations, "disposition must cite retrieved policy"
    tools = [s.tool for s in inv.trace]
    assert tools == ["score_transaction", "get_customer_history",
                     "retrieve_policy", "draft_disposition"]
