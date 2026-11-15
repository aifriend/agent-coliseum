import os

import condor_rag_agent as c


def test_facts_path_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert os.path.isabs(c.FACTS_PATH) and os.path.exists(c.FACTS_PATH)
    _, matrix, facts = c.build_rag_index(max_facts=5)
    assert len(facts) == 5 and matrix.shape[0] == 5


def test_index_is_populated():
    assert 0 < len(c.rag_facts) <= 50


def test_finds_relevant_fact():
    hits = c.search_rag("Octavio Paz Premio Nobel de Literatura", top_k=3)
    assert hits and "Octavio Paz" in hits[0]


def test_top_k_limits_results():
    assert len(c.search_rag("México volcán", top_k=1)) <= 1


def test_no_match_returns_empty():
    assert c.search_rag("zzzqqq xxyyzz") == []
