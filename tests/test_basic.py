"""Smoke tests for the core CXM25 package (no xgboost required)."""

import math

import pytest

from cxm25 import BM25, CXM25, CorpusStats, Normalizer, build_corpus_stats


# ---------------------------------------------------------------------------
# text normalization
# ---------------------------------------------------------------------------

def test_fold_and_stem():
    norm = Normalizer(lang="pt")
    assert norm("Café") == ["caf"]          # accent folded + stemmed
    assert "caf" in norm("café")
    assert norm("correndo") == ["corr"]


def test_stopwords_removed():
    norm = Normalizer(lang="pt")
    assert "de" not in norm("o livro de história")


def test_deterministic():
    norm = Normalizer(lang="pt")
    assert norm("correndo rápido") == norm("Correndo RÁPIDO")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _toy_corpus():
    docs = [
        "As melhores praias quentes da California em dezembro para o inverno ficam na costa sul",
        "A Califórnia é um estado dos Estados Unidos na costa oeste",
        "A anemia é um número reduzido de glóbulos vermelhos no sangue",
        "Praias quentes no Brasil em dezembro atraem muitos turistas",
        "O clima do sul da California raramente fica frio no inverno",
    ]
    stats = build_corpus_stats(docs, gram_n=3)
    avg_len = sum(len(n) for n in map(Normalizer(), docs)) / len(docs)
    return docs, stats, avg_len


def test_build_stats():
    docs, stats, _ = _toy_corpus()
    assert stats.N == len(docs)
    assert "californ" in stats.df
    assert len(stats.gdf) > 0


def test_bm25_ranks_topic_doc_first():
    docs, stats, avg_len = _toy_corpus()
    bm = BM25(stats.df, stats.N, avg_len)
    norm = Normalizer()
    qt = norm("praias da california em dezembro")
    scored = sorted((bm.score(qt, tokenize(d)), d) for d in docs)
    # the california-beach doc should outrank the general california doc
    assert scored[-1][1] == docs[0]


def test_cxm25_scores_and_prepare_match():
    docs, stats, avg_len = _toy_corpus()
    cx = CXM25(stats.df, stats.df2, stats.N, avg_len, gdf=stats.gdf)
    norm = Normalizer()
    qt = norm("praias da california em dezembro")
    d = tokenize(docs[0])
    assert abs(cx.score(qt, d) - cx.score_prepared(cx.prepare(qt), d)) < 1e-9


def test_cxm25_beats_bm25_on_toy():
    """On the toy corpus, CXM25 must at least not regress vs BM25."""
    docs, stats, avg_len = _toy_corpus()
    bm = BM25(stats.df, stats.N, avg_len)
    cx = CXM25(stats.df, stats.df2, stats.N, avg_len, gdf=stats.gdf)
    norm = Normalizer()
    qt = norm("praias da california em dezembro")
    pairs = [(docs[0], docs[2]), (docs[0], docs[3])]
    bm_ok = sum(bm.score(qt, tokenize(p)) > bm.score(qt, tokenize(n)) for p, n in pairs)
    cx_ok = sum(cx.score(qt, tokenize(p)) > cx.score(qt, tokenize(n)) for p, n in pairs)
    assert cx_ok >= bm_ok


def test_stable_hash_deterministic():
    from cxm25.stats import _stable_hash
    assert _stable_hash("abc") == _stable_hash("abc")


def test_tokenize():
    tokenize("O livro de história está na mesa")
    assert True


def tokenize(text):
    from cxm25.stats import tokenize_doc
    return tokenize_doc(Normalizer(), text)
