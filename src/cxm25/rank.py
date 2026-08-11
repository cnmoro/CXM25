"""High-level, one-call ranking API: ``cxm25.rank(query, docs)``.

Builds whatever it needs on the fly (corpus statistics, scorers) so the common
case is a single function call. For production workloads, prefer building the
statistics once and reusing a scorer — see the lower-level API
(:class:`~cxm25.scoring.CXM25`, :func:`~cxm25.stats.build_corpus_stats`).
"""

import os

from .baked import (
    DATA_DIR as _BAKED_DATA_DIR,
    load_baked_coindex,
    load_baked_meta,
    load_baked_stats,
)
from .scoring import BM25, CXM25
from .stats import build_corpus_stats, tokenize_doc
from .textnorm import Normalizer

__all__ = ["rank"]


def rank(query, docs, *, model="cxm25", lang="pt", gram_n=3, top_k=None,
         stats=None, avg_len=None, coindex=None, use_baked_stats=None,
         n_jobs=1, stemmer=None, stopwords=None, scorer=None):
    """Rank ``docs`` against ``query`` and return a ranked result list.

    Args:
        query: query text.
        docs: iterable of document strings (or an iterator over strings).
        model: ``"cxm25"`` (default, pure algorithm), ``"bm25"`` (reference),
            or ``"ltr"`` (the baked CXM25-LTR model; requires xgboost).
        lang: language code for the normalizer (``"pt"`` built in).
        gram_n: character-gram size.
        top_k: return only the top-k results (None = all).
        stats: optional prebuilt :class:`~cxm25.stats.CorpusStats`. If omitted,
            statistics are built from ``docs`` for ``bm25``/``cxm25``, and the
            shipped reference statistics are used for ``ltr``.
        avg_len: optional average content-stem length (computed from ``docs``
            or taken from the baked metadata if omitted).
        coindex: optional thesaurus (only used with ``model="ltr"``; defaults
            to the baked thesaurus).
        use_baked_stats: force the shipped reference statistics even for
            ``bm25``/``cxm25``.
        n_jobs: parallel workers when building statistics from ``docs``.
        stemmer / stopwords: optional overrides for the normalizer.
        scorer: an already-built scorer to use directly (skips model/stats
            resolution).

    Returns:
        List of ``{"rank": int, "doc": str, "score": float}`` dicts, sorted by
        score descending (ties keep document order).

    Example:
        >>> cxm25.rank("praias da california em dezembro",
        ...            ["As melhores praias da California ...", "A California é ..."])
        [{'rank': 1, 'doc': 'As melhores praias ...', 'score': 4.38}, ...]
    """
    if scorer is not None:
        scores = _score_with(scorer, query, docs, lang, stemmer, stopwords)
        return _results(docs, scores, top_k)

    norm = Normalizer(lang=lang, stemmer=stemmer, stopwords=stopwords)
    qt = norm(query)
    if not qt:
        raise ValueError("query has no content tokens after normalization")

    docs = list(docs)
    if not docs:
        return []

    # tokenize each unique document once, preserving the caller's order
    cache = {}
    for d in docs:
        if d not in cache:
            cache[d] = tokenize_doc(norm, d)
    docobjs = [cache[d] for d in docs]

    use_baked = use_baked_stats if use_baked_stats is not None else (model == "ltr")

    if stats is None:
        if use_baked:
            stats = load_baked_stats()
            if avg_len is None:
                avg_len = load_baked_meta()["avg_len"]
        else:
            stats = build_corpus_stats(docs, gram_n=gram_n, n_jobs=n_jobs)
            if avg_len is None:
                avg_len = _avg_len(docobjs)
    elif avg_len is None:
        avg_len = (
            load_baked_meta()["avg_len"] if use_baked else _avg_len(docobjs)
        )

    if model == "bm25":
        sc = BM25(stats.df, stats.N, avg_len)
        scores = [sc.score(qt, o) for o in docobjs]
    elif model == "ltr":
        from .ltr import LTRScorer
        if coindex is None:
            coindex = load_baked_coindex()
        sc = LTRScorer(os.path.join(_BAKED_DATA_DIR, "model.ubj"),
                       stats, avg_len, coindex=coindex)
        scores = sc.score_pool(qt, docobjs)
    else:  # cxm25
        sc = CXM25(stats.df, stats.df2, stats.N, avg_len, gdf=stats.gdf,
                   gram_n=gram_n)
        ctx = sc.prepare(qt)
        scores = [sc.score_prepared(ctx, o) for o in docobjs]

    return _results(docs, scores, top_k)


def _score_with(scorer, query, docs, lang, stemmer, stopwords):
    norm = Normalizer(lang=lang, stemmer=stemmer, stopwords=stopwords)
    qt = norm(query)
    cache = {}
    for d in docs:
        if d not in cache:
            cache[d] = tokenize_doc(norm, d)
    docobjs = [cache[d] for d in docs]
    if hasattr(scorer, "score_pool"):
        return scorer.score_pool(qt, docobjs)
    if hasattr(scorer, "prepare"):
        ctx = scorer.prepare(qt)
        return [scorer.score_prepared(ctx, o) for o in docobjs]
    return [scorer.score(qt, o) for o in docobjs]


def _avg_len(docobjs):
    return sum(len(o[0]) for o in docobjs) / max(len(docobjs), 1)


def _results(docs, scores, top_k):
    order = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), i))
    if top_k is not None:
        order = order[:top_k]
    return [{"rank": r + 1, "doc": docs[i], "score": float(scores[i])}
            for r, i in enumerate(order)]
