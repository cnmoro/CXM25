"""Corpus statistics needed by the scorers, plus the optional distributional
thesaurus used by the learned (LTR) variant.

``build_corpus_stats`` computes:
  - N    number of documents in the sample
  - df   term -> #docs containing the term          (for IDF)
  - gdf  char-gram -> #docs containing the gram     (for the G component)
  - df2  (term, term) -> #docs containing both      (for the optional B
         component / LTR co-occurrence features)

``df2`` is only tracked for term pairs that appear in ``query_terms`` (when
provided) — that is all scoring ever needs, and it keeps the pass cheap.

``build_coindex`` derives, for every query term, a ranked list of corpus terms
that co-occur with it (local-MI weighted) — a collection-level statistical
thesaurus used to expand queries in the LTR variant. Both builders are generic
collection statistics; no dataset-specific rules.
"""

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool

from .textnorm import Normalizer

__all__ = ["CorpusStats", "build_corpus_stats", "build_coindex", "tokenize_doc"]

_DOC = "doc"


def tokenize_doc(norm, text):
    """Tokenize one document into a (toks, tf, firstpos) scoring object."""
    toks = norm(text)
    tf = Counter(toks)
    fp = {}
    for i, t in enumerate(toks):
        if t not in fp:
            fp[t] = i
    return (toks, dict(tf), fp)


def _grams(utoks, gram_n):
    out = set()
    for t in utoks:
        if len(t) <= gram_n:
            out.add(t)
            continue
        for i in range(len(t) - gram_n + 1):
            out.add(t[i:i + gram_n])
    return out


@dataclass
class CorpusStats:
    """Container for corpus statistics."""

    N: int
    df: dict = field(default_factory=dict)
    df2: dict = field(default_factory=dict)
    gdf: dict = field(default_factory=dict)


def _stable_hash(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


# ---------------------------------------------------------------------------
# build_corpus_stats
# ---------------------------------------------------------------------------

_STATS_GLOBALS = None


def _stats_worker(task):
    norm, gram_n, sample_num, sample_den, path = task
    query_terms, pairs_set = _STATS_GLOBALS
    df = Counter()
    df2 = Counter()
    gdf = Counter()
    N = 0
    seen = set()
    for text in _iter_docs(path):
        if _stable_hash(text) % sample_den >= sample_num:
            continue
        if text in seen:
            continue
        seen.add(text)
        utoks = set(norm(text))
        df.update(utoks)
        gdf.update(_grams(utoks, gram_n))
        if query_terms is not None:
            inter = utoks & query_terms
            if len(inter) > 1:
                it = sorted(inter)
                for i in range(len(it) - 1):
                    for j in range(i + 1, len(it)):
                        p = (it[i], it[j])
                        if p in pairs_set:
                            df2[p] += 1
        N += 1
    return N, df, df2, gdf


def _iter_docs(x):
    """Accept a list of doc strings, or a path to a file with one doc per
    line, or a path to a parquet file with a ``text`` column."""
    if isinstance(x, str):
        if x.endswith(".parquet"):
            import pandas as pd
            for chunk in pd.read_parquet(x, columns=["text"], chunksize=200000):
                for v in chunk["text"]:
                    if isinstance(v, str):
                        yield v
        else:
            with open(x, encoding="utf-8") as fh:
                for line in fh:
                    yield line.rstrip("\n")
    else:
        for d in x:
            yield d


def build_corpus_stats(docs, query_terms=None, gram_n=3, sample_rate=1.0,
                       max_docs=None, n_jobs=1, lang="pt", progress=None):
    """Compute corpus statistics from a document collection.

    Args:
        docs: iterable of document strings, a text file path (one doc/line),
            or a parquet path with a ``text`` column.
        query_terms: optional set of content terms; ``df2`` is only computed
            for pairs inside this set.
        gram_n: character-gram size for ``gdf``.
        sample_rate: fraction of (unique, by text) documents kept.
        max_docs: hard cap on the number of documents used.
        n_jobs: parallel workers (only for path-based inputs; list inputs are
            always processed in-process).
        lang: language code passed to :class:`Normalizer`.
        progress: optional callable ``(ndocs) -> None``.
    """
    norm = Normalizer(lang=lang)
    if isinstance(docs, str):
        paths = [docs] if not docs.endswith(".parquet") else _parquet_shards(docs)
    else:
        paths = list(docs)

    sample_num = int(round(sample_rate * 100))
    sample_den = 100

    df = Counter()
    df2 = Counter()
    gdf = Counter()
    N = 0

    if n_jobs > 1 and isinstance(docs, str):
        query_terms = set(query_terms) if query_terms is not None else None
        pairs = set()
        if query_terms is not None:
            for a in query_terms:
                for b in query_terms:
                    if a < b:
                        pairs.add((a, b))
        tasks = [(norm, gram_n, sample_num, sample_den, p) for p in paths]
        with Pool(n_jobs, initializer=_stats_init, initargs=((query_terms, pairs),)) as pool:
            for n, df_, df2_, gdf_ in pool.imap_unordered(_stats_worker, tasks):
                df.update(df_)
                df2.update(df2_)
                gdf.update(gdf_)
                N += n
                if progress:
                    progress(N)
                if max_docs and N >= max_docs:
                    break
    else:
        query_terms = set(query_terms) if query_terms is not None else None
        pairs = set()
        if query_terms is not None:
            for a in query_terms:
                for b in query_terms:
                    if a < b:
                        pairs.add((a, b))
        seen = set()
        for text in _iter_docs(docs):
            if _stable_hash(text) % sample_den >= sample_num:
                continue
            if text in seen:
                continue
            seen.add(text)
            utoks = set(norm(text))
            df.update(utoks)
            gdf.update(_grams(utoks, gram_n))
            if query_terms is not None:
                inter = utoks & query_terms
                if len(inter) > 1:
                    it = sorted(inter)
                    for i in range(len(it) - 1):
                        for j in range(i + 1, len(it)):
                            p = (it[i], it[j])
                            if p in pairs:
                                df2[p] += 1
            N += 1
            if progress and N % 10000 == 0:
                progress(N)
            if max_docs and N >= max_docs:
                break

    return CorpusStats(N=N, df=dict(df), df2=dict(df2), gdf=dict(gdf))


def _stats_init(payload):
    global _STATS_GLOBALS
    _STATS_GLOBALS = payload


def _parquet_shards(path):
    return [path]


# ---------------------------------------------------------------------------
# build_coindex
# ---------------------------------------------------------------------------

_CO_GLOBALS = None


def _co_worker(path):
    norm, qt, sample_num, sample_den, path = path
    co = defaultdict(Counter)
    seen = set()
    N = 0
    for text in _iter_docs(path):
        if _stable_hash(text) % sample_den >= sample_num:
            continue
        if text in seen:
            continue
        seen.add(text)
        utoks = set(norm(text))
        inter = utoks & qt
        if inter:
            for t in inter:
                co[t].update(utoks)
        N += 1
    return co, N


def _co_init(qt):
    global _CO_GLOBALS
    _CO_GLOBALS = qt


def build_coindex(docs, query_terms, sample_rate=1.0, max_docs=None,
                  min_count=5, k=30, n_jobs=1, lang="pt", stats=None,
                  progress=None):
    """Build a distributional co-occurrence thesaurus.

    For every term in ``query_terms`` that appears in the collection, returns a
    list of the ``k`` terms that co-occur with it most, ranked by local-MI
    (count-weighted PMI). Generic statistical thesaurus construction.

    Args:
        docs: same accepted forms as :func:`build_corpus_stats`.
        query_terms: set of stemmed query terms to build entries for.
        sample_rate / max_docs: sampling controls.
        min_count: minimum co-occurrence count to keep a candidate.
        k: number of thesaurus terms kept per query term.
        n_jobs: parallel workers.
        stats: optional :class:`CorpusStats` to compute PMI document
            frequencies (falls back to ``docs`` if None).
    """
    norm = Normalizer(lang=lang)
    sample_num = int(round(sample_rate * 100))
    sample_den = 100
    qt = set(query_terms)

    co = defaultdict(Counter)
    N_co = 0
    if n_jobs > 1 and isinstance(docs, str):
        paths = [docs] if not docs.endswith(".parquet") else _parquet_shards(docs)
        tasks = [(norm, qt, sample_num, sample_den, p) for p in paths]
        with Pool(n_jobs, initializer=_co_init, initargs=(qt,)) as pool:
            for part, n_docs in pool.imap_unordered(_co_worker, tasks):
                for t, cnt in part.items():
                    co[t].update(cnt)
                N_co += n_docs
                if progress:
                    progress(len(co))
    else:
        seen = set()
        for text in _iter_docs(docs):
            if _stable_hash(text) % sample_den >= sample_num:
                continue
            if text in seen:
                continue
            seen.add(text)
            utoks = set(norm(text))
            inter = utoks & qt
            if inter:
                for t in inter:
                    co[t].update(utoks)
            N_co += 1
            if progress and N_co % 10000 == 0:
                progress(N_co)
            if max_docs and N_co >= max_docs:
                break

    if N_co == 0:
        N_co = max_docs or 1

    # document frequencies for PMI
    if stats is not None:
        df = stats.df
        N_corpus = stats.N
    else:
        tmp = build_corpus_stats(docs, query_terms=None, sample_rate=sample_rate,
                                 max_docs=max_docs, n_jobs=n_jobs, lang=lang)
        df = tmp.df
        N_corpus = tmp.N

    out = {}
    for t, cnt in co.items():
        dft = df.get(t, 1)
        scored = []
        for u, c in cnt.items():
            if u == t or c < min_count or len(u) < 3 or any(ch.isdigit() for ch in u):
                continue
            dfu = df.get(u, 1)
            pmi = math.log((c / N_co) / ((dft / N_corpus) * (dfu / N_corpus)) + 1e-12)
            if pmi <= 0:
                continue
            lmi = c * min(pmi, 8.0)
            scored.append((u, min(pmi, 8.0), lmi))
        scored.sort(key=lambda x: -x[2])
        out[t] = [(u, p) for u, p, _ in scored[:k]]
    return out
