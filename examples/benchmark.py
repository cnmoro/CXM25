"""Throughput benchmark: CXM25 / CXM25-LTR vs BM25.

Measures (on tokenized documents, so preprocessing is excluded):

  - indexing throughput  : documents tokenized per second (shared by all models)
  - scoring throughput   : documents scored per second for one query
  - query latency        : wall-clock ms to score one query against K docs

Run:
    python examples/benchmark.py                 # synthetic Portuguese docs
    python examples/benchmark.py --docs data.txt # one document per line
    python examples/benchmark.py --docs docs.parquet --text-col text

Set ``--k 2000 --queries 30`` to reproduce the numbers in the README.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cxm25
from cxm25 import BM25, CXM25, Normalizer
from cxm25.stats import tokenize_doc

_SNIPPETS = [
    "as melhores praias da california para o clima quente do inverno ficam na costa sul",
    "a california e um estado dos estados unidos localizado na costa oeste do pais",
    "a anemia e um termo medico que se refere a um numero reduzido de globulos vermelhos",
    "praias quentes no brasil em dezembro atraem milhares de turistas europeus",
    "o clima do sul da california raramente fica frio mesmo no inverno ensolarado",
    "o custo para instalar pisos de tijolos depende do tamanho do projeto e dos materiais",
    "a hemofilia e uma doenca hereditaria ligada ao cromossomo x que afeta a coagulacao",
    "as criancas devem comparecer ao primeiro exame dentario antes de completar dois anos",
    "o bebe pode aprender a limpar os dentes e identificar suas necessidades de fluor",
    "os atrasos no desenvolvimento fisico ocorrem quando as criancas nao realizam atividades",
]
_QUERIES = [
    "praias da california em dezembro",
    "o que causa a anemia",
    "quanto custa instalar tijolos",
    "o que e hemofilia",
    "primeira consulta dentaria do bebe",
    "clima quente no inverno",
]


def _make_docs(n, seed=0):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(_SNIPPETS, 4)) for _ in range(n)]


def _load_docs(path, text_col):
    if path is None:
        return _make_docs(50000)
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path, columns=[text_col])[text_col].dropna().astype(str).tolist()
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh]


def _bench_scoring(model, qt, docs, k, repeats):
    """Return mean docs/sec and ms per query for scoring `qt` against `k` docs."""
    idx = np.random.default_rng(0).choice(len(docs), size=k, replace=False)
    pool = [docs[i] for i in idx]

    if hasattr(model, "score_pool"):            # batched scorer (LTR)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model.score_pool(qt, pool)
        dt = time.perf_counter() - t0
    elif hasattr(model, "score_prepared"):      # prepare-once scorer (CXM25)
        ctx = model.prepare(qt)
        t0 = time.perf_counter()
        for _ in range(repeats):
            for d in pool:
                model.score_prepared(ctx, d)
        dt = time.perf_counter() - t0
    else:                                       # BM25
        t0 = time.perf_counter()
        for _ in range(repeats):
            for d in pool:
                model.score(qt, d)
        dt = time.perf_counter() - t0

    return k * repeats / dt, dt / repeats * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=None)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    norm = Normalizer(lang="pt")
    raw = _load_docs(args.docs, args.text_col)
    print(f"docs loaded: {len(raw)}", flush=True)

    # ---- indexing throughput (tokenization, shared by all models) ----
    t0 = time.perf_counter()
    docs = [tokenize_doc(norm, d) for d in raw[:20000]]
    idx_dt = time.perf_counter() - t0
    index_rate = len(docs) / idx_dt
    print(f"indexing: {index_rate:.0f} docs/s (tokenize, single core)")

    # ---- scorer setup ----
    stats = cxm25.load_baked_stats()
    avg_len = cxm25.load_baked_meta()["avg_len"]
    bm = BM25(stats.df, stats.N, avg_len)
    cx = cxm25.load_baked_cxm25()
    ltr = cxm25.load_baked_ltr()

    qs = [norm(q) for q in _QUERIES] * ((args.queries // len(_QUERIES)) + 1)
    qs = [q for q in qs[:args.queries] if q]
    print(f"queries: {len(qs)}  candidates/query: {args.k}\n")
    print(f"{'model':10s} {'docs/s':>12s} {'ms/query@k':>12s} {'vs BM25':>9s}")
    print("-" * 48)

    results = {}
    for name, model in [("BM25", bm), ("CXM25", cx), ("CXM25-LTR", ltr)]:
        # warm-up
        _bench_scoring(model, qs[0], docs, min(50, len(docs)), 1)
        rates, lats = [], []
        for q in qs:
            r, lat = _bench_scoring(model, q, docs, args.k, args.repeats)
            rates.append(r)
            lats.append(lat)
        docps = float(np.mean(rates))
        lat = float(np.mean(lats))
        results[name] = (docps, lat)
        if name == "BM25":
            rel = "reference"
        else:
            rel = f"{results['BM25'][0] / docps:.0f}x slower"
        print(f"{name:10s} {docps:12,.0f} {lat:12.1f} {rel:>12s}", flush=True)

    print(f"\nindexing: {index_rate:.0f} docs/s (single core, tokenize only)")


if __name__ == "__main__":
    main()
