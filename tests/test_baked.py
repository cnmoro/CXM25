"""Tests for the baked (shipped) CXM25-LTR model. Requires xgboost."""

import pytest

pytest.importorskip("xgboost")

import cxm25
from cxm25 import Normalizer
from cxm25.stats import tokenize_doc


def _toks(text):
    return tokenize_doc(Normalizer(lang="pt"), text)


def test_baked_assets_load():
    meta = cxm25.load_baked_meta()
    stats = cxm25.load_baked_stats()
    assert stats.N > 1_000_000
    assert "avg_len" in meta
    assert meta["features"][:3] == ["bm25_raw", "U", "G"]


def test_baked_cxm25_scores():
    cx = cxm25.load_baked_cxm25()
    q = cxm25.Normalizer()("praias da california em dezembro")
    assert cx.score(q, _toks("as praias da california no inverno")) > cx.score(
        q, _toks("a anemia reduz os globulos vermelhos"))


def test_baked_ltr_is_deterministic_and_discriminates():
    scorer = cxm25.load_baked_ltr()
    q = cxm25.Normalizer()("praias da california em dezembro")
    rel = _toks("as melhores praias da california em dezembro ficam na costa sul "
                "e sao ideais para a temporada de verao que atrai milhares de "
                "turistas brasileiros todos os anos")
    irrel = _toks("a anemia e um termo medico que se refere a um numero reduzido "
                  "de globulos vermelhos circulantes no sangue e pode ser causada "
                  "por diversas doencas ou deficiencias nutricionais graves")
    s1 = scorer.score(q, rel)
    s2 = scorer.score(q, rel)
    s3 = scorer.score(q, irrel)
    assert s1 == s2                      # deterministic
    assert s1 > s3                       # topic doc ranks above unrelated doc


def test_baked_ltr_reproduces_pairwise_on_sample():
    """The baked model must rank the positive above the negative for the
    majority of a held-out sample of reference triplets."""
    import os
    import pickle
    import numpy as np

    scorer = cxm25.load_baked_ltr()
    norm = Normalizer(lang="pt")
    cache_pkl = "/mnt/nvme1tb/Carlo/heuristic-mining/results/val_doc_cache.pkl"
    tri_parquet = "/mnt/nvme1tb/Carlo/heuristic-mining/results/val_triplets.parquet"
    vdf = None
    if not (os.path.exists(cache_pkl) and os.path.exists(tri_parquet)):
        pytest.skip("reference data not available in this environment")
    import pandas as pd
    with open(cache_pkl, "rb") as f:
        cache = pickle.load(f)
    tri = pd.read_parquet(tri_parquet)
    vdf = pd.read_parquet(
        "/mnt/nvme1tb/Carlo/heuristic-mining/data/data/validation-00000-of-00001.parquet")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(tri), 2000, replace=False)
    ok = 0
    n = 0
    for i in idx:
        qt = norm(vdf["anchor"].iloc[i])
        if not qt:
            continue
        sp = scorer.score(qt, cache[tri["p"].iloc[i]])
        sn = scorer.score(qt, cache[tri["n"].iloc[i]])
        ok += sp > sn
        n += 1
    acc = ok / n
    assert acc > 0.75, f"baked model pairwise accuracy too low: {acc:.3f}"
