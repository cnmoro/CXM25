"""CXM25-LTR: the learned refinement of CXM25.

A gradient-boosted ranker (XGBoost) over the generic lexical features from
:mod:`cxm25.features`, trained on (query, positive, negative) triplets. No
embeddings are used anywhere.

This module requires the optional ``xgboost`` dependency
(``pip install cxm25[ltr]``). Core scoring does not.
"""

import os
import tempfile

import numpy as np

from .features import FEATURES, N_FEATURES, FeatureExtractor
from .stats import CorpusStats, tokenize_doc
from .textnorm import Normalizer

__all__ = ["LTRScorer", "train_ltr"]

_FEATS = FEATURES


def _require_xgb():
    try:
        import xgboost
    except ImportError:  # pragma: no cover
        raise ImportError(
            "CXM25-LTR requires the optional dependency xgboost. "
            "Install it with `pip install cxm25[ltr]`."
        )
    return xgboost


class LTRScorer:
    """Score (query, document) pairs with a trained CXM25-LTR model.

    Args:
        model: an XGBoost Booster (or path to a saved ``.json`` model).
        stats: :class:`~cxm25.stats.CorpusStats` used for feature extraction.
        avg_len: average content-stem length of the collection.
        coindex: optional thesaurus used at training time (must match the
            one used to train the model).
    """

    def __init__(self, model, stats, avg_len, coindex=None):
        import xgboost
        if isinstance(model, (str, os.PathLike)):
            path = str(model)
            model = xgboost.Booster()
            model.load_model(path)
        self.model = model
        self.fe = FeatureExtractor(stats, avg_len, coindex=coindex)

    def predict(self, X):
        """Score a feature matrix (rows = documents for one query)."""
        import xgboost
        return self.model.predict(xgboost.DMatrix(np.asarray(X, dtype=np.float32),
                                                  feature_names=_FEATS))

    def score_pool(self, qt, docs):
        """Score one tokenized query against a list of tokenized documents."""
        ctx = self.fe.prepare(qt)
        X = np.zeros((len(docs), N_FEATURES), dtype=np.float32)
        for k, d in enumerate(docs):
            X[k] = self.fe.feats(ctx, d)
        return self.predict(X)

    def score(self, qt, doc):
        return float(self.score_pool(qt, [doc])[0])


def train_ltr(triplets, stats, avg_len, coindex=None, out_path=None,
              n_jobs=1, max_rounds=1500, early_stopping=60,
              eval_fraction=0.05, chunk_rows=20000, progress=None,
              max_depth=7, eta=0.06, min_child_weight=4, seed=42):
    """Train a CXM25-LTR model on ``(query, positive, negative)`` triplets.

    Args:
        triplets: iterable of ``(query_text, positive_text, negative_text)``.
        stats: :class:`~cxm25.stats.CorpusStats` (should cover the corpus the
            model will be applied to).
        avg_len: average content-stem length of the collection.
        coindex: optional thesaurus from :func:`~cxm25.stats.build_coindex`.
        out_path: optional path to save the trained model (``.json``).
        n_jobs: parallel workers for feature extraction.
        max_rounds: max boosting rounds.
        early_stopping: stop if the validation loss does not improve for this
            many rounds.
        eval_fraction: fraction of examples held out for early stopping.
        chunk_rows: feature extraction is chunked to bound memory.
        progress: optional callable ``(message) -> None``.
        max_depth / eta / min_child_weight: XGBoost hyper-parameters. On very
            small training sets lower ``min_child_weight`` (e.g. 1) so the
            trees can actually split.
        seed: random seed for sampling / splits.

    Returns:
        The trained XGBoost Booster (already saved to ``out_path`` if given).
    """
    xgb = _require_xgb()
    fe = FeatureExtractor(stats, avg_len, coindex=coindex)
    norm = Normalizer(lang="pt")

    rows = list(triplets)
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    if progress:
        progress(f"triplets: {len(rows)}")

    # ---- chunked, memory-safe feature extraction ----
    tmpdir = tempfile.mkdtemp(prefix="cxm25_ltr_")
    chunk_files = []
    n_pairs = 0
    for ci in range(0, len(rows), chunk_rows):
        chunk = rows[ci:ci + chunk_rows]
        Xp, Xn = _extract_chunk(fe, norm, chunk, n_jobs)
        fp = os.path.join(tmpdir, f"chunk_{ci}.npz")
        np.savez_compressed(fp, Xp=Xp, Xn=Xn)
        chunk_files.append(fp)
        n_pairs += len(Xp)
        if progress:
            progress(f"chunk {ci // chunk_rows + 1}: {n_pairs} pairs")

    Xp = np.concatenate([np.load(f)["Xp"] for f in chunk_files])
    Xn = np.concatenate([np.load(f)["Xn"] for f in chunk_files])
    for f in chunk_files:
        os.remove(f)
    os.rmdir(tmpdir)

    X = np.concatenate([Xp, Xn])
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xp))])
    if progress:
        progress(f"feature matrix: {X.shape}")

    perm = np.random.default_rng(seed + 1).permutation(len(X))
    if eval_fraction and eval_fraction > 0:
        nval = max(1, int(len(X) * eval_fraction))
        val_idx = perm[:nval]
        tr_idx = perm[nval:]
        dtr = xgb.DMatrix(X[tr_idx], label=y[tr_idx], feature_names=_FEATS)
        dva = xgb.DMatrix(X[val_idx], label=y[val_idx], feature_names=_FEATS)
        evals = [(dva, "val")]
        early = early_stopping
    else:
        dtr = xgb.DMatrix(X, label=y, feature_names=_FEATS)
        dva = None
        evals = None
        early = None

    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=max_depth,
        eta=eta,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=min_child_weight,
        tree_method="hist",
        nthread=n_jobs,
    )
    model = xgb.train(
        params, dtr, num_boost_round=max_rounds,
        evals=evals, early_stopping_rounds=early,
        verbose_eval=False,
    )
    if out_path:
        model.save_model(out_path)
    return model


def _extract_chunk(fe, norm, chunk, n_jobs):
    """Extract positive/negative feature matrices for a chunk of triplets."""
    if n_jobs > 1:
        from multiprocessing import Pool
        sub = np.array_split(np.arange(len(chunk)), min(n_jobs * 2, len(chunk)))
        tasks = [(fe, [(chunk[i][0], chunk[i][1], chunk[i][2]) for i in c])
                 for c in sub if len(c)]
        with Pool(n_jobs) as pool:
            parts = list(pool.imap(_worker, tasks))
        Xp = np.concatenate([p[0] for p in parts])
        Xn = np.concatenate([p[1] for p in parts])
        return Xp, Xn
    Xp, Xn = _worker((fe, chunk))
    return Xp, Xn


def _worker(args):
    fe, rows = args
    norm = Normalizer(lang="pt")
    cache = {}
    Xp = np.zeros((len(rows), N_FEATURES), dtype=np.float32)
    Xn = np.zeros((len(rows), N_FEATURES), dtype=np.float32)
    k = 0
    for q, p, n in rows:
        qt = norm(q)
        if not qt:
            continue
        pdoc = cache.get(p)
        if pdoc is None:
            pdoc = tokenize_doc(norm, p)
            if len(cache) < 300000:
                cache[p] = pdoc
        ndoc = cache.get(n)
        if ndoc is None:
            ndoc = tokenize_doc(norm, n)
            if len(cache) < 300000:
                cache[n] = ndoc
        ctx = fe.prepare(qt)
        Xp[k] = fe.feats(ctx, pdoc)
        Xn[k] = fe.feats(ctx, ndoc)
        k += 1
    return Xp[:k], Xn[:k]
