"""Baked assets: the CXM25-LTR model and the corpus statistics it was trained
with, shipped inside the package.

The reference model was trained on 12M triplets of
``cnmoro/AllTripletsMsMarco-PTBR`` (machine-translated MS MARCO, Brazilian
Portuguese) and reaches 0.7775 pairwise accuracy on the held-out validation
split (BM25 reference: 0.6734). See ``data/meta.json`` for provenance.

Using the baked loaders means no corpus statistics have to be built by the
user: the exact feature space the model expects is already provided. Building
statistics for *your own* collection is still supported — see
:func:`cxm25.build_corpus_stats` and the ``examples/demo.py`` script.
"""

import json
import os
import pickle

from .scoring import BM25, CXM25
from .stats import CorpusStats

__all__ = [
    "DATA_DIR",
    "load_baked_stats",
    "load_baked_coindex",
    "load_baked_meta",
    "load_baked_bm25",
    "load_baked_cxm25",
    "load_baked_ltr",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _path(name):
    return os.path.join(DATA_DIR, name)


def load_baked_stats():
    """Return the :class:`~cxm25.stats.CorpusStats` used to train the model."""
    with open(_path("stats.pkl"), "rb") as f:
        raw = pickle.load(f)
    return CorpusStats(N=raw["N"], df=raw["df"], df2=raw["df2"], gdf=raw["gdf"])


def load_baked_coindex():
    """Return the distributional thesaurus used by the trained model."""
    with open(_path("coindex.pkl"), "rb") as f:
        return pickle.load(f)


def load_baked_meta():
    """Return the model metadata dict (avg_len, features, training notes)."""
    with open(_path("meta.json")) as f:
        return json.load(f)


def load_baked_bm25():
    """A BM25 scorer configured with the reference corpus statistics."""
    stats = load_baked_stats()
    return BM25(stats.df, stats.N, load_baked_meta()["avg_len"])


def load_baked_cxm25():
    """A ready-to-use :class:`~cxm25.scoring.CXM25` (pure algorithm) with the
    reference corpus statistics baked in."""
    stats = load_baked_stats()
    return CXM25(stats.df, stats.df2, stats.N, load_baked_meta()["avg_len"],
                 gdf=stats.gdf)


def load_baked_ltr():
    """A ready-to-use :class:`~cxm25.ltr.LTRScorer` (CXM25-LTR) with the
    reference model and corpus statistics baked in.

    Requires the optional ``xgboost`` dependency.
    """
    from .ltr import LTRScorer

    meta = load_baked_meta()
    stats = load_baked_stats()
    return LTRScorer(_path("model.ubj"), stats, meta["avg_len"],
                     coindex=load_baked_coindex())
