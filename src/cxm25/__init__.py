"""CXM25 — a better lexical retrieval model, BM25-inspired.

Two models are provided:

* :class:`~cxm25.scoring.CXM25` — the pure scoring algorithm (zero runtime
  dependencies). Components: query-normalized BM25-style unigram saturation,
  character-3-gram overlap, phrase runs, and leading-position ("title")
  weighting, all grounded in generic IR / linguistic techniques.
* :class:`~cxm25.ltr.LTRScorer` — the learned refinement (CXM25-LTR), a
  gradient-boosted ranker over 31 generic lexical features (requires the
  optional ``xgboost`` dependency).

A :class:`~cxm25.scoring.BM25` reference implementation is included for
baseline comparisons.
"""

from .baked import (
    load_baked_bm25,
    load_baked_coindex,
    load_baked_cxm25,
    load_baked_ltr,
    load_baked_meta,
    load_baked_stats,
)
from .rank import rank
from .scoring import BM25, CXM25, grams
from .stats import CorpusStats, build_coindex, build_corpus_stats
from .textnorm import Normalizer

try:  # optional heavy dependency
    from .features import FEATURES, FeatureExtractor
    from .ltr import LTRScorer, train_ltr
except ImportError:  # pragma: no cover
    FEATURES = None
    FeatureExtractor = None
    LTRScorer = None
    train_ltr = None

__version__ = "0.1.0"

__all__ = [
    "BM25",
    "CXM25",
    "CorpusStats",
    "Normalizer",
    "build_coindex",
    "build_corpus_stats",
    "grams",
    "load_baked_bm25",
    "load_baked_coindex",
    "load_baked_cxm25",
    "load_baked_ltr",
    "load_baked_meta",
    "load_baked_stats",
    "rank",
    "FEATURES",
    "FeatureExtractor",
    "LTRScorer",
    "train_ltr",
    "__version__",
]
