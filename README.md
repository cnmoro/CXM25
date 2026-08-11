# CXM25 & CXM25-LTR

**BM25-inspired lexical retrieval with no embeddings.** A drop-in replacement
for BM25 scoring that is significantly more accurate on hard retrieval pairs,
plus a learned refinement (`CXM25-LTR`) that ships with a pre-trained model.

The core algorithm (`CXM25`) has **zero runtime dependencies** (pure Python
standard library, including a vendored Portuguese Snowball stemmer). The
learned variant (`CXM25-LTR`) needs only the optional `xgboost` dependency.

| Model | Pairwise accuracy * | Error reduction vs BM25 | Relative scoring speed |
|------:|--------------------:|------------------------:|-----------------------:|
| BM25 (tuned) | 0.6734 | — | 1× (reference) |
| **CXM25** (pure, no training) | **0.7048** | 9.6% | ~47× slower |
| **CXM25-LTR** (baked model, 12M triplets) | **0.7775** | 31.9% | ~124× slower |

\* fraction of `(query, positive, negative)` triples where the positive is
scored above the negative, measured on the held-out validation split of
`cnmoro/AllTripletsMsMarco-PTBR` (527,832 triples). On the per-query
retrieval task (positive ranked #1 among all of the query's candidate
documents), BM25 hits **64.4%** and CXM25-LTR hits **70.2%**. Throughput is
measured on real passages with `examples/benchmark.py` (see
[Performance](#performance)).

---

## Table of contents

1. [What it is](#what-it-is)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [The baked model](#the-baked-model)
5. [Training your own model](#training-your-own-model)
6. [How it works](#how-it-works)
7. [The learned variant (CXM25-LTR)](#the-learned-variant-cxm25-ltr)
8. [How it was developed](#how-it-was-developed)
9. [Technical details](#technical-details)
10. [Performance](#performance)
11. [Limitations and the "80%" question](#limitations-and-the-80-question)
12. [Reproducing the reference numbers](#reproducing-the-reference-numbers)
13. [License](#license)

---

## What it is

> **What does CXM25 stand for?** "Co-occurrence-eXtended Matching 25". The
> *25* mirrors **BM25** (BM + 25), signalling that this is the next step in
> that lineage. The "co-occurrence" part is a leftover from the original
> design: a PMI-weighted bi-term co-occurrence component that was later
> dropped because it hurt accuracy, so the name stuck even though it no longer
> appears in the final score.

CXM25 is a lexical (term-overlap) retrieval model designed as a better BM25.
It keeps everything that makes BM25 good — term saturation, document-length
normalisation, IDF weighting — and fixes its main weaknesses by adding
generic IR components that BM25 lacks:

- **character-gram matching** (robust to morphology, typos, OOV terms and
  machine-translation noise),
- **phrase / word-order matching**,
- **leading-position ("title") weighting**,
- **query-normalised term weights** (so a document covering the whole query
  beats one that covers a single rare term).

Everything is grounded in standard IR and NLP techniques. There are **no
embeddings, no neural networks, no GPU**, and no dataset-specific rules.
`CXM25-LTR` additionally learns *how to combine* these signals from triplets
using gradient-boosted decision trees (XGBoost) — still fully lexical.

The package ships:

- `cxm25.rank(query, docs, ...)` — the one-call API (tokenizes, builds
  statistics, scores and returns a ranked list).
- `cxm25.CXM25` — the pure algorithm, tuned defaults baked in.
- `cxm25.BM25` — a reference implementation for baseline comparisons.
- `cxm25.load_baked_ltr()` — the pre-trained `CXM25-LTR` model (12M MS MARCO
  PT-BR triplets) with the corpus statistics it was trained on, ready to use
  out of the box.
- `cxm25.train_ltr()` — train your own model on your own triplets.
- `cxm25.build_corpus_stats()`, `cxm25.build_coindex()` — build the corpus
  statistics the scorers need.

## Installation

```bash
pip install cxm25               # core (CXM25, BM25) — zero dependencies
pip install cxm25[ltr]          # + xgboost for the LTR variant
```

Requires Python ≥ 3.9.

## Quickstart

The one-call API does everything for you — tokenization, corpus statistics and
scoring:

```python
import cxm25

results = cxm25.rank(
    query="praias da california em dezembro",
    docs=[
        "As melhores praias da California para o clima quente do inverno ...",
        "A California é um estado dos Estados Unidos na costa oeste ...",
    ],
    lang="pt",
    gram_n=3,
)

for r in results:
    print(r["rank"], round(r["score"], 3), r["doc"])
```

`rank()` returns `[{"rank": int, "doc": str, "score": float}, ...]` sorted by
score. Pick the model with `model=`:

- `model="cxm25"` (default) — the pure algorithm; statistics are built from
  your documents automatically.
- `model="bm25"` — the reference baseline.
- `model="ltr"` — the baked CXM25-LTR model (requires xgboost). It uses the
  shipped reference statistics, so it works out of the box on Portuguese
  text, and is best used as a reranker over passage-length documents.

For production workloads (large corpora, repeated ranking), build the
statistics once and reuse a scorer instead of rebuilding them per call:

```python
from cxm25 import Normalizer, build_corpus_stats
from cxm25.scoring import CXM25

stats = build_corpus_stats(docs, gram_n=3)          # build once
avg_len = sum(len(Normalizer()(d)) for d in docs) / len(docs)
cx = CXM25(stats.df, stats.df2, stats.N, avg_len, gdf=stats.gdf)

qt = Normalizer()("praias da california em dezembro")
ctx = cx.prepare(qt)                                 # prepare the query once
for d in docs:                                       # ... then score each doc
    print(cx.score_prepared(ctx, d))
```

You can also pass a prebuilt `stats=` to `rank()`, or hand it an already-built
scorer via `scorer=`.

## The baked model

The package ships the model trained on 12M triplets of
`cnmoro/AllTripletsMsMarco-PTBR` (a Brazilian-Portuguese machine translation
of MS MARCO) together with the exact corpus statistics used at training time.
This means you can rank Portuguese text immediately, without building any
statistics:

```python
scorer = cxm25.load_baked_ltr()          # requires xgboost

q = norm("praias da california em dezembro")
scores = scorer.score_pool(q, [tokenize_doc(norm, d) for d in docs])
```

`score_pool(qt, docs)` scores a whole candidate list at once (feature
extraction is batched). `score(qt, doc)` scores a single document.

`load_baked_cxm25()` and `load_baked_bm25()` return the pure `CXM25` and a
`BM25` reference configured with the same baked statistics, so you can compare
all three models on your own text with three lines of code:

```python
cx = cxm25.load_baked_cxm25()
bm = cxm25.load_baked_bm25()
```

Model provenance is in `cxm25/data/meta.json` (accuracy, training corpus,
feature list, number of trees).

## Training your own model

`CXM25-LTR` is trained on `(query, positive, negative)` triplets — exactly the
format of the MS MARCO triplet datasets:

```python
from cxm25 import build_coindex, train_ltr, LTRScorer

query_terms = set(norm(q) for q in queries)
coindex = cxm25.build_coindex(docs, query_terms)   # distributional thesaurus

model = train_ltr(
    triplets,          # iterable of (query_text, positive_text, negative_text)
    stats,             # from build_corpus_stats
    avg_len,
    coindex=coindex,
    out_path="my_model.json",
    n_jobs=8,          # parallel feature extraction
    max_rounds=2500,
)

scorer = LTRScorer("my_model.json", stats, avg_len, coindex=coindex)
```

`train_ltr` streams feature extraction in bounded chunks (memory-safe on large
corpora — see [Technical details](#technical-details)). On small toy datasets,
pass `min_child_weight=1` and a higher `eta` so the trees can actually split
(see `examples/demo.py --train`).

## How it works

### Tokenisation

All text goes through the same pipeline (`cxm25.Normalizer`):

1. **Unicode accent folding** — `café` → `cafe`, `canção` → `cancao`. Keeps
   matched terms consistent even when a query and a document use different
   accent spellings (common in machine-translated text).
2. **Tokenise** on `[a-z0-9]+`.
3. **Stem** with a vendored Portuguese Snowball stemmer (`correndo` → `corr`,
   `praias` → `pra`), a faithful port of the NLTK implementation with zero
   dependencies.
4. **Drop stopwords** (a fixed Portuguese list — `de`, `o`, `que`, …).

### The CXM25 score

For a query with content terms `q = (t1, …, tn)` and a document `d`, CXM25
computes

```
score(q, d) = U + lam·P + tau·T + gamma·G
```

with tuned defaults `lam = 0.8`, `tau = 1.0`, `gamma = 2.5`.

**U — unigram component.** BM25's saturated term weight per matched query term,
with the terms' contributions normalised by their query-side importance:

```
U = Σ_i  ŵ_i · tf_i·(k1+1) / (tf_i + k1·(1 − b + b·|d|/avgdl))
```

where `ŵ_i = idf(t_i) / Σ_j idf(t_j)` (so the weights sum to 1). The
normalisation means a document covering *all* of the query inherently beats one
covering a single rare term — one of BM25's classic failure modes. IDF is
floored and capped to keep it well-behaved on noisy/translated text
(`idf_cap = 10`, floor `0.2`). Tuned on the reference corpus: `k1 = 1.2`,
`b = 0.5`.

**P — phrase component.** The longest contiguous run of query terms that
appears in the document, weighted by the run's share of the query:

```
P = (Σ run terms ŵ) · (run length / n)
```

This captures word order — a document that literally contains the query phrase
is almost always the answer.

**T — title / leading-position component.** Credit for query terms that appear
in the document's first `title_len = 12` content stems (the leading sentence of
a passage usually states its topic):

```
T = Σ_{i : firstpos(t_i) < title_len} ŵ_i
```

**G — character-gram component.** The strongest single component. For each
query term, its character-3-grams are formed (`praias` → `pra rai aia ias`).
The overlap between query grams and document grams is computed, weighted by
gram IDF:

```
G = Σ_{g ∈ query grams ∩ doc grams} gidf(g) / Σ_{g ∈ query grams} gidf(g)
```

Sub-word overlap recovers matches that stemming misses (`dentes` vs `dentário`),
foreign fragments, OOV acronyms (`srtp`), and translation artifacts — exactly
the cases that break pure term matching on this kind of data. A bi-term
co-occurrence component (PMI-weighted pairs) was implemented and **removed**
because it measurably *hurt* accuracy on the reference corpus.

### BM25 reference

`cxm25.BM25` is the standard Okapi formula with the tuned reference parameters
`k1 = 0.8`, `b = 0.8`. It exists so baselines are reproducible and CXM25 can be
compared against it on identical input.

## The learned variant (CXM25-LTR)

`CXM25-LTR` is a gradient-boosted binary classifier (XGBoost) over **31
generic lexical features** extracted per `(query, document)` pair. Every
feature is a standard IR signal derived from corpus statistics — no embeddings,
no hand-written dataset-specific rules:

| # | feature | # | feature |
|---:|---|---:|---|
| 0 | `bm25_raw` | 16 | `pmi_sum` |
| 1 | `U` | 17 | `pmi_max` |
| 2 | `G` | 18 | `n_pairs_matched` |
| 3 | `P` | 19 | `gram_unw` |
| 4 | `T` | 20 | `title_count` |
| 5 | `cov_idf` | 21 | `first_pos` |
| 6 | `n_matched` | 22 | `longest_run` |
| 7 | `cov_frac` | 23 | `run_frac` |
| 8 | `doc_len` | 24 | `tf_sum` |
| 9 | `len_ratio` | 25 | `rarest_matched` |
| 10 | `nq` | 26 | `exp_cov` |
| 11 | `sum_idf_matched` | 27 | `exp_n` |
| 12 | `max_idf_matched` | 28 | `exp_top5` |
| 13 | `min_idf_matched` | 29 | `exp_wsum` |
| 14 | `mean_sat` | 30 | `win_cov` |
| 15 | `max_sat` | | |

Features 26–29 use a **distributional thesaurus** (`cxm25.build_coindex`): for
each query term, the corpus terms that most often co-occur with it (ranked by
count-weighted PMI / local-MI) act as soft synonyms, so a document that talks
about the query's *topic* without using its exact words still receives some
credit. Feature 30 (`win_cov`) is the maximum fraction of query terms found
within a 15-token window of the document (a passage-level "the whole query is
answered in one place" signal).

Training feeds each `(query, positive, negative)` triple as two labelled
feature vectors (positive → 1, negative → 0). The baked model was trained on
**12,000,000 triples** with 2,500 trees (max depth 7, eta 0.06) and reaches
0.7775 pairwise accuracy on the held-out validation split.

## How it was developed

The model was developed against `cnmoro/AllTripletsMsMarco-PTBR`, a
Portuguese translation of the MS MARCO triplet corpus: 25.9M training and
527.8K validation triples of `(anchor, positive, negative)`. Queries average
6.5 words; passages about 60. The task is deliberately adversarial — negatives
share ~46% of their words with the query on average, so naive lexical scoring
struggles.

The development process was measurement-driven, with every idea ablated on a
fixed dev set and the final numbers reported on the held-out validation split:

1. **Baseline.** Tuned BM25 → **0.6734** pairwise.
2. **Preprocessing.** Accent folding + Snowball stemming + stopwords was the
   first large jump: Portuguese morphology (`correr / correu / correndo`) made
   raw token matching miss most inflections.
3. **Query normalisation + coverage.** Normalising term weights by query IDF
   stopped rare terms from dominating and made coverage meaningful.
4. **Character grams.** The biggest single component (+2.5 points): recovered
   morphological variants, OOV acronyms and machine-translation noise.
5. **Phrase and title components.** Small but consistent further gains.
6. **What did *not* help (and was dropped):**
   - **PMI bi-term co-occurrence** (the `B` component): consistently hurt by
     ~0.3 points — spurious co-occurrences in negatives were amplified.
   - **Aggressive thesaurus expansion as a soft-matching feature**: hurt by
     ~2 points; the expansion terms were common in negatives too.
   - **LambdaRank (rank:ndcg / rank:pairwise) on query groups**: no better
     than plain pairwise binary training.
   - **Pairwise-difference features** (score `feats(pos) − feats(neg)`):
     memorised the training pairs; worse on held-out data.
   - **Expansion as a rescue for zero-overlap positives**: measured at chance
     level — when a positive shares *no* words with the query, the negative is
     always lexically closer.
7. **Scaling.** More training data helped monotonically: 800K triples → 0.7375,
   2M → 0.7648, 5M → 0.7707, **12M → 0.7775**.
8. **Generalisation check.** Because 98% of validation queries also appear in
   train (the splits share queries), the model's accuracy on pairs whose
   `(query, positive)` combination was *never seen* in training is 0.7535 — the
   gain over BM25 is real, not memorisation.

Ablations, category analyses of remaining errors and the full progression are
documented in the project history under `src/` in the development repository.

## Technical details

### Corpus statistics

`cxm25.build_corpus_stats` computes, from a sample of documents:

- `N` — number of documents,
- `df` — term → document frequency (for IDF),
- `gdf` — character-gram → document frequency (for the `G` component),
- `df2` — (term, term) → co-document frequency, restricted to pairs that
  appear in the provided `query_terms` (that is all scoring ever needs; it
  keeps the pass cheap).

Sampling is deterministic (blake2b hash of the document text), so repeated
runs are reproducible. It accepts an iterable of documents, a text file (one
document per line) or a parquet file with a `text` column, and supports
multiprocessing via `n_jobs`.

### The distributional thesaurus

`cxm25.build_coindex` builds, for each query term, the top-`k` corpus terms
that co-occur with it in documents, ranked by **local-MI** (count × PMI).
Pure collection statistics — the same idea as a statistical thesaurus or
pseudo-relevance vocabulary. Only terms with ≥ `min_count` co-occurrences and
no digits are kept (generic filters).

### Memory safety

Training and statistics builders **never accumulate the dataset in RAM**.
Features are extracted in chunks and written to temporary `.npz` files, then
concatenated once for training. This was the critical engineering decision:
earlier memory-hungry versions OOM-killed the development server twice.

### Determinism

- Sampling uses `blake2b` hashes (not Python's per-process randomised `hash()`).
- Tokenisation and scoring are pure functions.
- XGBoost predictions are deterministic for a fixed model.

### Reproducibility of the baked model

The baked model's feature space must match the statistics it was trained with.
`load_baked_ltr()` wires the model to the exact shipped statistics, so it works
out of the box. If you build your *own* statistics (e.g. for a domain-specific
corpus), you should retrain the model on those statistics — the features are
corpus-relative (idf, gram frequencies, thesaurus).

## Performance

Throughput measured with `examples/benchmark.py` on the shipped reference
corpus statistics, scoring real validation passages of MS MARCO PT-BR (~60
words each) against short queries. Single process, Python 3.12, 16-core Linux
server (XGBoost predict is multithreaded; everything else single-threaded).
`docs/s` = tokenized documents scored per second; `ms/query@2k` = wall-clock
latency to score one query against 2,000 candidate documents.

| model | docs/s (score) | ms/query @ 2,000 docs | vs BM25 |
|------:|---------------:|----------------------:|--------:|
| BM25 | ~2,340,000 | ~1.0 | reference |
| CXM25 | ~50,000 | ~40 | ~47× slower |
| CXM25-LTR | ~19,000 | ~108 | ~124× slower |
| indexing (tokenize all) | ~1,400 docs/s | — | shared by all |

How to read this:

- **Preprocessing (tokenisation/stemming) is the same pipeline for every
  model**, so indexing cost is identical; the table isolates *scoring* cost.
- The scoring cost is Python overhead, not algorithmics: CXM25 pays for
  character-gram set construction and phrase scanning per document, CXM25-LTR
  additionally extracts the 31-feature vector per document.
- **For collection-scale retrieval, never score every document.** Use an
  inverted index over content stems (both BM25 and CXM25 are pure
  term-overlap scorers, so they use the same posting lists) and score only the
  candidates that share at least one query term. Per-query latency then scales
  with *matched* documents, not collection size. For a 1M-document index where
  ~2% of documents match a query term (~20K candidates): BM25 ≈ 9 ms, CXM25 ≈
  400 ms.
- **Use CXM25-LTR as a reranker.** Generate candidates with BM25/CXM25 (fast,
  index-friendly), then rerank the top 1,000 with the learned model: ~108 ms
  for 2,000 documents → ~54 ms for 1,000. This is the standard
  retrieve-then-rerank setup and keeps latency in the tens of milliseconds
  while adding the ~10 accuracy points over BM25.
- **Parallelise freely** — scoring is embarrassingly parallel across documents
  and queries (and tokenization across documents), so throughput scales with
  cores.

The accuracy/speed trade-off: BM25 is fastest but weakest; CXM25 is ~47×
slower than BM25 but already more accurate; CXM25-LTR is ~124× slower and
gives the largest accuracy gain. All three are orders of magnitude cheaper
than any embedding-based retriever (no GPU, no model servers).

## Limitations and the "80%" question

Two structural limits bound any no-embedding lexical model on this task:

- **~7% of positives share zero content terms with the query.** The connection
  is pure paraphrase; no amount of term/gram matching can find it (measured:
  thesauri cannot recover these).
- **~6.5% of triples are genuinely ambiguous**, with both documents covering
  100% of the query terms.

An 80% *relative* improvement over a 67% baseline would require >100%
accuracy (arithmetically impossible), and the standard reading — an 80%
reduction of BM25's error — would require ~93.5% accuracy, above this lexical
ceiling. The achievable, honest result is what is reported here: CXM25-LTR
reduces BM25's error by ~32% and closes ~32% of the gap to perfect retrieval,
with no embeddings.

## Reproducing the reference numbers

```bash
# dataset: https://huggingface.co/datasets/cnmoro/AllTripletsMsMarco-PTBR
# download the validation parquet + a sample of the train parquets, then:

python - <<'EOF'
import cxm25
from cxm25 import Normalizer
from cxm25.stats import tokenize_doc
import pandas as pd, glob

norm = Normalizer(lang="pt")
stats = cxm25.load_baked_stats()
avg_len = cxm25.load_baked_meta()["avg_len"]
scorer = cxm25.load_baked_ltr()
bm = cxm25.load_baked_bm25()

val = pd.read_parquet("data/validation-00000-of-00001.parquet")
ok_bm = ok_cx = 0; n = 0
for _, row in val.iterrows():
    qt = norm(row["anchor"])
    if not qt: continue
    pos = tokenize_doc(norm, row["positive"])
    neg = tokenize_doc(norm, row["negative"])
    ok_bm += bm.score(qt, pos) > bm.score(qt, neg)
    ok_cx += scorer.score(qt, pos) > scorer.score(qt, neg)
    n += 1
print("BM25 pairwise:", ok_bm / n)        # ~0.673
print("CXM25-LTR pairwise:", ok_cx / n)   # ~0.777
EOF
```

(For the full corpus statistics used at training time, build them with
`cxm25.build_corpus_stats(train_docs, query_terms=...)` over your train
collection — the shipped statistics correspond to a ~3M-document sample.)

## License

MIT. The vendored Portuguese stemmer is a port of NLTK's Snowball
implementation (NLTK project, Apache-2.0); the underlying Snowball algorithm
is BSD-licensed.
