"""End-to-end usage of CXM25 (pure), the baked CXM25-LTR model, and training
your own CXM25-LTR model on a small corpus.

Run:  python examples/demo.py             # core + baked LTR
      python examples/demo.py --train     # also train your own toy model
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cxm25
from cxm25 import BM25, CXM25, Normalizer, build_corpus_stats

# Realistic, passage-length documents (close to the distribution the baked
# model was trained on).
DOCS = [
    "As melhores praias da California para o clima quente do inverno estão ao longo da "
    "costa sul, particularmente as margens viradas para o sul, onde as temperaturas "
    "raramente caem abaixo dos dezoito graus mesmo em dezembro.",
    "A California é um estado dos Estados Unidos localizado na costa oeste do pais, "
    "fazendo fronteira com o Oregon ao norte, Nevada ao leste e o Mexico ao sul.",
    "A anemia é um termo medico que se refere a um numero reduzido de globulos "
    "vermelhos circulantes no sangue e pode ser causada por deficiencias "
    "nutricionais ou por outras doencas cronicas.",
    "Praias quentes no Brasil em dezembro atraem milhares de turistas europeus que "
    "fogem do inverno e procuram o calor do verão brasileiro durante as festas de fim de ano.",
    "O clima do sul da California raramente fica frio mesmo no inverno, com dias "
    "ensolarados e temperaturas amenas que tornam a regiao popular durante todo o ano.",
]

QUERIES = {
    "praias da california em dezembro": [0],        # doc 0 answers it
    "onde fica o estado da california": [1],
    "o que causa a anemia": [2],
    "praias quentes no brasil": [3],
    "clima do sul da california no inverno": [4],
}


def main(train=False):
    print(f"cxm25 v{cxm25.__version__}\n")

    norm = Normalizer(lang="pt")
    print("query tokens:", {q: norm(q) for q in list(QUERIES)[:2]})

    # 1) corpus statistics for the local corpus
    stats = build_corpus_stats(DOCS, gram_n=3)
    avg_len = sum(len(norm(d)) for d in DOCS) / len(DOCS)
    print(f"local corpus: N={stats.N} terms={len(stats.df)} grams={len(stats.gdf)} avg_len={avg_len:.1f}\n")

    from cxm25.stats import tokenize_doc

    def toks(doc):
        return tokenize_doc(norm, doc)

    # 2) reference baseline and 3) the CXM25 algorithm
    bm = BM25(stats.df, stats.N, avg_len)
    cx = CXM25(stats.df, stats.df2, stats.N, avg_len, gdf=stats.gdf)

    q = list(QUERIES)[0]
    qt = norm(q)
    print(f"ranking for query: {q!r}")
    print(f"{'doc':62s} {'BM25':>8s} {'CXM25':>8s}")
    for i, d in enumerate(DOCS):
        dtok = toks(d)
        print(f"[{i}] {d[:58]:60s} {bm.score(qt, dtok):8.3f} {cx.score(qt, dtok):8.3f}")

    # 4) the baked CXM25-LTR model (trained on 12M MS MARCO PT-BR triplets)
    print("\n--- CXM25-LTR (baked model, requires xgboost) ---")
    scorer = cxm25.load_baked_ltr()
    scores = scorer.score_pool(qt, [toks(d) for d in DOCS])
    for i, (d, s) in enumerate(zip(DOCS, scores)):
        print(f"[{i}] {d[:58]:60s} {float(s):8.3f}")

    if train:
        # 5) train YOUR OWN model on your own (query, positive, negative)
        #    triplets. Labels are generated here from the QUERIES map.
        print("\n--- training a toy CXM25-LTR model on local triplets ---")
        query_terms = set()
        for q_ in QUERIES:
            query_terms.update(norm(q_))
        coindex = cxm25.build_coindex(DOCS, query_terms)

        triplets = []
        for q_, pos_idx in QUERIES.items():
            for neg_idx in range(len(DOCS)):
                if neg_idx != pos_idx[0]:
                    triplets.append((q_, DOCS[pos_idx[0]], DOCS[neg_idx]))
        print(f"generated {len(triplets)} training triplets")

        from cxm25 import LTRScorer, train_ltr

        model = train_ltr(
            triplets, stats, avg_len, coindex=coindex,
            n_jobs=1, max_rounds=400, max_depth=5, eta=0.15,
            min_child_weight=1, eval_fraction=0.0,
            out_path="/tmp/cxm25_demo.json",
        )
        own = LTRScorer("/tmp/cxm25_demo.json", stats, avg_len, coindex=coindex)
        print(f"own LTR scores for {q!r}:")
        for i, (d, s) in enumerate(zip(DOCS, own.score_pool(qt, [toks(d) for d in DOCS]))):
            print(f"[{i}] {d[:58]:60s} {float(s):8.3f}")


if __name__ == "__main__":
    main(train="--train" in sys.argv)
