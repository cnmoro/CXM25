"""Generic lexical features for the learned (CXM25-LTR) variant.

Every feature is a standard IR signal derived from corpus statistics
(idf / co-occurrence / character grams); nothing is dataset-specific and no
embeddings are used. The feature list is fixed; a trained XGBoost model is
expected to consume exactly this ordering (see ``cxm25.ltr``).
"""

import math

from .scoring import grams

__all__ = ["FeatureExtractor", "FEATURES"]

FEATURES = [
    "bm25_raw", "U", "G", "P", "T",
    "cov_idf", "n_matched", "cov_frac", "doc_len", "len_ratio", "nq",
    "sum_idf_matched", "max_idf_matched", "min_idf_matched",
    "mean_sat", "max_sat",
    "pmi_sum", "pmi_max", "n_pairs_matched",
    "gram_unw", "title_count", "first_pos",
    "longest_run", "run_frac", "tf_sum", "rarest_matched",
    "exp_cov", "exp_n", "exp_top5", "exp_wsum",
    "win_cov",
]

N_FEATURES = len(FEATURES)


class FeatureExtractor:
    """Extract the 31-feature vector for a (query, document) pair.

    Args:
        stats: :class:`~cxm25.stats.CorpusStats`.
        avg_len: average content-stem length of the collection.
        coindex: optional thesaurus from
            :func:`~cxm25.stats.build_coindex` (enables the expansion
            features; the LTR model was trained with it).
        title_len: leading-window size for the title features.
        gram_n: character-gram size.
    """

    def __init__(self, stats, avg_len, coindex=None,
                 title_len=12, gram_n=3, idf_floor=0.2, idf_cap=10.0,
                 exp_k=8):
        N = stats.N
        df = stats.df
        df2 = stats.df2
        gdf = stats.gdf
        self.N = N
        self.avg_len = avg_len
        self.title_len = title_len
        self.gram_n = gram_n
        self.exp_k = exp_k

        self.idf = {}
        for t, f in df.items():
            v = math.log(1 + (N - f + 0.5) / (f + 0.5))
            if idf_cap is not None:
                v = min(v, idf_cap)
            self.idf[t] = max(v, idf_floor)

        self.pmi = {}
        for p, c in df2.items():
            a, b = p
            pa = df.get(a, 1) / N
            pb = df.get(b, 1) / N
            pm = math.log((c / N) / (pa * pb) + 1e-12)
            if pm > 0:
                self.pmi[p] = min(pm, 4.0)

        self.gidf = {}
        if gdf:
            for g, f in gdf.items():
                v = math.log(1 + (N - f + 0.5) / (f + 0.5))
                if idf_cap is not None:
                    v = min(v, idf_cap)
                self.gidf[g] = max(v, 0.1)

        self.coindex = coindex

    # -- query-side --------------------------------------------------------

    def _expansion(self, qt):
        if not self.coindex:
            return {}
        ex = {}
        for t in qt:
            for u, p in self.coindex.get(t, [])[:self.exp_k]:
                if u in qt:
                    continue
                if u not in ex or p > ex[u]:
                    ex[u] = p
        return ex

    def prepare(self, qt):
        """Precompute the query side once, then call :meth:`feats` per doc."""
        idf = self.idf
        ws = [idf.get(t, 0.0) for t in qt]
        tot = sum(ws)
        nq = len(qt)
        if nq == 0:
            return {"qt": qt, "ws": [], "nq": 0, "qg": set(), "qgi": {},
                    "pairs": [], "exp": {}}
        ws = [w / tot for w in ws] if tot > 0 else [1.0 / nq] * nq
        qg = grams(qt, self.gram_n) if self.gidf else set()
        qgi = {g: self.gidf[g] for g in qg if g in self.gidf}
        pairs = []
        for i in range(nq - 1):
            for j in range(i + 1, nq):
                a, b = qt[i], qt[j]
                pairs.append((a, b) if a < b else (b, a))
        return {"qt": qt, "ws": ws, "nq": nq, "qg": qg, "qgi": qgi,
                "pairs": pairs, "exp": self._expansion(qt)}

    # -- feature vector ----------------------------------------------------

    def feats(self, ctx, doc):
        qt = ctx["qt"]
        ws = ctx["ws"]
        nq = ctx["nq"]
        qg = ctx["qg"]
        qgi = ctx["qgi"]
        pairs = ctx["pairs"]
        exp = ctx["exp"]
        toks, tf, firstpos = doc
        doc_len = len(toks)
        idf = self.idf
        f = [0.0] * N_FEATURES
        idx = FEATURES.index

        k1, b = 1.5, 0.75
        K = k1 * (1 - b + b * doc_len / self.avg_len)
        sats = []
        matched = []
        for i, t in enumerate(qt):
            c = tf.get(t, 0)
            if c:
                sats.append(c * (k1 + 1) / (c + K))
                matched.append(i)
            else:
                sats.append(0.0)

        f[idx("U")] = sum(ws[i] * sats[i] for i in range(nq))
        f[idx("bm25_raw")] = sum(idf.get(qt[i], 0.0) * sats[i] for i in range(nq))

        wsum_matched = sum(ws[i] for i in matched)
        f[idx("cov_idf")] = wsum_matched
        f[idx("n_matched")] = len(matched)
        f[idx("cov_frac")] = len(matched) / nq if nq else 0.0
        f[idx("doc_len")] = doc_len
        f[idx("len_ratio")] = doc_len / self.avg_len
        f[idx("nq")] = nq

        m_idf = [idf.get(qt[i], 0.0) for i in matched]
        f[idx("sum_idf_matched")] = sum(m_idf)
        f[idx("max_idf_matched")] = max(m_idf) if m_idf else 0.0
        f[idx("min_idf_matched")] = min(m_idf) if m_idf else 0.0
        f[idx("mean_sat")] = sum(sats[i] for i in matched) / nq if nq else 0.0
        f[idx("max_sat")] = max((sats[i] for i in matched), default=0.0)
        f[idx("tf_sum")] = sum(tf.get(qt[i], 0) for i in matched)

        pmi_sum = pmi_max = n_pm = 0.0
        pmi = self.pmi
        for i in range(nq - 1):
            for j in range(i + 1, nq):
                if sats[i] > 0 and sats[j] > 0:
                    a, b = qt[i], qt[j]
                    p = (a, b) if a < b else (b, a)
                    pm = pmi.get(p, 0.0)
                    if pm > 0:
                        pmi_sum += ws[i] * ws[j] * pm
                        pmi_max = max(pmi_max, pm)
                        n_pm += 1
        f[idx("pmi_sum")] = pmi_sum
        f[idx("pmi_max")] = pmi_max
        f[idx("n_pairs_matched")] = n_pm

        if qgi:
            dg = grams(toks, self.gram_n)
            num = den = 0.0
            for g, w in qgi.items():
                den += w
                if g in dg:
                    num += w
            f[idx("G")] = num / den if den else 0.0
            f[idx("gram_unw")] = len(qg & dg) / len(qg) if qg else 0.0

        T = 0.0
        tcnt = 0
        Tn = self.title_len
        for i in matched:
            if firstpos.get(qt[i], 1e9) < Tn:
                T += ws[i]
                tcnt += 1
        f[idx("T")] = T
        f[idx("title_count")] = tcnt

        longest = 0
        if matched:
            posmap = {}
            for i_, t in enumerate(toks):
                posmap.setdefault(t, []).append(i_)
            for start in matched:
                for pos in posmap.get(qt[start], []):
                    L = 1
                    k = 1
                    while start + k < nq and pos + k < doc_len:
                        if qt[start + k] == toks[pos + k]:
                            L += 1
                            k += 1
                        else:
                            break
                    longest = max(longest, L)
        f[idx("P")] = (wsum_matched * (longest / nq)) if nq else 0.0
        f[idx("longest_run")] = longest
        f[idx("run_frac")] = longest / nq if nq else 0.0

        f[idx("first_pos")] = min((firstpos.get(qt[i], 1e9) for i in matched), default=0.0)

        if m_idf:
            rarest = max(range(nq), key=lambda i: idf.get(qt[i], 0.0))
            f[idx("rarest_matched")] = 1.0 if rarest in matched else 0.0

        if exp:
            tot_w = 0.0
            wsum = 0.0
            n = 0
            top5_w = 0.0
            top5_tot = 0.0
            for k2, (u, p) in enumerate(sorted(exp.items(), key=lambda x: -x[1])):
                w = idf.get(u, 0.2)
                tot_w += w
                if u in tf:
                    wsum += w
                    n += 1
                if k2 < 5:
                    top5_tot += w
                    if u in tf:
                        top5_w += w
            f[idx("exp_cov")] = wsum / tot_w if tot_w else 0.0
            f[idx("exp_n")] = n
            f[idx("exp_top5")] = top5_w / top5_tot if top5_tot else 0.0
            f[idx("exp_wsum")] = wsum

        if nq >= 2:
            posmap = {}
            for i_, t in enumerate(toks):
                posmap.setdefault(t, []).append(i_)
            W = 15
            best = 0.0
            for i in range(nq):
                for p in posmap.get(qt[i], []):
                    cnt = 1
                    for j in range(nq):
                        if j == i:
                            continue
                        for p2 in posmap.get(qt[j], []):
                            if abs(p2 - p) < W:
                                cnt += 1
                                break
                    best = max(best, cnt / nq)
            f[idx("win_cov")] = best

        return f
