"""Lexical scoring: a BM25 reference and CXM25, the proposed algorithm.

CXM25 is a BM25-inspired lexical retrieval model whose components are all
generic IR / linguistic techniques (nothing dataset-specific):

  U  unigram component     BM25-style saturated term weights, query-normalized
  B  bi-term co-occurrence corpus PMI boost for query term pairs that co-occur
                           in the document (disabled by default: it did not
                           help on the reference collection)
  P  phrase component      longest contiguous query run found in the document
                           (word-order signal)
  T  position component    bonus when query terms appear in the document's
                           first ``title_len`` content stems (leading-sentence
                           / title signal)
  G  character-3-gram      sub-word overlap (robust to morphology, OOV terms,
                           typos and partial machine-translation artifacts)

score = U + beta*B + lam*P + tau*T + gamma*G

Documents are passed as pre-tokenized objects of the form
``(toks, tf, firstpos)`` where ``toks`` is the ordered content-stem list,
``tf`` a term-frequency dict and ``firstpos`` a dict term -> first position.
"""

import math

__all__ = ["BM25", "CXM25", "grams"]


def grams(tokens, gram_n=3):
    """Character-gram set of a token list (whole tokens shorter than
    ``gram_n`` are kept as-is, so short words still match exactly)."""
    out = set()
    for t in tokens:
        if len(t) <= gram_n:
            out.add(t)
            continue
        for i in range(len(t) - gram_n + 1):
            out.add(t[i:i + gram_n])
    return out


class BM25:
    """Classic BM25, provided as the reference baseline.

    Defaults are the values that were tuned on the reference collection
    (k1=0.8, b=0.8); pass explicit values to reproduce other settings.
    """

    def __init__(self, df, N, avg_len, k1=0.8, b=0.8):
        self.idf = {t: math.log(1 + (N - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.N = N
        self.avg_len = avg_len
        self.k1 = k1
        self.b = b

    def score(self, qt, doc):
        toks, tf, _ = doc
        k1, b = self.k1, self.b
        K = k1 * (1 - b + b * len(toks) / self.avg_len)
        idf = self.idf
        s = 0.0
        for t in qt:
            f = tf.get(t, 0)
            if f:
                s += idf.get(t, 0.0) * f * (k1 + 1) / (f + K)
        return s

    def prepare(self, qt):
        return qt

    def score_prepared(self, qt, doc):
        return self.score(qt, doc)


class CXM25:
    """The CXM25 lexical retrieval model.

    Defaults are the tuned values from the reference collection
    (``k1=1.2, b=0.5, beta=0.0, lam=0.8, tau=1.0, gamma=2.5, title_len=12``,
    ``idf_cap=10.0``, ``gram_n=3``). The co-occurrence component ``B`` is off
    by default because it slightly *hurt* accuracy on the reference data.
    """

    def __init__(self, df, df2, N, avg_len, gdf=None,
                 k1=1.2, b=0.5, beta=0.0, lam=0.8, tau=1.0, gamma=2.5,
                 rho_cap=1.5, title_len=12, idf_cap=10.0, idf_floor=0.2,
                 gram_n=3):
        self.N = N
        self.avg_len = avg_len
        self.k1 = k1
        self.b = b
        self.beta = beta
        self.lam = lam
        self.tau = tau
        self.gamma = gamma
        self.title_len = title_len
        self.gram_n = gram_n

        self.idf = {}
        for t, f in df.items():
            v = math.log(1 + (N - f + 0.5) / (f + 0.5))
            if idf_cap is not None:
                v = min(v, idf_cap)
            self.idf[t] = max(v, idf_floor)

        # pair PMI (positive part only), only built when the B component is used
        self.pmi = {}
        for p, c in df2.items():
            a, b_ = p
            pa = df.get(a, 1) / N
            pb = df.get(b_, 1) / N
            pm = math.log((c / N) / (pa * pb) + 1e-12)
            if pm > 0:
                self.pmi[p] = min(pm, rho_cap)

        self.gidf = {}
        if gdf:
            for g, f in gdf.items():
                v = math.log(1 + (N - f + 0.5) / (f + 0.5))
                if idf_cap is not None:
                    v = min(v, idf_cap)
                self.gidf[g] = max(v, 0.1)

    def _qweights(self, qt):
        idf = self.idf
        ws = [idf.get(t, 0.0) for t in qt]
        tot = sum(ws)
        nq = len(qt)
        if nq == 0:
            return []
        return [w / tot for w in ws] if tot > 0 else [1.0 / nq] * nq

    def prepare(self, qt):
        """Precompute the query-side of scoring once, so one query can be
        scored cheaply against many documents."""
        qg = grams(qt, self.gram_n) if (self.gamma > 0 and self.gidf) else None
        return {"qt": qt, "nq": len(qt), "ws": self._qweights(qt), "qg": qg}

    def score(self, qt, doc):
        return self.score_prepared(self.prepare(qt), doc)

    def score_prepared(self, ctx, doc):
        qt = ctx["qt"]
        ws = ctx["ws"]
        nq = ctx["nq"]
        qg = ctx["qg"]
        toks, tf, firstpos = doc
        if nq == 0:
            return 0.0

        k1, b = self.k1, self.b
        K = k1 * (1 - b + b * len(toks) / self.avg_len)

        sat = [0.0] * nq
        matched = [False] * nq
        for i, t in enumerate(qt):
            f = tf.get(t, 0)
            if f:
                sat[i] = f * (k1 + 1) / (f + K)
                matched[i] = True

        # ---- U: unigram (query-normalized weights) ----
        U = sum(ws[i] * sat[i] for i in range(nq))

        # ---- B: bi-term co-occurrence ----
        B = 0.0
        if self.beta > 0 and nq >= 2:
            pmi = self.pmi
            for i in range(nq - 1):
                if not matched[i]:
                    continue
                ti = qt[i]
                for j in range(i + 1, nq):
                    if not matched[j]:
                        continue
                    p = (ti, qt[j]) if ti < qt[j] else (qt[j], ti)
                    rho = pmi.get(p, 0.0)
                    if rho > 0:
                        B += 2.0 * ws[i] * ws[j] * sat[i] * sat[j] * (1.0 + rho)

        # ---- P: longest contiguous query run in doc ----
        P = 0.0
        if any(matched):
            posmap = {}
            for idx, t in enumerate(toks):
                posmap.setdefault(t, []).append(idx)
            best_L = 0
            best_w = 0.0
            for start in range(nq):
                if not matched[start]:
                    continue
                for pos in posmap.get(qt[start], []):
                    L = 1
                    wsum = ws[start]
                    k = 1
                    while start + k < nq and pos + k < len(toks):
                        if qt[start + k] == toks[pos + k]:
                            L += 1
                            wsum += ws[start + k]
                            k += 1
                        else:
                            break
                    if L > best_L or (L == best_L and wsum > best_w):
                        best_L = L
                        best_w = wsum
            if best_L > 0:
                P = best_w * (best_L / nq)

        # ---- T: title / leading-position component ----
        T = 0.0
        Tn = self.title_len
        for i in range(nq):
            if matched[i] and firstpos.get(qt[i], 1e9) < Tn:
                T += ws[i]

        # ---- G: character-gram overlap (sub-word matching) ----
        G = 0.0
        if self.gamma > 0 and qg is not None:
            if qg:
                dg = grams(toks, self.gram_n)
                gidf = self.gidf
                num = 0.0
                den = 0.0
                for g in qg:
                    w = gidf.get(g, 0.0)
                    if w <= 0:
                        continue
                    den += w
                    if g in dg:
                        num += w
                if den > 0:
                    G = num / den

        return U + self.beta * B + self.lam * P + self.tau * T + self.gamma * G
