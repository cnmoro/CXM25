"""Vendored Portuguese Snowball stemmer (pure stdlib, no dependencies).

Faithful port of the well-tested NLTK ``_PortugueseStemmer``
(``nltk/stem/snowball.py``, NLTK project, Apache-2.0) plus its region helpers
from ``nltk/stem/_StandardStemmer`` and ``suffix_replace`` from
``nltk/stem/util.py``. The algorithm is the standard Snowball Portuguese
stemmer (snowballstem.org/algorithms/portuguese/stemmer.html).

The pipeline normalizes Unicode accents *before* stemming, so the stemmer is
normally fed ASCII-only, accent-folded tokens. The accented suffix patterns
are kept for robustness when accented text is passed directly.

Only ``stem()`` is needed by the public API; everything else is internal.
"""

import re

__all__ = ["PortugueseStemmer"]

_VOWELS = "aeiouáéíóúâêô"


def _suffix_replace(original, old, new):
    return original[: -len(old)] + new


def _r1r2(word, vowels):
    """R1: region after the first non-vowel following a vowel.
    R2: same rule applied inside R1."""
    r1 = ""
    r2 = ""
    for i in range(1, len(word)):
        if word[i] not in vowels and word[i - 1] in vowels:
            r1 = word[i + 1 :]
            break
    for i in range(1, len(r1)):
        if r1[i] not in vowels and r1[i - 1] in vowels:
            r2 = r1[i + 1 :]
            break
    return r1, r2


def _rv(word, vowels):
    """RV: if second letter is a consonant -> after next vowel;
    if first two letters are vowels -> after next consonant;
    otherwise -> after the third letter."""
    rv = ""
    if len(word) >= 2:
        if word[1] not in vowels:
            for i in range(2, len(word)):
                if word[i] in vowels:
                    rv = word[i + 1 :]
                    break
        elif word[0] in vowels and word[1] in vowels:
            for i in range(2, len(word)):
                if word[i] not in vowels:
                    rv = word[i + 1 :]
                    break
        else:
            rv = word[3:]
    return rv


_STEP1_SUFFIXES = (
    "amentos", "imentos", "uço~es", "amento", "imento", "adoras", "adores",
    "aço~es", "logias", "ências", "amente", "idades", "anças", "ismos",
    "istas", "adora", "aça~o", "antes", "ância", "logia", "uça~o", "ência",
    "mente", "idade", "ança", "ezas", "icos", "icas", "ismo", "ável",
    "ível", "ista", "osos", "osas", "ador", "ante", "ivas", "ivos", "iras",
    "eza", "ico", "ica", "oso", "osa", "iva", "ivo", "ira",
)

_STEP2_SUFFIXES = (
    "aríamos", "eríamos", "iríamos", "ássemos", "êssemos", "íssemos",
    "aríeis", "eríeis", "iríeis", "ásseis", "ésseis", "ísseis", "áramos",
    "éramos", "íramos", "ávamos", "aremos", "eremos", "iremos", "ariam",
    "eriam", "iriam", "assem", "essem", "issem", "ara~o", "era~o", "ira~o",
    "arias", "erias", "irias", "ardes", "erdes", "irdes", "asses", "esses",
    "isses", "astes", "estes", "istes", "áreis", "areis", "éreis", "ereis",
    "íreis", "ireis", "áveis", "íamos", "armos", "ermos", "irmos", "aria",
    "eria", "iria", "asse", "esse", "isse", "aste", "este", "iste", "arei",
    "erei", "irei", "aram", "eram", "iram", "avam", "arem", "erem", "irem",
    "ando", "endo", "indo", "adas", "idas", "arás", "aras", "erás", "eras",
    "irás", "avas", "ares", "eres", "ires", "íeis", "ados", "idos", "ámos",
    "amos", "emos", "imos", "iras", "ada", "ida", "ará", "ara", "erá",
    "era", "irá", "ava", "iam", "ado", "ido", "ias", "ais", "eis", "ira",
    "ia", "ei", "am", "em", "ar", "er", "ir", "as", "es", "is", "eu",
    "iu", "ou",
)

_STEP4_SUFFIXES = ("os", "a", "i", "o", "á", "í", "ó")


class PortugueseStemmer:
    """Port of NLTK's Portuguese Snowball stemmer.

    Example:
        >>> PortugueseStemmer().stem("correndo")
        'corr'
    """

    def stem(self, word):
        word = word.lower()

        word = (
            word.replace("ã", "a~")
            .replace("õ", "o~")
            .replace("qü", "qu")
            .replace("gü", "gu")
        )

        r1, r2 = _r1r2(word, _VOWELS)
        rv = _rv(word, _VOWELS)

        step1_success = False
        step2_success = False

        # STEP 1: standard suffix removal
        for suffix in _STEP1_SUFFIXES:
            if word.endswith(suffix):
                if suffix == "amente" and r1.endswith(suffix):
                    step1_success = True
                    word = word[:-6]
                    r2 = r2[:-6]
                    rv = rv[:-6]
                    if r2.endswith("iv"):
                        word = word[:-2]
                        r2 = r2[:-2]
                        rv = rv[:-2]
                        if r2.endswith("at"):
                            word = word[:-2]
                            rv = rv[:-2]
                    elif r2.endswith(("os", "ic", "ad")):
                        word = word[:-2]
                        rv = rv[:-2]

                elif (
                    suffix in ("ira", "iras")
                    and rv.endswith(suffix)
                    and word[-len(suffix) - 1 : -len(suffix)] == "e"
                ):
                    step1_success = True
                    word = _suffix_replace(word, suffix, "ir")
                    rv = _suffix_replace(rv, suffix, "ir")

                elif r2.endswith(suffix):
                    step1_success = True
                    if suffix in ("logia", "logias"):
                        word = _suffix_replace(word, suffix, "log")
                        rv = _suffix_replace(rv, suffix, "log")
                    elif suffix in ("uça~o", "uço~es"):
                        word = _suffix_replace(word, suffix, "u")
                        rv = _suffix_replace(rv, suffix, "u")
                    elif suffix in ("ência", "ências"):
                        word = _suffix_replace(word, suffix, "ente")
                        rv = _suffix_replace(rv, suffix, "ente")
                    elif suffix == "mente":
                        word = word[:-5]
                        r2 = r2[:-5]
                        rv = rv[:-5]
                        if r2.endswith(("ante", "avel", "ivel")):
                            word = word[:-4]
                            rv = rv[:-4]
                    elif suffix in ("idade", "idades"):
                        word = word[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                        rv = rv[: -len(suffix)]
                        if r2.endswith(("ic", "iv")):
                            word = word[:-2]
                            rv = rv[:-2]
                        elif r2.endswith("abil"):
                            word = word[:-4]
                            rv = rv[:-4]
                    elif suffix in ("iva", "ivo", "ivas", "ivos"):
                        word = word[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                        rv = rv[: -len(suffix)]
                        if r2.endswith("at"):
                            word = word[:-2]
                            rv = rv[:-2]
                    else:
                        word = word[: -len(suffix)]
                        rv = rv[: -len(suffix)]
                break

        # STEP 2: verb suffixes
        if not step1_success:
            for suffix in _STEP2_SUFFIXES:
                if rv.endswith(suffix):
                    step2_success = True
                    word = word[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    break

        # STEP 3
        if step1_success or step2_success:
            if rv.endswith("i") and word[-2] == "c":
                word = word[:-1]
                rv = rv[:-1]

        # STEP 4: residual suffix
        if not step1_success and not step2_success:
            for suffix in _STEP4_SUFFIXES:
                if rv.endswith(suffix):
                    word = word[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    break

        # STEP 5
        if rv.endswith(("e", "é", "ê")):
            word = word[:-1]
            rv = rv[:-1]
            if (word.endswith("gu") and rv.endswith("u")) or (
                word.endswith("ci") and rv.endswith("i")
            ):
                word = word[:-1]
        elif word.endswith("ç"):
            word = _suffix_replace(word, "ç", "c")

        word = word.replace("a~", "ã").replace("o~", "õ")
        return word
