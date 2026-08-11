"""Text normalization: lower-case -> Unicode accent folding (NFKD) ->
tokenize -> drop stopwords -> Snowball stem.

All steps are generic NLP techniques parameterized by a language code; nothing
here is tuned to a specific dataset. Default language is Portuguese (the
library was developed and measured on a Portuguese collection), but other
languages can be plugged in by supplying a stemmer and a stopword set.
"""

import re
import unicodedata

from ._stemmer import PortugueseStemmer

_STOPWORDS_PT = frozenset({
    "a", "o", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "sob", "sobre",
    "e", "ou", "mas", "que", "se", "como", "quando", "onde", "porque", "porque",
    "nao", "mais", "menos", "muito", "muita", "muitos", "muitas", "pouco", "pouca",
    "tambem", "ao", "aos", "ao", "dela", "dele", "deles", "delas", "este", "esta",
    "estes", "estas", "esse", "essa", "esses", "essas", "aquele", "aquela", "aqueles",
    "aquelas", "isto", "isso", "aquilo", "eu", "tu", "ele", "ela", "nos", "vos",
    "eles", "elas", "meu", "minha", "meus", "minhas", "teu", "tua", "seus", "suas",
    "nosso", "nossa", "nossos", "nossas", "vosso", "vossa", "quem", "qual", "quais",
    "quanto", "quantos", "quanta", "quantas", "ha", "houve", "ser", "sao", "era",
    "foi", "estao", "esta", "estou", "estava", "estive", "sendo", "sao", "aqui",
    "ali", "la", "ja", "ainda", "sempre", "nunca", "tambem", "sim", "pode", "podem",
    "ter", "tem", "tinha", "teve", "sua", "tudo", "nada", "algo", "algum", "alguma",
    "cada", "outro", "outra", "outros", "outras", "entre", "atraves", "durante",
    "antes", "depois", "apos", "ate", "nem", "tambem", "quando", "porque", "porem",
    "todavia", "contudo", "entretanto", "seja", "sejam", "fazer", "fez", "feita",
    "faz", "fazem", "partir", "desse", "dessa", "nesses", "nessa", "daquilo",
    "portanto", "assim", "via", "voce", "voces", "min", "nosso", "nossa", "me",
    "te", "se", "lhe", "lhes", "nos", "vos", "deste", "desta", "nestes", "nesta",
    "mesmo", "mesma", "mesmos", "mesmas", "so", "quase", "tal", "tais", "vez",
    "vezes", "dia", "anos", "coisa", "coisas", "ser", "tipo", "tipos", "forma",
    "formas", "parte", "partes", "caso", "casos", "modo", "maneira", "dentro",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Normalizer:
    """Accent-fold, tokenize, stopword-filter and stem a text.

    Args:
        lang: language id (``"pt"`` built in). Others require passing
            ``stemmer``/``stopwords`` explicitly.
        remove_stopwords: drop the language stopword list.
        stem: apply the language stemmer.
        stemmer: optional callable token -> stem (defaults to the vendored
            Portuguese Snowball stemmer for ``lang="pt"``).
        stopwords: optional set of stopword tokens (defaults to the built-in
            Portuguese list for ``lang="pt"``).
    """

    def __init__(self, lang="pt", remove_stopwords=True, stem=True,
                 stemmer=None, stopwords=None):
        self.lang = lang
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        if stemmer is None:
            stemmer = PortugueseStemmer().stem if lang == "pt" else None
        self._stemmer = stemmer
        if stopwords is None:
            stopwords = _STOPWORDS_PT if lang == "pt" else frozenset()
        self._stop = frozenset(stopwords)

    @staticmethod
    def fold(text):
        """NFKD-normalize and strip all diacritics (``"café" -> "cafe"``)."""
        if not isinstance(text, str):
            text = str(text)
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    def tokenize_raw(self, text):
        """Fold + lowercase + regex tokenize. Returns raw tokens."""
        return _TOKEN_RE.findall(self.fold(text).lower())

    def stem_tokens(self, tokens):
        if not self.stem or self._stemmer is None:
            return list(tokens)
        return [self._stemmer(tok) for tok in tokens]

    def __call__(self, text):
        toks = self.tokenize_raw(text)
        toks = self.stem_tokens(toks)
        if self.remove_stopwords:
            toks = [t for t in toks if t not in self._stop]
        return toks
