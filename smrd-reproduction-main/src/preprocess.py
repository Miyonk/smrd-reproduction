"""
Text preprocessing for the SMRD reproduction.

This module contains **two generations** of preprocessing:

1. The original Phase 1 helpers (``clean_tweet_for_transformer`` and
   ``clean_tweet_classical``), retained because
   ``src/data_loader.py`` and ``tests/verify_phase1.py`` depend on them.
2. The Phase 3 specification implementation (``TextPreprocessor``), which
   follows Section 2 of ``PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md``
   exactly and must be used for all model features.

The Phase 3 pipeline is deliberately *not* the same as the Phase 1 helpers.
Section 2 forbids whole-text lowercasing, @mention removal, hashtag
stripping, punctuation removal, number removal, and HTML entity/tag
handling.  Those operations are only present in the legacy helpers.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Iterable, List, Optional

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy Phase 1 helpers (do not use for Phase 3 features)
# ---------------------------------------------------------------------------

URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
MENTION_REGEX = re.compile(r"@\w+")
MULTIPLE_SPACES = re.compile(r"\s+")

# Standard English stopwords (self-contained fallback used when the NLTK
# corpus is unavailable).  Kept byte-for-byte compatible with Phase 1.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


def clean_tweet_for_transformer(text: str, remove_urls: bool = True) -> str:
    """Legacy Phase 1 cleaning.  Not the Phase 3 specification pipeline."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = URL_REGEX.sub("", text) if remove_urls else URL_REGEX.sub("[URL]", text)
    text = MENTION_REGEX.sub("@user", text)
    return MULTIPLE_SPACES.sub(" ", text).strip()


def clean_tweet_classical(text: str, remove_stopwords: bool = True) -> str:
    """Legacy Phase 1 classical cleaning.  Not the Phase 3 pipeline."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = URL_REGEX.sub("", text)
    text = MENTION_REGEX.sub("", text)
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Phase 3 specification implementation
# ---------------------------------------------------------------------------


class PreprocessingError(ValueError):
    """Raised when a text value cannot be preprocessed at all."""


def _load_nltk_components(allow_download: bool = False):
    """Return (tokenizer, stopwords, lemmatizer, backend_name).

    Falls back to deterministic local components when NLTK data is missing
    so that preprocessing never silently changes meaning or crashes a run.
    """
    try:
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import TweetTokenizer

        def ensure(resource: str) -> None:
            try:
                nltk.data.find(resource)
            except LookupError:
                if allow_download:
                    nltk.download(resource.split("/")[-1], quiet=True)

        ensure("corpora/stopwords")
        ensure("corpora/wordnet")
        ensure("corpora/omw-1.4")

        tokenizer = TweetTokenizer(
            preserve_case=True, reduce_len=False, strip_handles=False
        )
        words = set(nltk_stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        # Probe both resources; they raise LookupError when unavailable.
        lemmatizer.lemmatize("sharing", pos="v")
        return tokenizer, words, lemmatizer, "nltk"
    except Exception as exc:  # pragma: no cover - depends on local NLTK data
        LOGGER.warning(
            "NLTK resources unavailable (%s). Falling back to the bundled "
            "stopword list and identity lemmatization. Install NLTK corpora "
            "for spec-exact preprocessing.",
            exc,
        )
        return _FallbackTokenizer(), set(STOPWORDS), _FallbackLemmatizer(), "fallback"


class _FallbackTokenizer:
    """Regex tokenizer that preserves case, punctuation, handles, hashtags."""

    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        return self._pattern.findall(text)


class _FallbackLemmatizer:
    """Conservative suffix lemmatizer used only when WordNet is missing."""

    _rules = (
        ("ies", "y"),
        ("sses", "ss"),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
        ("ly", ""),
        ("s", ""),
    )

    def lemmatize(self, word: str, pos: str = "n") -> str:
        if pos == "v":
            for suffix, replacement in self._rules:
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    return word[: -len(suffix)] + replacement
        if pos == "n" and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word


class TextPreprocessor:
    """Exact Section 2.1 pipeline.

    Order: URL removal -> ``text[:512]`` -> lexical tokenization ->
    case-insensitive NLTK stopword removal -> Light-10 terminal ``and``
    stripping -> two-pass WordNet lemmatization -> single-space join.
    """

    def __init__(
        self,
        max_chars: int = 512,
        remove_stopwords: bool = True,
        light10: bool = True,
        lemmatize: bool = True,
        allow_nltk_download: bool = False,
    ) -> None:
        self.max_chars = max_chars
        self.remove_stopwords = remove_stopwords
        self.light10 = light10
        self.lemmatize = lemmatize

        self._tokenizer, self._stopwords, self._lemmatizer, self.backend = (
            _load_nltk_components(allow_download=allow_nltk_download)
        )
        self._stopwords_lower = {w.lower() for w in self._stopwords}

    # -- individual operations --------------------------------------------
    @staticmethod
    def remove_urls(text: str) -> str:
        return URL_REGEX.sub("", text)

    def slice_chars(self, text: str) -> str:
        return text[: self.max_chars]

    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer.tokenize(text)

    def remove_stopword_tokens(self, tokens: Iterable[str]) -> List[str]:
        if not self.remove_stopwords:
            return list(tokens)
        return [t for t in tokens if t.lower() not in self._stopwords_lower]

    @staticmethod
    def light10_token(token: str) -> str:
        if token.lower().endswith("and") and (len(token) - 3) >= 3:
            return token[:-3]
        return token

    def lemmatize_token(self, token: str) -> str:
        if not self.lemmatize:
            return token
        lower = token.lower()
        lemma_v = self._lemmatizer.lemmatize(lower, pos="v")
        if lemma_v != lower:
            return lemma_v
        return self._lemmatizer.lemmatize(lower, pos="n")

    # -- full pipeline ------------------------------------------------------
    def __call__(self, text: str) -> str:
        if not isinstance(text, str):
            raise PreprocessingError(
                f"expected str, got {type(text).__name__}"
            )
        if not text.strip():
            raise PreprocessingError("input text is empty")

        text = self.remove_urls(text)
        text = self.slice_chars(text)

        tokens = self.tokenize(text)
        tokens = self.remove_stopword_tokens(tokens)

        if self.light10:
            tokens = [self.light10_token(t) for t in tokens]
        tokens = [self.lemmatize_token(t) for t in tokens]

        return " ".join(tokens)


def preprocess_series(values: Iterable[str], **kwargs) -> List[str]:
    """Preprocess an iterable with a single shared ``TextPreprocessor``."""
    prep = TextPreprocessor(**kwargs)
    return [prep(v) for v in values]
