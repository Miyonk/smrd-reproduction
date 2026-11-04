r"""
Text Preprocessing Module for Rumor Detection.

Specification: Section 2 of PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md.
Implements the canonical 7-step pipeline:
  Step 1: URL Removal (regex: https?://\S+|www\.\S+) [Ref 23]
  Step 2: String Slicing (text[:max_chars], default 512 chars) [Section 3.2]
  Step 3: Lexical Tokenization (TweetTokenizer, preserving case, handles, hashtags, punctuation)
  Step 4: Stopword Removal (NLTK English stopwords, case-insensitive match) [Ref 24]
  Step 5: Light-10 Suffix Rule (strip terminal 'and' if remaining len >= 3, token len >= 6)
  Step 6: Dictionary Lemmatization (WordNet two-pass: verb first, fallback noun)
  Step 7: Token Sequence Reconstruction (re-join with single space)
"""

from __future__ import annotations

import html
import logging
import re
from typing import List, Optional, Set

LOGGER = logging.getLogger("smrd.preprocess")

URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
MENTION_REGEX = re.compile(r"@\w+")
MULTIPLE_SPACES = re.compile(r"\s+")

# Fallback English stopwords if NLTK is not available
DEFAULT_STOPWORDS: Set[str] = {
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


class PreprocessingError(ValueError):
    """Raised when text preprocessing fails due to invalid input type or format."""
    pass


class TextPreprocessor:
    """
    Standard preprocessor for the SMRD paper reproduction.
    Follows Section 2 of the Phase 2B.4 implementation specification.
    """

    def __init__(
        self,
        max_chars: int = 512,
        remove_stopwords: bool = True,
        light10: bool = True,
        lemmatize: bool = True,
        allow_nltk_download: bool = True,
    ) -> None:
        self.max_chars = max_chars
        self.remove_stopwords = remove_stopwords
        self.light10 = light10
        self.lemmatize = lemmatize
        self.allow_nltk_download = allow_nltk_download
        self.backend = "fallback"

        self._tokenizer = None
        self._lemmatizer = None
        self._stopwords: Set[str] = set(DEFAULT_STOPWORDS)

        self._init_nltk()

    def _init_nltk(self) -> None:
        try:
            import nltk
            if self.allow_nltk_download:
                for corpus in ("stopwords", "wordnet", "omw-1.4", "punkt"):
                    try:
                        nltk.download(corpus, quiet=True)
                    except Exception as exc:
                        LOGGER.debug("NLTK download failed for %s: %s", corpus, exc)

            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            from nltk.tokenize import TweetTokenizer

            self._tokenizer = TweetTokenizer(
                preserve_case=True, reduce_len=False, strip_handles=False
            )
            self._lemmatizer = WordNetLemmatizer()
            self._stopwords = set(stopwords.words("english"))
            self.backend = "nltk"
        except Exception as exc:
            LOGGER.warning("NLTK not fully available (%s); using fallback preprocessor backend", exc)
            self.backend = "fallback"

    @staticmethod
    def light10_token(token: str) -> str:
        """
        Light-10 operational suffix rule:
        If token ends with 'and' (case-insensitive) and remaining length >= 3:
            strip the terminal 'and'.
        Matches Section 2.2 step 5.
        """
        lower = token.lower()
        if lower.endswith("and") and (len(token) - 3) >= 3:
            return token[:-3]
        return token

    def _lemmatize_token(self, token: str) -> str:
        """
        Two-pass WordNet lemmatizer:
        1. Prioritize verb: lemmatize(token.lower(), pos='v')
        2. If unchanged, try noun: lemmatize(token.lower(), pos='n')
        If modified, returns the lowercase lemma; if unchanged, preserves original case.
        """
        if self._lemmatizer is None:
            return token
        try:
            lower = token.lower()
            lemma_v = self._lemmatizer.lemmatize(lower, pos="v")
            if lemma_v != lower:
                return lemma_v
            lemma_n = self._lemmatizer.lemmatize(lower, pos="n")
            if lemma_n != lower:
                return lemma_n
            return token
        except Exception:
            return token

    def __call__(self, text: str) -> str:
        if not isinstance(text, str):
            raise PreprocessingError(f"Input text must be a string, got {type(text).__name__}")

        # Step 1: URL removal
        cleaned = URL_REGEX.sub("", text)

        # Step 2: 512-character slicing
        if self.max_chars is not None and self.max_chars > 0:
            cleaned = cleaned[: self.max_chars]

        # Step 3: Lexical tokenization
        if self._tokenizer is not None:
            tokens = self._tokenizer.tokenize(cleaned)
        else:
            # Fallback regex tokenization preserving handles, hashtags, words, punctuation
            tokens = re.findall(r"@\w+|#\w+|\w+(?:'\w+)?|[^\s\w]", cleaned)

        processed_tokens: List[str] = []
        for tok in tokens:
            # Step 4: Stopword removal (case-insensitive)
            if self.remove_stopwords and tok.lower() in self._stopwords:
                continue

            # Step 5: Light-10 suffix rule
            if self.light10:
                tok = self.light10_token(tok)

            # Step 6: Dictionary lemmatization
            if self.lemmatize:
                tok = self._lemmatize_token(tok)

            processed_tokens.append(tok)

        # Step 7: Token sequence reconstruction
        return " ".join(processed_tokens)


# ---------------------------------------------------------------------------
# Phase 1 Backward Compatibility Helpers
# ---------------------------------------------------------------------------
def clean_tweet_for_transformer(text: str, remove_urls: bool = True) -> str:
    """Phase 1 helper for basic transformer cleaning."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    if remove_urls:
        text = URL_REGEX.sub("", text)
    else:
        text = URL_REGEX.sub("[URL]", text)
    text = MENTION_REGEX.sub("@user", text)
    return MULTIPLE_SPACES.sub(" ", text).strip()


def clean_tweet_classical(text: str, remove_stopwords: bool = True) -> str:
    """Phase 1 helper for classical tweet cleaning."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_REGEX.sub("", text)
    text = MENTION_REGEX.sub("", text)
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS]
    return " ".join(tokens)
