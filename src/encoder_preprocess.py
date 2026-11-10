"""
Minimal tweet normalisation for the fine-tuned encoder path (stage 1+).

Pretrained encoders expect natural text, so this deliberately does *not*
stem, lemmatise or remove stopwords (the legacy ``TextPreprocessor`` in
``src/preprocess.py`` does, and stays available for the GPT-2/DHOA ablation).

Only three things change:

* HTML entities are unescaped (``&amp;`` -> ``&``),
* URLs become the literal token ``URL`` (Twitter15/16 already ship this way,
  so PHEME is brought in line with them),
* @mentions become ``@USER``.

This is a pure function with no model state, so a serving layer can import
it without importing any training code.
"""

from __future__ import annotations

import html
import re

NORMALIZER_NAME = "normalize_tweet"
NORMALIZER_VERSION = "1"

URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
MENTION_REGEX = re.compile(r"@\w+")
WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_tweet(text: object) -> str:
    """Return ``text`` with URLs/mentions normalised and whitespace collapsed."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_REGEX.sub("URL", text)
    text = MENTION_REGEX.sub("@USER", text)
    return WHITESPACE_REGEX.sub(" ", text).strip()


def normalizer_config() -> dict:
    return {"name": NORMALIZER_NAME, "version": NORMALIZER_VERSION}
