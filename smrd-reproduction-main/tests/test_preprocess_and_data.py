from pathlib import Path

import pytest

from src.config import Config
from src.data import build_arm_view
from src.preprocess import TextPreprocessor, PreprocessingError


@pytest.fixture(scope="module")
def prep():
    return TextPreprocessor(allow_nltk_download=True)


def test_url_removed(prep):
    cleaned = prep("Check https://example.com/a?b=1 and www.foo.org now")
    assert "http" not in cleaned
    assert "www." not in cleaned


def test_mentions_hashtags_punctuation_preserved_when_not_lemmatized():
    prep = TextPreprocessor(
        remove_stopwords=False, light10=False, lemmatize=False,
        allow_nltk_download=True,
    )
    cleaned = prep("RT @User: #Ferguson, don't STOP! 2024")
    assert "@User" in cleaned
    assert "#Ferguson" in cleaned
    assert "don't" in cleaned
    assert "," in cleaned
    assert "!" in cleaned
    assert "2024" in cleaned


def test_case_preserved_without_modifying_operations():
    prep = TextPreprocessor(
        remove_stopwords=False, light10=False, lemmatize=False,
        allow_nltk_download=True,
    )
    assert prep("Ferguson") == "Ferguson"


def test_light10_examples():
    assert TextPreprocessor.light10_token("expand") == "exp"
    assert TextPreprocessor.light10_token("demand") == "dem"
    assert TextPreprocessor.light10_token("command") == "comm"
    assert TextPreprocessor.light10_token("understand") == "underst"
    assert TextPreprocessor.light10_token("island") == "isl"
    assert TextPreprocessor.light10_token("contraband") == "contrab"
    # Fewer than 3 characters would remain: unchanged.
    assert TextPreprocessor.light10_token("and") == "and"
    assert TextPreprocessor.light10_token("stand") == "stand"


def test_stopwords_are_case_insensitive(prep):
    prep = TextPreprocessor(lemmatize=False, allow_nltk_download=True)
    assert "The" not in prep("The cat").split()
    assert "AND" not in prep("cats AND dogs").split()


def test_lemmatization_sharing_to_share(prep):
    if prep.backend != "nltk":
        pytest.skip("WordNet unavailable; fallback lemmatizer is not spec-exact")
    tokens = prep("sharing").split()
    assert tokens == ["share"]


def test_invalid_input_raises():
    prep = TextPreprocessor(allow_nltk_download=True)
    with pytest.raises(PreprocessingError):
        prep(None)  # type: ignore[arg-type]


def test_twitter16_arm_counts():
    expected = {
        "arm_b": (572, 82, 164),
        "arm_a": (572, 82, 164),
        "arm_d": (288, 41, 83),
    }
    for arm, counts in expected.items():
        view = build_arm_view(Config(task_arm=arm, dataset_name="twitter16"))
        actual = (len(view.train_indices), len(view.val_indices), len(view.test_indices))
        assert actual == counts


def test_arm_d_excludes_non_rumour_and_unverified():
    view = build_arm_view(Config(task_arm="arm_d", dataset_name="twitter16"))
    assert set(view.labels.values()) == {0, 1}
    assert len(view.labels) == 205 + 207


def test_arm_b_label_mapping():
    view = build_arm_view(Config(task_arm="arm_b", dataset_name="twitter16"))
    assert set(view.labels.values()) == {0, 1}
    assert len(view.labels) == 818


def test_splits_disjoint():
    view = build_arm_view(Config(task_arm="arm_a", dataset_name="twitter15"))
    train, val, test = set(view.train_indices), set(view.val_indices), set(view.test_indices)
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert len(train) + len(val) + len(test) == 1490
