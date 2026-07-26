import pytest
from modules.text_matcher.text_matcher import TextMatcher


@pytest.fixture
def matcher():
    return TextMatcher()


def test_high_similarity(matcher):
    t1 = "Sol kulağı çentikli siyah tekir kedi."
    t2 = "Siyah renkli tekir, sol kulağında çentik var."
    skor = matcher.calculate_similarity(t1, t2)
    assert skor > 0.75


def test_low_similarity(matcher):
    t1 = "Siyah tekir kedi mavi tasmalı."
    t2 = "Sarı golden retriever köpek tasmasız."
    skor = matcher.calculate_similarity(t1, t2)
    assert skor < 0.40


def test_empty_input(matcher):
    with pytest.raises(ValueError):
        matcher.calculate_similarity("", "Siyah kedi")