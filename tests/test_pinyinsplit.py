import unicodedata
from itertools import product
from types import SimpleNamespace

import pytest

from py_pinyin_split import PinyinTokenizer


def test_benchmark_simple(benchmark):
    tokenizer = PinyinTokenizer()
    result = benchmark(tokenizer.tokenize, "xīnniánkuàilè")
    assert result == ["xīn", "nián", "kuài", "lè"]


def test_no_tone_splits():
    """We can split pinyin without tones"""
    tokenizer = PinyinTokenizer()
    assert tokenizer.tokenize("nihao") == ["ni", "hao"]
    assert tokenizer.tokenize("zhongguo") == ["zhong", "guo"]
    assert tokenizer.tokenize("beijing") == ["bei", "jing"]


def test_tone_splits():
    """Test handling of tone marks"""
    tokenizer = PinyinTokenizer()
    assert tokenizer.tokenize("nǐhǎo") == ["nǐ", "hǎo"]
    assert tokenizer.tokenize("Běijīng") == ["Běi", "jīng"]
    assert tokenizer.tokenize("WǑMEN") == ["WǑ", "MEN"]
    assert tokenizer.tokenize("lǜsè") == ["lǜ", "sè"]
    assert tokenizer.tokenize("lvse") == ["lv", "se"]
    assert tokenizer.tokenize("xīnniánkuàilè") == ["xīn", "nián", "kuài", "lè"]
    assert tokenizer.tokenize("màn") == ["màn"]


def test_difficult_tone_splits():
    tokenizer = PinyinTokenizer()

    # Frequency matters
    assert tokenizer.tokenize("kěnéng") == ["kě", "néng"]
    assert tokenizer.tokenize("dàngāo") == ["dàn", "gāo"]
    assert tokenizer.tokenize("bàngōngshì") == ["bàn", "gōng", "shì"]

    # Vowel matters
    assert tokenizer.tokenize("rènào") == ["rè", "nào"]
    assert tokenizer.tokenize("shēngāo") == ["shēn", "gāo"]
    assert tokenizer.tokenize("qīněr") == ["qīn", "ěr"]

    # Test that tones help resolve ambiguity
    assert tokenizer.tokenize("xīan") == ["xī", "an"]
    assert tokenizer.tokenize("xián") == ["xián"]

    # Test apostrophe handling
    assert tokenizer.tokenize("Xī'ān") == ["Xī", "'", "ān"]
    assert tokenizer.tokenize("yī'er") == ["yī", "'", "er"]
    assert tokenizer.tokenize("wǎn'ān") == ["wǎn", "'", "ān"]


def test_invalid_pinyin():
    """Inputs with invalid pinyin throw ValueErrors"""
    tokenizer = PinyinTokenizer()

    # Single consonant should raise ValueError
    with pytest.raises(ValueError):
        tokenizer.tokenize("x")

    # Invalid pinyin should raise ValueError
    with pytest.raises(ValueError):
        tokenizer.tokenize("english")

    # Unsupported pinyin should raise ValueError
    with pytest.raises(ValueError):
        tokenizer.tokenize("ni3hao3")


def test_text_with_whitespace():
    """Test handling of longer text with whitespace"""
    tokenizer = PinyinTokenizer()

    # Simple whitespace
    assert tokenizer.tokenize("nǐ hǎo") == ["nǐ", "hǎo"]

    # Multiple words with mixed tones
    assert tokenizer.tokenize("Wǒ hěn xǐhuān Zhōngguó") == [
        "Wǒ",
        "hěn",
        "xǐ",
        "huān",
        "Zhōng",
        "guó",
    ]

    # Leading/trailing whitespace
    assert tokenizer.tokenize("  nǐ hǎo  ") == ["nǐ", "hǎo"]

    # Multiple whitespace characters
    assert tokenizer.tokenize("nǐ  hǎo\t\nma") == ["nǐ", "hǎo", "ma"]


def test_nonstandard_syllables():
    """Test handling of non-standard syllables"""
    standard_tokenizer = PinyinTokenizer(include_nonstandard=False)
    nonstandard_tokenizer = PinyinTokenizer(include_nonstandard=True)

    # Should fail with standard tokenizer
    with pytest.raises(ValueError):
        standard_tokenizer.tokenize("zhèige")

    # Should work with non-standard tokenizer
    assert nonstandard_tokenizer.tokenize("zhèige") == ["zhèi", "ge"]


def test_span_tokenize():
    """Test span_tokenize method returns correct character indices"""
    tokenizer = PinyinTokenizer()

    # Basic case
    assert list(tokenizer.span_tokenize("nihao")) == [(0, 2), (2, 5)]

    # With tone marks
    assert list(tokenizer.span_tokenize("nǐhǎo")) == [(0, 2), (2, 5)]

    # Mixed case
    assert list(tokenizer.span_tokenize("NiHao")) == [(0, 2), (2, 5)]

    # Multi-character syllables
    assert list(tokenizer.span_tokenize("zhōngguó")) == [(0, 5), (5, 8)]

    # Test preservation of original text slices
    text = "Nǐhǎo"
    spans = list(tokenizer.span_tokenize(text))
    assert [text[start:end] for start, end in spans] == ["Nǐ", "hǎo"]


def test_erhua():
    """Test handling of erhua"""
    tokenizer = PinyinTokenizer()

    # Test standalone er syllable
    assert tokenizer.tokenize("er") == ["er"]
    assert tokenizer.tokenize("ér") == ["ér"]

    # Test common erhua words
    assert tokenizer.tokenize("erzi") == ["er", "zi"]
    assert tokenizer.tokenize("yidiǎnr") == ["yi", "diǎn", "r"]
    assert tokenizer.tokenize("wánr") == ["wán", "r"]


@pytest.mark.parametrize("suffix", ["", "x"])
def test_long_ambiguous_input_has_bounded_search(monkeypatch, suffix):
    tokenizer = PinyinTokenizer()
    text = "xian" * 25 + suffix
    original_prefixes = tokenizer.trie.prefixes
    prefix_calls = 0

    def bounded_prefixes(remaining):
        nonlocal prefix_calls
        prefix_calls += 1
        assert prefix_calls <= len(text), "Repeatedly searched the same positions"
        return original_prefixes(remaining)

    monkeypatch.setattr(tokenizer, "trie", SimpleNamespace(prefixes=bounded_prefixes))
    if suffix:
        with pytest.raises(ValueError):
            tokenizer.tokenize(text)
    else:
        assert tokenizer.tokenize(text) == ["xian"] * 25


@pytest.mark.parametrize("prefix", ["", "  ", "nǐ ", "nǐ, ", "WǑMEN\t\n"])
def test_ambiguity_is_independent_of_preceding_text(prefix):
    tokenizer = PinyinTokenizer()
    text = prefix + "rènào"

    assert tokenizer.tokenize(text) == tokenizer.tokenize(prefix) + ["rè", "nào"]
    assert list(tokenizer.span_tokenize(text))[-2:] == [
        (len(prefix), len(prefix) + 2),
        (len(prefix) + 2, len(text)),
    ]


@pytest.mark.parametrize("prefix", ["", "nǐ, "])
@pytest.mark.parametrize(
    "text, expected", [("guao", ["gu", "ao"]), ("chuao", ["chu", "ao"])]
)
def test_frequency_tiebreaker_scores_each_candidate(prefix, text, expected):
    tokenizer = PinyinTokenizer()
    assert tokenizer.tokenize(prefix + text) == tokenizer.tokenize(prefix) + expected


@pytest.mark.parametrize(
    "syllable, variants",
    [
        ("ê", ["ê\u0304", "ế", "ê\u030c", "ề"]),
        ("nv", ["nv\u0304", "nv\u0301", "nv\u030c", "nv\u0300"]),
        ("lv", ["lv\u0304", "lv\u0301", "lv\u030c", "lv\u0300"]),
    ],
)
def test_multicodepoint_tone_variants(syllable, variants):
    tokenizer = PinyinTokenizer()
    assert tokenizer._get_tone_variants(syllable) == [syllable] + variants
    for variant in variants:
        assert tokenizer._remove_tone(variant) == syllable


@pytest.mark.parametrize(
    "syllable",
    [
        "ê\u0304",
        "ế",
        "ê\u030c",
        "ề",
        "nv\u0304",
        "nv\u0301",
        "nv\u030c",
        "nv\u0300",
        "lv\u0304",
        "lv\u0301",
        "lv\u030c",
        "lv\u0300",
    ],
)
@pytest.mark.parametrize("normalization", ["NFC", "NFD"])
def test_combining_tones_preserve_syllables(syllable, normalization):
    tokenizer = PinyinTokenizer()
    for original in [syllable, syllable.upper()]:
        text = unicodedata.normalize(normalization, original)
        assert tokenizer.tokenize(text) == [text]
        assert list(tokenizer.span_tokenize(text)) == [(0, len(text))]


def test_decomposed_text_preserves_original_spans():
    tokenizer = PinyinTokenizer()
    text = unicodedata.normalize("NFD", "  Nǐhǎo, lǜsè!")
    expected = [
        unicodedata.normalize("NFD", token)
        for token in ["Nǐ", "hǎo", ",", "lǜ", "sè", "!"]
    ]
    spans = [(2, 5), (5, 9), (9, 10), (11, 15), (15, 18), (18, 19)]

    assert tokenizer.tokenize(text) == expected
    assert list(tokenizer.span_tokenize(text)) == spans
    assert [text[start:end] for start, end in spans] == expected


@pytest.mark.parametrize(
    "text, expected, spans",
    [
        ("nǐ≠wǒ", ["nǐ", "≠", "wǒ"], [(0, 2), (2, 3), (3, 5)]),
        ("nü\u0300sè", ["nü\u0300", "sè"], [(0, 3), (3, 5)]),
        ("nǐ!? hǎo", ["nǐ", "!?", "hǎo"], [(0, 2), (2, 4), (5, 8)]),
        ("", [], []),
        (" \t\n", [], []),
    ],
)
def test_original_text_and_punctuation_are_preserved(text, expected, spans):
    tokenizer = PinyinTokenizer()
    assert tokenizer.tokenize(text) == expected
    assert list(tokenizer.span_tokenize(text)) == spans


@pytest.mark.parametrize("text", ["n\u030c", "ni\u030c\u0300", "\u0304", "n\u0304i"])
def test_malformed_tones_are_rejected(text):
    tokenizer = PinyinTokenizer()
    with pytest.raises(ValueError):
        tokenizer.tokenize(text)


def test_best_split_matches_exhaustive_ranking():
    tokenizer = PinyinTokenizer()

    def all_splits(text):
        if not text:
            return [[]]
        return [
            [prefix] + suffix
            for prefix in tokenizer.trie.prefixes(text)
            for suffix in all_splits(text[len(prefix) :])
        ]

    def rank(syllables):
        return (
            len(syllables),
            sum(syllable[0] in tokenizer.VOWEL_TONE_VARIANTS for syllable in syllables),
            -sum(
                int(tokenizer.SYLLABLE_FREQUENCIES.get(syllable, "0"))
                for syllable in syllables
            ),
        )

    for parts in product(["gu", "a", "o", "chua", "xian", "r"], repeat=3):
        text = "".join(parts)
        candidates = all_splits(text)
        result = tokenizer.tokenize(text)
        assert result in candidates
        assert rank(result) == min(rank(candidate) for candidate in candidates)
