import base64
import hashlib
import re

from qr_assay.synthetic import grammar_matched


def test_surface_null_ladder_preserves_bytes_and_delimiters():
    payload = "https://long-name.example/path/a1?q=value"
    for mode in ("token_shuffle", "class_permute", "grammar_random"):
        result = grammar_matched(payload, seed=1, match_id=2, corpus="surface", mode=mode)
        assert result.startswith("https://")
        assert len(result.encode()) == len(payload.encode())
        assert [c for c in result if not c.isalnum()] == [c for c in payload if not c.isalnum()]
        assert result != payload


def test_class_permute_preserves_ascii_unigram_multisets_by_class():
    payload = "https://abcde-zz.example/xy12?q=abba"
    result = grammar_matched(payload, seed=4, match_id=5, corpus="surface", mode="class_permute")
    # Scheme is protected; outside it the class-permute null preserves the exact
    # lowercase and digit multisets while destroying positional order.
    original_tail = payload.split("://", 1)[1]
    result_tail = result.split("://", 1)[1]
    assert sorted(c for c in original_tail if c.islower()) == sorted(
        c for c in result_tail if c.islower()
    )
    assert sorted(c for c in original_tail if c.isdigit()) == sorted(
        c for c in result_tail if c.isdigit()
    )


def test_onion_synthetic_preserves_suffix_byte_length_and_v3_checksum():
    payload = "http://" + "a" * 56 + ".onion/path99?q=abc"
    for mode in ("token_shuffle", "class_permute", "grammar_random"):
        result = grammar_matched(payload, seed=3, match_id=4, corpus="onion", mode=mode)
        assert result.startswith("http://")
        assert ".onion/" in result
        assert len(result.encode()) == len(payload.encode())
        assert result != payload
        label = result.split("://", 1)[1].split(".onion", 1)[0]
        decoded = base64.b32decode(label.upper())
        public_key, checksum, version = decoded[:32], decoded[32:34], decoded[34:]
        assert version == b"\x03"
        assert checksum == hashlib.sha3_256(b".onion checksum" + public_key + version).digest()[:2]


def test_synthetic_is_deterministic_within_mode():
    payload = "https://example.org/alpha"
    first = grammar_matched(payload, seed=7, match_id=9, corpus="surface", mode="token_shuffle")
    second = grammar_matched(payload, seed=7, match_id=9, corpus="surface", mode="token_shuffle")
    assert first == second


def test_default_null_draw_is_stable_when_match_id_changes():
    payload = "/stable/path?value=abcdef123456"
    first = grammar_matched(payload, seed=7, match_id=1, corpus="surface", mode="grammar_random")
    second = grammar_matched(
        payload, seed=7, match_id=9999, corpus="surface", mode="grammar_random"
    )
    assert first == second


def test_explicit_unit_key_can_define_a_distinct_observational_unit():
    payload = "/stable/path?value=abcdef123456"
    first = grammar_matched(
        payload,
        seed=7,
        match_id=1,
        corpus="surface",
        mode="grammar_random",
        unit_key="host-a",
    )
    second = grammar_matched(
        payload,
        seed=7,
        match_id=1,
        corpus="surface",
        mode="grammar_random",
        unit_key="host-b",
    )
    assert first != second


def test_percent_encoding_remains_syntactically_valid_under_null_ladder():
    payload = "/encoded/%2F/%aB/%CD?q=%20"
    token = grammar_matched(payload, seed=8, match_id=1, corpus="surface", mode="token_shuffle")
    assert re.findall(r"%[0-9A-Fa-f]{2}", token) == ["%2F", "%aB", "%CD", "%20"]

    randomized = grammar_matched(
        payload,
        seed=8,
        match_id=1,
        corpus="surface",
        mode="grammar_random",
    )
    escapes = re.findall(r"%[0-9A-Fa-f]{2}", randomized)
    assert len(escapes) == 4
    assert len(randomized.encode("utf-8")) == len(payload.encode("utf-8"))
    # Character-class positions inside each valid escape stay in the same class.
    for original, replacement in zip(re.findall(r"%[0-9A-Fa-f]{2}", payload), escapes, strict=True):
        for old, new in zip(original[1:], replacement[1:], strict=True):
            assert old.isdigit() == new.isdigit()
            assert old.islower() == new.islower()
            assert old.isupper() == new.isupper()


def test_degenerate_token_shuffle_stays_inside_the_declared_null_family():
    payload = "/aaaa/1111"
    result = grammar_matched(payload, seed=11, match_id=12, corpus="surface", mode="token_shuffle")
    # There is no distinct within-token permutation here. The correct null draw
    # is therefore identical, not a silent fallback to grammar_random.
    assert result == payload
