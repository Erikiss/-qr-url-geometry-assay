import base64
import hashlib

from qr_assay.synthetic import grammar_matched


def test_surface_synthetic_preserves_scheme_delimiters_and_bytes():
    payload = "https://long-name.example/path/a1?q=value"
    result = grammar_matched(payload, seed=1, match_id=2, corpus="surface")
    assert result.startswith("https://")
    assert len(result.encode()) == len(payload.encode())
    assert [c for c in result if not c.isalnum()] == [c for c in payload if not c.isalnum()]
    assert result != payload


def test_onion_synthetic_preserves_suffix_and_byte_length():
    payload = "http://" + "a" * 56 + ".onion/path99?q=abc"
    result = grammar_matched(payload, seed=3, match_id=4, corpus="onion")
    assert result.startswith("http://")
    assert ".onion/" in result
    assert len(result.encode()) == len(payload.encode())
    assert result != payload
    label = result.split("://", 1)[1].split(".onion", 1)[0]
    decoded = base64.b32decode(label.upper())
    public_key, checksum, version = decoded[:32], decoded[32:34], decoded[34:]
    assert version == b"\x03"
    assert checksum == hashlib.sha3_256(b".onion checksum" + public_key + version).digest()[:2]


def test_synthetic_is_deterministic():
    payload = "https://example.org/alpha"
    first = grammar_matched(payload, seed=7, match_id=9, corpus="surface")
    second = grammar_matched(payload, seed=7, match_id=9, corpus="surface")
    assert first == second
