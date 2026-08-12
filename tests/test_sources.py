import random

from qr_assay.sampling import _collect
from qr_assay.sources import normalize_onion, normalize_surface, valid_v3_onion_label
from qr_assay.synthetic import synthetic_onion_label


def test_surface_domain_is_canonicalized():
    assert normalize_surface("Example.COM") == "https://example.com/"


def test_surface_rejects_onion():
    host = synthetic_onion_label(56, random.Random(1)) + ".onion"
    assert normalize_surface(host) is None


def test_onion_extractor_finds_checksum_valid_v3_url_and_path():
    label = synthetic_onion_label(56, random.Random(2))
    host = label + ".onion"
    text = f"prefix https://{host}/a/b?q=1 suffix"
    assert valid_v3_onion_label(label)
    assert normalize_onion(text) == [f"https://{host}/a/b?q=1"]


def test_onion_extractor_rejects_malformed_v3_checksum():
    label = "a" * 56
    assert not valid_v3_onion_label(label)
    assert normalize_onion(f"http://{label}.onion/") == []


def test_legacy_v2_syntax_is_still_accepted():
    host = "b" * 16 + ".onion"
    assert normalize_onion(host) == [f"http://{host}/"]


def test_origin_granularity_deduplicates_paths_and_strips_scheme(tmp_path):
    host = synthetic_onion_label(56, random.Random(3)) + ".onion"
    source = tmp_path / "onions.txt"
    source.write_text(f"http://{host}/a\nhttps://{host}/b\n", encoding="utf-8")
    config = {
        "seed": 1,
        "sources": {
            "onion": {
                "paths": [str(source)],
                "scan_limit": None,
                "deduplicate": True,
                "granularity": "origin",
            }
        },
        "sampling": {
            "reservoir_per_length": 10,
            "min_bytes": 1,
            "max_bytes": 200,
            "scheme_policy": "strip",
        },
        "outputs": {"directory": str(tmp_path / "out")},
    }
    buckets, stats = _collect(config, "onion")
    assert stats["accepted"] == 1
    assert sum(len(bucket.items) for bucket in buckets.values()) == 1
    assert next(iter(buckets.values())).items[0][1] == f"{host}/"
