import random

from qr_assay.sampling import _canonicalize_payload, _collect
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


def test_legacy_v2_syntax_is_still_parseable_but_not_confirmatory_by_default():
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
                "versions": [3],
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
    assert next(iter(buckets.values())).items[0].payload == f"{host}/"
    assert stats["allowed_versions"] == [3]


def test_path_query_removes_host_and_scheme_for_both_corpora():
    suffix = "/docs/chapter-7?lang=en&mode=compact"
    surface = f"https://example.org{suffix}"
    onion_host = synthetic_onion_label(56, random.Random(4)) + ".onion"
    onion = f"http://{onion_host}{suffix}"
    assert _canonicalize_payload(surface, granularity="path_query", scheme_policy="preserve") == suffix
    assert _canonicalize_payload(onion, granularity="path_query", scheme_policy="https") == suffix


def test_path_query_deduplicates_within_host_but_not_across_hosts(tmp_path):
    source = tmp_path / "surface.txt"
    source.write_text(
        "https://a.example/login\n"
        "https://a.example/login\n"
        "https://b.example/login\n",
        encoding="utf-8",
    )
    config = {
        "seed": 1,
        "sources": {
            "surface": {
                "paths": [str(source)],
                "scan_limit": None,
                "deduplicate": True,
                "granularity": "path_query",
            },
            "onion": {"versions": [3]},
        },
        "sampling": {
            "reservoir_per_length": 10,
            "min_bytes": 1,
            "max_bytes": 200,
            "scheme_policy": "strip",
        },
        "outputs": {"directory": str(tmp_path / "out")},
    }
    buckets, stats = _collect(config, "surface")
    items = [item for bucket in buckets.values() for item in bucket.items]
    assert stats["accepted"] == 2
    assert stats["dedup_unit"] == "natural_host+payload"
    assert [item.payload for item in items] == ["/login", "/login"]
    assert len({item.natural_host_sha256 for item in items}) == 2


def test_v2_onions_are_excluded_by_default_but_can_be_enabled(tmp_path):
    v2_host = "b" * 16 + ".onion"
    v3_host = synthetic_onion_label(56, random.Random(5)) + ".onion"
    source = tmp_path / "onions.txt"
    source.write_text(f"http://{v2_host}/path\nhttp://{v3_host}/path\n", encoding="utf-8")

    base = {
        "seed": 1,
        "sources": {
            "onion": {
                "paths": [str(source)],
                "scan_limit": None,
                "deduplicate": True,
                "granularity": "url",
            }
        },
        "sampling": {
            "reservoir_per_length": 10,
            "min_bytes": 1,
            "max_bytes": 200,
            "scheme_policy": "strip",
        },
        "outputs": {"directory": str(tmp_path / "out-default")},
    }
    default_buckets, default_stats = _collect(base, "onion")
    default_items = [item for bucket in default_buckets.values() for item in bucket.items]
    assert len(default_items) == 1
    assert default_items[0].onion_version == 3
    assert default_stats["rejected_by_version"] == {2: 1}

    both = {
        **base,
        "sources": {"onion": {**base["sources"]["onion"], "versions": [2, 3]}},
        "outputs": {"directory": str(tmp_path / "out-both")},
    }
    both_buckets, _ = _collect(both, "onion")
    both_items = [item for bucket in both_buckets.values() for item in bucket.items]
    assert {item.onion_version for item in both_items} == {2, 3}
