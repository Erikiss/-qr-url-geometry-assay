from qr_assay.sampling import _collect


def test_shared_length_support_is_applied_before_dedup(tmp_path):
    source = tmp_path / "surface.txt"
    source.write_text(
        "https://a.example/keep\nhttps://b.example/drop-long\nhttps://c.example/keep\n",
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
                "hash_inputs": True,
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

    buckets, stats = _collect(config, "surface", allowed_lengths={5})
    items = [item for bucket in buckets.values() for item in bucket.items]

    assert stats["read"] == 3
    assert stats["support_filter_lengths"] == [5]
    assert stats["support_filtered_out"] == 1
    assert stats["eligible_before_dedup"] == 2
    assert stats["accepted"] == 2
    assert [item.payload for item in items] == ["/keep", "/keep"]
