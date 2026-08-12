import json
import random
from collections import Counter
from pathlib import Path

from qr_assay.fileio import open_text
from qr_assay.sampling import Reservoir, SourcePayload, _match_payloads, prepare_payloads
from qr_assay.synthetic import synthetic_onion_label


def _reservoir(length: int, count: int, corpus: str) -> Reservoir:
    reservoir = Reservoir(capacity=count, rng=random.Random(length))
    for index in range(count):
        payload = ("x" * length)[:-6] + f"{index:06d}"
        reservoir.add(
            SourcePayload(
                source=f"{corpus}-crawl",
                payload=payload,
                natural_host_sha256=f"{corpus}-host-{index}",
                onion_version=3 if corpus == "onion" else None,
            )
        )
    return reservoir


def test_overlap_weighting_tracks_shared_empirical_length_support():
    surface = {20: _reservoir(20, 90, "surface"), 30: _reservoir(30, 10, "surface")}
    onion = {20: _reservoir(20, 90, "onion"), 30: _reservoir(30, 10, "onion")}
    matches, stats = _match_payloads(surface, onion, 50, 7, "overlap")
    counts = Counter(length for length, _, _ in matches)
    assert counts == {20: 45, 30: 5}
    assert stats["length_weighting"] == "overlap"
    assert stats["empirical_overlap_total"] == 100


def test_equal_length_mode_is_explicit_sensitivity_not_default_estimand():
    surface = {20: _reservoir(20, 90, "surface"), 30: _reservoir(30, 10, "surface")}
    onion = {20: _reservoir(20, 90, "onion"), 30: _reservoir(30, 10, "onion")}
    matches, _ = _match_payloads(surface, onion, 20, 7, "equal_length")
    counts = Counter(length for length, _, _ in matches)
    assert counts == {20: 10, 30: 10}


def test_path_query_synthetic_rows_inherit_natural_host_and_source_clusters(tmp_path: Path):
    surface_path = tmp_path / "surface.txt"
    onion_path = tmp_path / "onion.txt"
    onion_host = synthetic_onion_label(56, random.Random(8)) + ".onion"
    surface_path.write_text(
        "https://example.org/deep/path/alpha?mode=test\n",
        encoding="utf-8",
    )
    onion_path.write_text(
        f"http://{onion_host}/deep/path/bravo?mode=test\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    config = {
        "seed": 5,
        "sources": {
            "surface": {
                "paths": [str(surface_path)],
                "deduplicate": True,
                "granularity": "path_query",
            },
            "onion": {
                "paths": [str(onion_path)],
                "deduplicate": True,
                "granularity": "path_query",
                "versions": [3],
            },
        },
        "sampling": {
            "target_pairs": 1,
            "reservoir_per_length": 10,
            "min_bytes": 1,
            "max_bytes": 200,
            "scheme_policy": "strip",
            "synthetic_mode": "grammar_random",
            "length_weighting": "overlap",
        },
        "outputs": {
            "directory": str(output),
            "payloads_file": "payloads.jsonl.gz",
            "store_payload_text": True,
        },
    }
    result = prepare_payloads(config)
    with open_text(Path(result["payloads_path"]), "rt") as handle:
        rows = [json.loads(line) for line in handle]

    by_class = {row["payload_class"]: row for row in rows}
    assert (
        by_class["surface_natural"]["natural_host_sha256"]
        == by_class["surface_synthetic"]["natural_host_sha256"]
    )
    assert (
        by_class["onion_natural"]["natural_host_sha256"]
        == by_class["onion_synthetic"]["natural_host_sha256"]
    )
    assert (
        by_class["surface_natural"]["natural_source_sha256"]
        == by_class["surface_synthetic"]["natural_source_sha256"]
    )
    assert (
        by_class["onion_natural"]["natural_source_sha256"]
        == by_class["onion_synthetic"]["natural_source_sha256"]
    )
    assert ".onion" not in by_class["onion_natural"]["payload"]
    assert "example.org" not in by_class["surface_natural"]["payload"]
