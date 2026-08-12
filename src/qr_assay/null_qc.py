from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .fileio import open_text

PAIRS = {
    "surface": ("surface_natural", "surface_synthetic"),
    "onion": ("onion_natural", "onion_synthetic"),
}


def _iter_payloads(path: Path):
    with open_text(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def analyze_null_qc(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["outputs"]["directory"])
    path = output_dir / config["outputs"]["payloads_file"]
    if not path.exists():
        raise FileNotFoundError(f"Prepare payloads first: {path}")

    matches: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    non_ascii_natural = Counter()
    natural_counts = Counter()
    for row in _iter_payloads(path):
        match_id = int(row["match_id"])
        matches[match_id][row["payload_class"]] = row
        if not bool(row.get("synthetic")):
            corpus = str(row["grammar"])
            natural_counts[corpus] += 1
            payload = row.get("payload")
            if payload is not None and not payload.isascii():
                non_ascii_natural[corpus] += 1

    unchanged = Counter()
    compared = Counter()
    byte_length_failures = 0
    incomplete_matches = 0
    synthetic_hashes: dict[str, set[str]] = defaultdict(set)

    for records in matches.values():
        if set(records) != {
            "surface_natural",
            "surface_synthetic",
            "onion_natural",
            "onion_synthetic",
        }:
            incomplete_matches += 1
            continue
        for corpus, (natural_class, synthetic_class) in PAIRS.items():
            natural = records[natural_class]
            synthetic = records[synthetic_class]
            compared[corpus] += 1
            if natural["payload_sha256"] == synthetic["payload_sha256"]:
                unchanged[corpus] += 1
            if int(natural["byte_length"]) != int(synthetic["byte_length"]):
                byte_length_failures += 1
            synthetic_hashes[corpus].add(str(synthetic["payload_sha256"]))

    corpora: dict[str, Any] = {}
    for corpus in PAIRS:
        n = compared[corpus]
        natural_n = natural_counts[corpus]
        corpora[corpus] = {
            "pairs_compared": n,
            "unchanged_controls": unchanged[corpus],
            "unchanged_fraction": unchanged[corpus] / n if n else 0.0,
            "unique_synthetic_payloads": len(synthetic_hashes[corpus]),
            "unique_synthetic_fraction": len(synthetic_hashes[corpus]) / n if n else 0.0,
            "non_ascii_natural_payloads": non_ascii_natural[corpus],
            "non_ascii_natural_fraction": (
                non_ascii_natural[corpus] / natural_n if natural_n else 0.0
            ),
        }

    result = {
        "synthetic_mode": str(config["sampling"].get("synthetic_mode", "grammar_random")),
        "matches_seen": len(matches),
        "incomplete_matches": incomplete_matches,
        "byte_length_failures": byte_length_failures,
        "passed_hard_invariants": incomplete_matches == 0 and byte_length_failures == 0,
        "interpretation": (
            "Unchanged controls are retained, not repaired by switching null families. A high "
            "unchanged fraction means the declared null is weak/degenerate for that corpus and "
            "must be reported. Non-ASCII characters are currently preserved by the ASCII null "
            "operators and therefore require a separate sensitivity analysis if prevalent."
        ),
        "corpora": corpora,
    }
    output_path = output_dir / "null_qc.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["output"] = str(output_path)
    return result
