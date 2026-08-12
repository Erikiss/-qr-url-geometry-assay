from __future__ import annotations

import hashlib
import json
import random
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fileio import open_text
from .sources import iter_urls, source_inventory, stable_u64
from .synthetic import grammar_matched


@dataclass
class Reservoir:
    capacity: int
    rng: random.Random

    def __post_init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self.seen = 0

    def add(self, item: tuple[str, str]) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        position = self.rng.randrange(self.seen)
        if position < self.capacity:
            self.items[position] = item


def _canonicalize_payload(payload: str, *, granularity: str, scheme_policy: str) -> str | None:
    """Create the actual QR payload after source parsing.

    ``granularity`` is applied symmetrically to surface and onion sources.
    ``scheme_policy=strip`` removes the otherwise arbitrary http/https difference
    before byte-length matching; ``https`` forces a common scheme; ``preserve``
    retains the observed scheme.
    """
    try:
        parsed = urllib.parse.urlsplit(payload)
    except ValueError:
        return None
    if not parsed.hostname:
        return None
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    query = parsed.query
    if granularity == "origin":
        path = "/"
        query = ""
    if scheme_policy == "strip":
        return netloc + path + (f"?{query}" if query else "")
    scheme = parsed.scheme.lower() if scheme_policy == "preserve" else "https"
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def _host_hash(payload: str) -> str:
    candidate = payload if "://" in payload else f"https://{payload}"
    host = urllib.parse.urlsplit(candidate).hostname or ""
    return hashlib.sha256(host.lower().encode("utf-8")).hexdigest()


def _collect(config: dict[str, Any], kind: str) -> tuple[dict[int, Reservoir], dict[str, Any]]:
    source_config = config["sources"][kind]
    paths = source_config.get("paths", [])
    if not paths:
        raise ValueError(f"No {kind} source paths configured")
    seed = int(config["seed"]) + (0 if kind == "surface" else 1)
    inventory = source_inventory(paths, hash_inputs=bool(source_config.get("hash_inputs", True)))
    capacity = int(config["sampling"]["reservoir_per_length"])
    min_bytes = int(config["sampling"]["min_bytes"])
    max_bytes = int(config["sampling"]["max_bytes"])
    granularity = str(source_config.get("granularity", "url"))
    scheme_policy = str(config["sampling"].get("scheme_policy", "strip"))
    buckets: dict[int, Reservoir] = {}
    total = 0
    accepted = 0
    seen_canonical: set[int] = set()
    for source, raw_payload in iter_urls(
        kind,
        paths,
        scan_limit=source_config.get("scan_limit"),
        deduplicate=bool(source_config.get("deduplicate", True)),
    ):
        total += 1
        payload = _canonicalize_payload(
            raw_payload, granularity=granularity, scheme_policy=scheme_policy
        )
        if not payload:
            continue
        digest = stable_u64(payload)
        if digest in seen_canonical:
            continue
        seen_canonical.add(digest)
        length = len(payload.encode("utf-8"))
        if length < min_bytes or length > max_bytes:
            continue
        accepted += 1
        if length not in buckets:
            buckets[length] = Reservoir(capacity, random.Random(seed ^ (length * 0x9E3779B1)))
        buckets[length].add((source, payload))
    return buckets, {
        "read": total,
        "accepted": accepted,
        "length_buckets": len(buckets),
        "granularity": granularity,
        "scheme_policy": scheme_policy,
        "files": inventory,
    }


def _round_robin_matches(
    surface: dict[int, Reservoir], onion: dict[int, Reservoir], target: int, seed: int
) -> list[tuple[int, tuple[str, str], tuple[str, str]]]:
    rng = random.Random(seed)
    common = sorted(set(surface) & set(onion))
    rng.shuffle(common)
    for length in common:
        rng.shuffle(surface[length].items)
        rng.shuffle(onion[length].items)
    positions = {length: 0 for length in common}
    result: list[tuple[int, tuple[str, str], tuple[str, str]]] = []
    while len(result) < target:
        progressed = False
        for length in common:
            position = positions[length]
            available = min(len(surface[length].items), len(onion[length].items))
            if position >= available:
                continue
            result.append((length, surface[length].items[position], onion[length].items[position]))
            positions[length] += 1
            progressed = True
            if len(result) >= target:
                break
        if not progressed:
            break
    return result


def prepare_payloads(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / config["outputs"]["payloads_file"]
    surface, surface_stats = _collect(config, "surface")
    onion, onion_stats = _collect(config, "onion")
    target = int(config["sampling"]["target_pairs"])
    matches = _round_robin_matches(surface, onion, target, int(config["seed"]))
    if not matches:
        surface_lengths = sorted(surface)
        onion_lengths = sorted(onion)
        raise ValueError(
            "No byte-length overlap between surface and onion sources. "
            f"Surface lengths={surface_lengths[:12]}..., onion lengths={onion_lengths[:12]}..."
        )
    store_text = bool(config["outputs"].get("store_payload_text", True))
    synthetic_mode = str(config["sampling"].get("synthetic_mode", "grammar_random"))
    class_counts: dict[str, int] = defaultdict(int)
    with open_text(payload_path, "wt") as handle:
        for match_id, (byte_length, surface_item, onion_item) in enumerate(matches):
            surface_source, surface_payload = surface_item
            onion_source, onion_payload = onion_item
            rows = [
                ("surface_natural", "surface", False, surface_source, surface_payload),
                ("onion_natural", "onion", False, onion_source, onion_payload),
                (
                    "surface_synthetic",
                    "surface",
                    True,
                    f"generated:surface_natural:{synthetic_mode}",
                    grammar_matched(
                        surface_payload,
                        seed=int(config["seed"]),
                        match_id=match_id,
                        corpus="surface",
                        mode=synthetic_mode,
                    ),
                ),
                (
                    "onion_synthetic",
                    "onion",
                    True,
                    f"generated:onion_natural:{synthetic_mode}",
                    grammar_matched(
                        onion_payload,
                        seed=int(config["seed"]),
                        match_id=match_id,
                        corpus="onion",
                        mode=synthetic_mode,
                    ),
                ),
            ]
            for payload_class, grammar, synthetic, source, payload in rows:
                record = {
                    "match_id": match_id,
                    "payload_class": payload_class,
                    "grammar": grammar,
                    "synthetic": synthetic,
                    "synthetic_mode": synthetic_mode if synthetic else None,
                    "source": source,
                    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    "host_sha256": _host_hash(payload),
                    "payload": payload if store_text else None,
                    "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    "byte_length": len(payload.encode("utf-8")),
                    "matched_byte_length": byte_length,
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                class_counts[payload_class] += 1
    return {
        "payloads_path": str(payload_path),
        "matched_pairs": len(matches),
        "requested_pairs": target,
        "complete": len(matches) == target,
        "class_counts": dict(class_counts),
        "synthetic_mode": synthetic_mode,
        "surface": surface_stats,
        "onion": onion_stats,
        "common_byte_lengths": sorted(set(surface) & set(onion)),
    }
