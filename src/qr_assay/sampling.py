from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fileio import open_text
from .sources import iter_urls, source_inventory, stable_digest128
from .synthetic import grammar_matched


@dataclass(frozen=True)
class SourcePayload:
    source: str
    payload: str
    natural_host_sha256: str
    onion_version: int | None
    natural_source_sha256: str = ""


@dataclass
class Reservoir:
    capacity: int
    rng: random.Random

    def __post_init__(self) -> None:
        self.items: list[SourcePayload] = []
        self.seen = 0

    def add(self, item: SourcePayload) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        position = self.rng.randrange(self.seen)
        if position < self.capacity:
            self.items[position] = item


def _parse_url(payload: str) -> urllib.parse.SplitResult | None:
    try:
        parsed = urllib.parse.urlsplit(payload)
    except ValueError:
        return None
    return parsed if parsed.hostname else None


def _natural_host_sha256(payload: str) -> str:
    parsed = _parse_url(payload)
    host = (parsed.hostname or "").lower() if parsed else ""
    return hashlib.sha256(host.encode("utf-8")).hexdigest()


def _onion_version(payload: str) -> int | None:
    parsed = _parse_url(payload)
    host = (parsed.hostname or "").lower() if parsed else ""
    if not host.endswith(".onion"):
        return None
    label = host.rsplit(".onion", 1)[0].split(".")[-1]
    if len(label) == 56:
        return 3
    if len(label) == 16:
        return 2
    return None


def _canonicalize_payload(payload: str, *, granularity: str, scheme_policy: str) -> str | None:
    """Create the actual QR payload after source parsing.

    `origin` means host only: ports, path and query are removed. `url` keeps
    host/port + path/query. `path_query` removes host and scheme entirely so the
    suffix grammar can be compared without a DNS-vs-onion-host contrast.
    """
    parsed = _parse_url(payload)
    if parsed is None:
        return None
    host = (parsed.hostname or "").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    query = parsed.query

    if granularity == "path_query":
        return path + (f"?{query}" if query else "")
    if granularity == "origin":
        netloc = host
        path = "/"
        query = ""
    if scheme_policy == "strip":
        return netloc + path + (f"?{query}" if query else "")
    scheme = parsed.scheme.lower() if scheme_policy == "preserve" else "https"
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def _open_seen_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")
    return db


def _dedup_key(payload: str, host_sha256: str, granularity: str) -> str:
    # In the host-neutral grammar arm, the same suffix on two independent hosts
    # is two observations in different clusters. Collapse only duplicates within
    # the same natural host. Other granularities deduplicate the effective payload.
    if granularity == "path_query":
        return f"{host_sha256}:{payload}"
    return payload


def _source_content_ids(inventory: list[dict[str, Any]]) -> tuple[dict[str, str], str]:
    result: dict[str, str] = {}
    used_fallback = False
    for record in inventory:
        path = str(Path(str(record["path"])).resolve())
        digest = record.get("sha256")
        if not digest:
            used_fallback = True
            digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
        result[path] = str(digest)
    basis = "file_sha256_or_path_fallback" if used_fallback else "file_sha256"
    return result, basis


def _collect(
    config: dict[str, Any],
    kind: str,
    *,
    allowed_lengths: set[int] | None = None,
) -> tuple[dict[int, Reservoir], dict[str, Any]]:
    source_config = config["sources"][kind]
    paths = source_config.get("paths", [])
    if not paths:
        raise ValueError(f"No {kind} source paths configured")
    seed = int(config["seed"]) + (0 if kind == "surface" else 1)
    inventory = source_inventory(paths, hash_inputs=bool(source_config.get("hash_inputs", True)))
    source_content_ids, source_cluster_id_basis = _source_content_ids(inventory)
    capacity = int(config["sampling"]["reservoir_per_length"])
    min_bytes = int(config["sampling"]["min_bytes"])
    max_bytes = int(config["sampling"]["max_bytes"])
    granularity = str(source_config.get("granularity", "url"))
    scheme_policy = str(config["sampling"].get("scheme_policy", "strip"))
    allowed_onion_versions = {
        int(value) for value in config["sources"].get("onion", {}).get("versions", [3])
    }
    buckets: dict[int, Reservoir] = {}
    total = 0
    eligible_before_dedup = 0
    accepted = 0
    support_filtered_out = 0
    onion_version_counts: Counter[int] = Counter()
    rejected_onion_versions: Counter[int] = Counter()

    # Deduplicate only payloads that can enter the declared estimand. In
    # particular, a billion-row surface-domain source must not fill a seen-set
    # with lengths that have zero support in the already collected onion corpus.
    must_dedup = granularity in {"origin", "path_query"} or bool(
        source_config.get("deduplicate", True)
    )
    default_backend = "memory" if granularity == "origin" else "sqlite"
    dedup_backend = str(source_config.get("dedup_backend", default_backend))
    if dedup_backend not in {"memory", "sqlite"}:
        raise ValueError(f"sources.{kind}.dedup_backend must be memory or sqlite")
    seen_memory: set[bytes] | None = set() if must_dedup and dedup_backend == "memory" else None
    seen_db: sqlite3.Connection | None = None
    pending = 0
    if must_dedup and dedup_backend == "sqlite":
        output_dir = Path(config.get("outputs", {}).get("directory", ".qr-assay-tmp"))
        seen_db = _open_seen_db(output_dir / ".dedup" / f"{kind}-canonical.sqlite")

    try:
        for source, raw_payload in iter_urls(
            kind,
            paths,
            scan_limit=source_config.get("scan_limit"),
            deduplicate=False,
        ):
            total += 1
            version = _onion_version(raw_payload) if kind == "onion" else None
            if kind == "onion":
                if version not in allowed_onion_versions:
                    if version is not None:
                        rejected_onion_versions[version] += 1
                    continue
                onion_version_counts[int(version)] += 1

            payload = _canonicalize_payload(
                raw_payload, granularity=granularity, scheme_policy=scheme_policy
            )
            if not payload:
                continue
            length = len(payload.encode("utf-8"))
            if length < min_bytes or length > max_bytes:
                support_filtered_out += 1
                continue
            if allowed_lengths is not None and length not in allowed_lengths:
                support_filtered_out += 1
                continue

            eligible_before_dedup += 1
            host_sha256 = _natural_host_sha256(raw_payload)
            if must_dedup:
                digest = stable_digest128(_dedup_key(payload, host_sha256, granularity))
                if seen_memory is not None:
                    if digest in seen_memory:
                        continue
                    seen_memory.add(digest)
                elif seen_db is not None:
                    cursor = seen_db.execute(
                        "INSERT OR IGNORE INTO seen(digest) VALUES (?)", (digest,)
                    )
                    if cursor.rowcount == 0:
                        continue
                    pending += 1
                    if pending >= 10000:
                        seen_db.commit()
                        pending = 0

            source_key = str(Path(source).resolve())
            source_sha256 = source_content_ids.get(source_key)
            if source_sha256 is None:
                # This should not happen for an inventoried file, but retain an
                # explicit deterministic fallback rather than emitting a null key.
                source_sha256 = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
                source_cluster_id_basis = "file_sha256_or_path_fallback"

            accepted += 1
            if length not in buckets:
                buckets[length] = Reservoir(capacity, random.Random(seed ^ (length * 0x9E3779B1)))
            buckets[length].add(
                SourcePayload(
                    source=source,
                    payload=payload,
                    natural_host_sha256=host_sha256,
                    onion_version=version,
                    natural_source_sha256=source_sha256,
                )
            )
    finally:
        if seen_db is not None:
            seen_db.commit()
            seen_db.close()

    stats: dict[str, Any] = {
        "read": total,
        "eligible_before_dedup": eligible_before_dedup,
        "accepted": accepted,
        "support_filtered_out": support_filtered_out,
        "support_filter_lengths": sorted(allowed_lengths) if allowed_lengths is not None else None,
        "length_buckets": len(buckets),
        "granularity": granularity,
        "scheme_policy": scheme_policy,
        "deduplicated_effective_payloads": must_dedup,
        "dedup_unit": "natural_host+payload" if granularity == "path_query" else "payload",
        "dedup_backend": dedup_backend if must_dedup else "none",
        "source_cluster_id_basis": source_cluster_id_basis,
        "files": inventory,
    }
    if kind == "onion":
        stats.update(
            {
                "allowed_versions": sorted(allowed_onion_versions),
                "accepted_by_version_before_length_filter": dict(
                    sorted(onion_version_counts.items())
                ),
                "rejected_by_version": dict(sorted(rejected_onion_versions.items())),
            }
        )
    return buckets, stats


def _equal_length_quotas(available: dict[int, int], target: int, seed: int) -> dict[int, int]:
    """Legacy sensitivity: approximately equal representation of byte lengths."""
    rng = random.Random(seed)
    lengths = list(available)
    rng.shuffle(lengths)
    quotas = {length: 0 for length in lengths}
    remaining = min(target, sum(available.values()))
    while remaining:
        progressed = False
        for length in lengths:
            if quotas[length] >= available[length]:
                continue
            quotas[length] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def _overlap_weighted_quotas(
    surface: dict[int, Reservoir],
    onion: dict[int, Reservoir],
    available: dict[int, int],
    target: int,
    seed: int,
) -> dict[int, int]:
    """Allocate matches proportional to min(empirical counts) per byte length.

    Reservoir `seen` counts represent the post-filter, post-dedup empirical
    support before per-length reservoir truncation. Allocation is capped by the
    actual retained reservoir capacity; any clipped mass is redistributed over
    lengths with remaining capacity.
    """
    rng = random.Random(seed)
    weights = {
        length: min(surface[length].seen, onion[length].seen)
        for length in available
        if available[length] > 0
    }
    quotas = {length: 0 for length in available}
    remaining = min(target, sum(available.values()))
    active = {length for length, capacity in available.items() if capacity > 0}

    while remaining > 0 and active:
        total_weight = sum(weights[length] for length in active)
        if total_weight <= 0:
            break
        raw = {length: remaining * weights[length] / total_weight for length in active}
        allocated = 0
        for length in sorted(active):
            room = available[length] - quotas[length]
            take = min(room, math.floor(raw[length]))
            if take > 0:
                quotas[length] += take
                remaining -= take
                allocated += take
        active = {length for length in active if quotas[length] < available[length]}
        if remaining == 0 or not active:
            break
        if allocated == 0:
            # Hamilton-style largest-remainder allocation. Seeded jitter only
            # breaks exact fractional ties, preserving deterministic runs.
            candidates = sorted(
                active,
                key=lambda length: (raw.get(length, 0.0) % 1.0, rng.random()),
                reverse=True,
            )
            for length in candidates:
                if remaining == 0:
                    break
                quotas[length] += 1
                remaining -= 1
            active = {length for length in active if quotas[length] < available[length]}

    return quotas


def _match_payloads(
    surface: dict[int, Reservoir],
    onion: dict[int, Reservoir],
    target: int,
    seed: int,
    length_weighting: str,
) -> tuple[list[tuple[int, SourcePayload, SourcePayload]], dict[str, Any]]:
    rng = random.Random(seed)
    common = sorted(set(surface) & set(onion))
    available = {
        length: min(len(surface[length].items), len(onion[length].items)) for length in common
    }
    available = {length: count for length, count in available.items() if count > 0}
    overlap = {length: min(surface[length].seen, onion[length].seen) for length in available}
    if length_weighting == "equal_length":
        quotas = _equal_length_quotas(available, target, seed)
    else:
        quotas = _overlap_weighted_quotas(surface, onion, available, target, seed)

    result: list[tuple[int, SourcePayload, SourcePayload]] = []
    for length in sorted(quotas):
        count = quotas[length]
        if count <= 0:
            continue
        rng.shuffle(surface[length].items)
        rng.shuffle(onion[length].items)
        for index in range(count):
            result.append((length, surface[length].items[index], onion[length].items[index]))
    rng.shuffle(result)
    return result, {
        "length_weighting": length_weighting,
        "empirical_overlap_total": sum(overlap.values()),
        "reservoir_pair_capacity": sum(available.values()),
        "overlap_by_length": {str(length): overlap[length] for length in sorted(overlap)},
        "available_by_length": {str(length): available[length] for length in sorted(available)},
        "selected_by_length": {
            str(length): quotas[length] for length in sorted(quotas) if quotas[length] > 0
        },
    }


def prepare_payloads(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / config["outputs"]["payloads_file"]

    # Onion is normally the much smaller corpus and, for v3 origins, fixes the
    # support to a single byte length. Collect it first, then skip surface lengths
    # that provably cannot enter an exact-length match. This is an exact pruning,
    # not a sampling approximation.
    onion, onion_stats = _collect(config, "onion")
    surface, surface_stats = _collect(config, "surface", allowed_lengths=set(onion))

    target = int(config["sampling"]["target_pairs"])
    length_weighting = str(config["sampling"].get("length_weighting", "overlap"))
    matches, matching_stats = _match_payloads(
        surface, onion, target, int(config["seed"]), length_weighting
    )
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
            surface_payload = surface_item.payload
            onion_payload = onion_item.payload
            rows = [
                (
                    "surface_natural",
                    "surface",
                    False,
                    surface_item.source,
                    surface_item,
                    surface_payload,
                ),
                (
                    "onion_natural",
                    "onion",
                    False,
                    onion_item.source,
                    onion_item,
                    onion_payload,
                ),
                (
                    "surface_synthetic",
                    "surface",
                    True,
                    f"generated:surface_natural:{synthetic_mode}",
                    surface_item,
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
                    onion_item,
                    grammar_matched(
                        onion_payload,
                        seed=int(config["seed"]),
                        match_id=match_id,
                        corpus="onion",
                        mode=synthetic_mode,
                    ),
                ),
            ]
            for payload_class, grammar, synthetic, source, natural_item, payload in rows:
                record = {
                    "match_id": match_id,
                    "payload_class": payload_class,
                    "grammar": grammar,
                    "synthetic": synthetic,
                    "synthetic_mode": synthetic_mode if synthetic else None,
                    "source": source,
                    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    # Cluster IDs always point to the observed parent, including
                    # synthetic controls. natural_source_sha256 is the content
                    # digest of the source file when hashing is enabled.
                    "natural_source_sha256": natural_item.natural_source_sha256,
                    "natural_host_sha256": natural_item.natural_host_sha256,
                    # Backward-compatible alias; semantically this is now the
                    # natural-parent host cluster, not necessarily a payload host.
                    "host_sha256": natural_item.natural_host_sha256,
                    "onion_version": natural_item.onion_version,
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
        "matching": matching_stats,
        "surface": surface_stats,
        "onion": onion_stats,
        "common_byte_lengths": sorted(set(surface) & set(onion)),
    }
