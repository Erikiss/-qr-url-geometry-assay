from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .fileio import open_text

CORE_METRICS = (
    "data_radial_mean",
    "data_centroid_radius",
    "data_orientation_cos2",
    "data_orientation_sin2",
)

CONFIRMATORY_CONTRASTS = {
    "surface_natural-minus-surface_synthetic": (
        "surface_natural",
        "surface_synthetic",
    ),
    "onion_natural-minus-onion_synthetic": (
        "onion_natural",
        "onion_synthetic",
    ),
}


@dataclass
class ClusterTotal:
    count: int = 0
    sums: np.ndarray = field(default_factory=lambda: np.zeros(len(CORE_METRICS), dtype=np.float64))


@dataclass
class ClusterAccumulator:
    n: int = 0
    sums: np.ndarray = field(default_factory=lambda: np.zeros(len(CORE_METRICS), dtype=np.float64))
    host: dict[str, ClusterTotal] = field(default_factory=dict)
    source: dict[str, ClusterTotal] = field(default_factory=dict)

    def add(self, values: np.ndarray, *, host: str, source: str) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(CORE_METRICS),):
            raise ValueError("Cluster effect vector has an unexpected shape")
        self.n += 1
        self.sums += values
        self._add_cluster(self.host, host, values)
        self._add_cluster(self.source, source, values)

    @staticmethod
    def _add_cluster(table: dict[str, ClusterTotal], key: str, values: np.ndarray) -> None:
        if not key:
            raise ValueError("Cluster key must be present for confirmatory inference")
        total = table.get(key)
        if total is None:
            total = ClusterTotal()
            table[key] = total
        total.count += 1
        total.sums += values

    def point_mean(self) -> np.ndarray:
        return self.sums / self.n if self.n else np.zeros(len(CORE_METRICS), dtype=np.float64)

    def summarize(self, level: str) -> list[dict[str, Any]]:
        table = self.host if level == "host" else self.source
        mean = self.point_mean()
        cluster_count = len(table)
        counts = np.asarray([total.count for total in table.values()], dtype=np.float64)
        effective_clusters = (
            float(counts.sum() ** 2 / np.square(counts).sum()) if counts.size else 0.0
        )

        if self.n == 0 or cluster_count < 2:
            standard_errors = np.full(len(CORE_METRICS), np.nan)
        else:
            score_squares = np.zeros(len(CORE_METRICS), dtype=np.float64)
            for total in table.values():
                score = total.sums - total.count * mean
                score_squares += np.square(score)
            # CR1 for an intercept-only mean estimator. With one observation per
            # cluster this reduces to the ordinary standard error up to rounding.
            variance = (cluster_count / (cluster_count - 1)) * score_squares / (self.n**2)
            standard_errors = np.sqrt(np.maximum(variance, 0.0))

        rows: list[dict[str, Any]] = []
        for index, metric in enumerate(CORE_METRICS):
            se = float(standard_errors[index])
            estimate = float(mean[index])
            estimable = math.isfinite(se)
            rows.append(
                {
                    "metric": metric,
                    "n_matches": self.n,
                    "cluster_level": level,
                    "cluster_count": cluster_count,
                    "effective_cluster_count": effective_clusters,
                    "max_cluster_size": int(counts.max()) if counts.size else 0,
                    "mean_difference": estimate,
                    "cr1_standard_error": se if estimable else None,
                    "ci95_low_normal_cr1": estimate - 1.96 * se if estimable else None,
                    "ci95_high_normal_cr1": estimate + 1.96 * se if estimable else None,
                    "estimable": estimable,
                    # A hard claim is inappropriate with a handful of crawl files
                    # even if the arithmetic variance exists.
                    "cluster_count_ge_20": cluster_count >= 20,
                }
            )
        return rows


def _iter_jsonl(path: Path):
    with open_text(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _primary(row: dict[str, Any]) -> bool:
    return (
        int(row["rotation"]) == 0
        and row["reflection"] == "none"
        and int(row["scale"]) == 1
        and not bool(row["inverted"])
    )


def analyze_clustered_core(config: dict[str, Any]) -> dict[str, Any]:
    """Run scalable one-way cluster-robust inference for confirmatory contrasts.

    Host and source clustering are valid because each confirmatory contrast stays
    within one natural payload and its generated control. The direct surface-vs-
    onion contrast is intentionally omitted here; it has distinct clusters on
    both sides and needs a separately specified multi-way procedure.
    """
    output_dir = Path(config["outputs"]["directory"])
    features_path = output_dir / config["outputs"]["features_file"]
    if not features_path.exists():
        raise FileNotFoundError(f"Generate features first: {features_path}")
    expected_masks = {int(mask) for mask in config["qr"]["masks"]}
    accumulators = {contrast: ClusterAccumulator() for contrast in CONFIRMATORY_CONTRASTS}

    current_match_id: int | None = None
    records: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    invalid_matches = 0
    complete_matches = 0

    def finish_match() -> None:
        nonlocal records, invalid_matches, complete_matches
        if current_match_id is None:
            return
        needed_classes = {name for pair in CONFIRMATORY_CONTRASTS.values() for name in pair}
        if set(records) != needed_classes or any(
            set(records[cls]) != expected_masks for cls in needed_classes
        ):
            invalid_matches += 1
            records = defaultdict(dict)
            return

        complete_matches += 1
        for contrast, (natural_class, synthetic_class) in CONFIRMATORY_CONTRASTS.items():
            natural_rows = list(records[natural_class].values())
            synthetic_rows = list(records[synthetic_class].values())
            natural_host = {row.get("natural_host_sha256") for row in natural_rows}
            synthetic_host = {row.get("natural_host_sha256") for row in synthetic_rows}
            natural_source = {row.get("natural_source_sha256") for row in natural_rows}
            synthetic_source = {row.get("natural_source_sha256") for row in synthetic_rows}
            if (
                len(natural_host) != 1
                or natural_host != synthetic_host
                or len(natural_source) != 1
                or natural_source != synthetic_source
            ):
                invalid_matches += 1
                continue

            values = []
            for metric in CORE_METRICS:
                natural_mean = sum(float(row[metric]) for row in natural_rows) / len(natural_rows)
                synthetic_mean = sum(float(row[metric]) for row in synthetic_rows) / len(
                    synthetic_rows
                )
                values.append(natural_mean - synthetic_mean)
            accumulators[contrast].add(
                np.asarray(values, dtype=np.float64),
                host=str(next(iter(natural_host))),
                source=str(next(iter(natural_source))),
            )
        records = defaultdict(dict)

    for row in _iter_jsonl(features_path):
        if not _primary(row):
            continue
        match_id = int(row["match_id"])
        if current_match_id is None:
            current_match_id = match_id
        elif match_id != current_match_id:
            finish_match()
            current_match_id = match_id
        records[row["payload_class"]][int(row["mask"])] = row
    finish_match()

    contrasts: dict[str, Any] = {}
    for contrast, accumulator in accumulators.items():
        contrasts[contrast] = {
            "host_clustered": accumulator.summarize("host"),
            "source_clustered": accumulator.summarize("source"),
        }

    result = {
        "core_metrics": list(CORE_METRICS),
        "confirmatory_contrasts": list(CONFIRMATORY_CONTRASTS),
        "complete_matches_seen": complete_matches,
        "invalid_matches": invalid_matches,
        "method": (
            "CR1 sandwich standard errors for the intercept-only mean effect, separately clustered "
            "by natural host and by natural source/crawl file. 95% intervals use normal 1.96; "
            "cluster counts are reported and <20 is flagged."
        ),
        "omitted_cross_corpus_contrast": (
            "onion_natural-minus-surface_natural is not assigned a one-way clustered SE because "
            "surface and onion observations carry different host/source clusters."
        ),
        "contrasts": contrasts,
    }
    output_path = output_dir / "cluster_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["output"] = str(output_path)
    return result
