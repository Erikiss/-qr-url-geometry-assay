from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .cluster import CORE_METRICS
from .fileio import open_text

DID_FAMILY_SIZE = len(CORE_METRICS)
DID_ALPHA = 0.05


@dataclass
class SideAccumulator:
    n: int = 0
    sums: np.ndarray = field(default_factory=lambda: np.zeros(len(CORE_METRICS), dtype=np.float64))
    clusters: dict[str, list[Any]] = field(default_factory=dict)

    def add(self, values: np.ndarray, *, cluster: str) -> None:
        if not cluster:
            raise ValueError("Natural cluster key must be present")
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (len(CORE_METRICS),):
            raise ValueError("Side effect vector has an unexpected shape")
        self.n += 1
        self.sums += vector
        if cluster not in self.clusters:
            self.clusters[cluster] = [0, np.zeros(len(CORE_METRICS), dtype=np.float64)]
        self.clusters[cluster][0] += 1
        self.clusters[cluster][1] += vector

    def mean(self) -> np.ndarray:
        return self.sums / self.n if self.n else np.full(len(CORE_METRICS), np.nan)

    def variance(self) -> np.ndarray:
        groups = len(self.clusters)
        if self.n == 0 or groups < 2:
            return np.full(len(CORE_METRICS), np.nan)
        mean = self.mean()
        score_squares = np.zeros(len(CORE_METRICS), dtype=np.float64)
        for count, sums in self.clusters.values():
            score = sums - count * mean
            score_squares += np.square(score)
        return (groups / (groups - 1)) * score_squares / (self.n**2)

    def effective_clusters(self) -> float:
        counts = np.asarray([value[0] for value in self.clusters.values()], dtype=np.float64)
        if not counts.size:
            return 0.0
        return float(counts.sum() ** 2 / np.square(counts).sum())

    def max_cluster(self) -> int:
        return max((int(value[0]) for value in self.clusters.values()), default=0)


class IndependentDifference:
    """Difference of two independently clustered corpus means.

    The artificial one-to-one Surface/Onion matching is never used as a covariance
    assumption. Variances are estimated on each natural side and then added.
    """

    def __init__(self) -> None:
        self.surface = SideAccumulator()
        self.onion = SideAccumulator()

    def add_surface(self, values: np.ndarray, *, cluster: str) -> None:
        self.surface.add(values, cluster=cluster)

    def add_onion(self, values: np.ndarray, *, cluster: str) -> None:
        self.onion.add(values, cluster=cluster)

    def summarize(self, *, surface_level: str, onion_level: str) -> list[dict[str, Any]]:
        mean = self.onion.mean() - self.surface.mean()
        var_surface = self.surface.variance()
        var_onion = self.onion.variance()
        rows: list[dict[str, Any]] = []
        for index, metric in enumerate(CORE_METRICS):
            finite = bool(np.isfinite(var_surface[index]) and np.isfinite(var_onion[index]))
            variance = float(var_surface[index] + var_onion[index]) if finite else None
            se = math.sqrt(max(0.0, variance)) if variance is not None else None
            rows.append(
                {
                    "metric": metric,
                    "n_surface": self.surface.n,
                    "n_onion": self.onion.n,
                    "mean_difference_in_differences": float(mean[index]),
                    "surface_cluster_level": surface_level,
                    "onion_cluster_level": onion_level,
                    "surface_cluster_count": len(self.surface.clusters),
                    "onion_cluster_count": len(self.onion.clusters),
                    "surface_effective_cluster_count": self.surface.effective_clusters(),
                    "onion_effective_cluster_count": self.onion.effective_clusters(),
                    "surface_max_cluster_size": self.surface.max_cluster(),
                    "onion_max_cluster_size": self.onion.max_cluster(),
                    "surface_cluster_variance": (
                        float(var_surface[index]) if np.isfinite(var_surface[index]) else None
                    ),
                    "onion_cluster_variance": (
                        float(var_onion[index]) if np.isfinite(var_onion[index]) else None
                    ),
                    "independent_cluster_standard_error": se,
                    "estimable": bool(se is not None and math.isfinite(se) and se > 0.0),
                    "cluster_counts_ge_20": (
                        len(self.surface.clusters) >= 20 and len(self.onion.clusters) >= 20
                    ),
                }
            )
        return rows


def _normal_p(mean: float, se: float) -> float:
    return math.erfc(abs(mean / se) / math.sqrt(2.0))


def _apply_familywise(rows: list[dict[str, Any]], *, alpha: float = DID_ALPHA) -> None:
    if len(rows) != DID_FAMILY_SIZE:
        raise ValueError(f"DiD family must contain exactly {DID_FAMILY_SIZE} core metrics")
    critical = NormalDist().inv_cdf(1.0 - alpha / (2.0 * DID_FAMILY_SIZE))
    estimable = []
    for index, row in enumerate(rows):
        se = row["independent_cluster_standard_error"]
        eligible = bool(row["estimable"] and row["cluster_counts_ge_20"])
        row["eligible_for_confirmatory_claim"] = eligible
        if se is not None and float(se) > 0 and math.isfinite(float(se)):
            mean = float(row["mean_difference_in_differences"])
            se_value = float(se)
            row["p_two_sided_normal"] = _normal_p(mean, se_value)
            row["bonferroni_simultaneous_ci_low"] = mean - critical * se_value
            row["bonferroni_simultaneous_ci_high"] = mean + critical * se_value
            estimable.append(index)
        else:
            row["p_two_sided_normal"] = None
            row["bonferroni_simultaneous_ci_low"] = None
            row["bonferroni_simultaneous_ci_high"] = None
        row["bonferroni_critical_value"] = critical

    ordered = sorted(estimable, key=lambda i: float(rows[i]["p_two_sided_normal"]))
    previous = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (DID_FAMILY_SIZE - rank) * float(rows[index]["p_two_sided_normal"]))
        adjusted = max(previous, adjusted)
        rows[index]["p_holm_familywise"] = adjusted
        previous = adjusted
    for index, row in enumerate(rows):
        if index not in estimable:
            row["p_holm_familywise"] = None
        low = row["bonferroni_simultaneous_ci_low"]
        high = row["bonferroni_simultaneous_ci_high"]
        holm = row["p_holm_familywise"]
        row["bonferroni_ci_excludes_zero"] = bool(
            row["eligible_for_confirmatory_claim"]
            and low is not None
            and high is not None
            and (float(low) > 0.0 or float(high) < 0.0)
        )
        row["holm_reject_familywise_0_05"] = bool(
            row["eligible_for_confirmatory_claim"] and holm is not None and float(holm) <= alpha
        )
        row["strict_confirmatory_pass"] = bool(
            row["holm_reject_familywise_0_05"] and row["bonferroni_ci_excludes_zero"]
        )


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


def analyze_difference_in_differences(config: dict[str, Any]) -> dict[str, Any]:
    """Compare Onion and Surface natural-minus-null effects without invented pairing covariance.

    Exact byte-length pairing is a design/balance device only. The point estimate is
    mean(Onion natural-null) - mean(Surface natural-null). Its variance is the sum
    of the two one-way natural-host-clustered variances, so arbitrary re-pairing
    within a matched length stratum cannot change either the estimate or its SE.
    """
    output_dir = Path(config["outputs"]["directory"])
    features_path = output_dir / config["outputs"]["features_file"]
    if not features_path.exists():
        raise FileNotFoundError(f"Generate features first: {features_path}")
    expected_masks = {int(mask) for mask in config["qr"]["masks"]}
    classes = {
        "surface_natural",
        "surface_synthetic",
        "onion_natural",
        "onion_synthetic",
    }
    host = {"surface": SideAccumulator(), "onion": SideAccumulator()}
    source = {"surface": SideAccumulator(), "onion": SideAccumulator()}
    current_match_id: int | None = None
    records: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    complete_matches = 0
    invalid_matches = 0

    def finish_match() -> None:
        nonlocal records, complete_matches, invalid_matches
        if current_match_id is None:
            return
        if set(records) != classes or any(set(records[cls]) != expected_masks for cls in classes):
            invalid_matches += 1
            records = defaultdict(dict)
            return

        effects: dict[str, np.ndarray] = {}
        cluster_keys: dict[str, tuple[str, str]] = {}
        for corpus in ("surface", "onion"):
            natural_rows = list(records[f"{corpus}_natural"].values())
            synthetic_rows = list(records[f"{corpus}_synthetic"].values())
            hosts = {row.get("natural_host_sha256") for row in natural_rows}
            hosts_syn = {row.get("natural_host_sha256") for row in synthetic_rows}
            sources = {row.get("natural_source_sha256") for row in natural_rows}
            sources_syn = {row.get("natural_source_sha256") for row in synthetic_rows}
            metadata = {
                (row.get("byte_length"), row.get("qr_version"), row.get("error_correction"))
                for row in natural_rows + synthetic_rows
            }
            if (
                len(hosts) != 1
                or hosts != hosts_syn
                or len(sources) != 1
                or sources != sources_syn
                or next(iter(hosts)) in {None, ""}
                or next(iter(sources)) in {None, ""}
                or len(metadata) != 1
                or any(value is None for value in next(iter(metadata)))
            ):
                invalid_matches += 1
                records = defaultdict(dict)
                return
            values = []
            for metric in CORE_METRICS:
                natural_mean = sum(float(row[metric]) for row in natural_rows) / len(natural_rows)
                synthetic_mean = sum(float(row[metric]) for row in synthetic_rows) / len(
                    synthetic_rows
                )
                values.append(natural_mean - synthetic_mean)
            effects[corpus] = np.asarray(values, dtype=np.float64)
            cluster_keys[corpus] = (str(next(iter(hosts))), str(next(iter(sources))))

        surface_meta = {
            (row.get("byte_length"), row.get("qr_version"), row.get("error_correction"))
            for cls in ("surface_natural", "surface_synthetic")
            for row in records[cls].values()
        }
        onion_meta = {
            (row.get("byte_length"), row.get("qr_version"), row.get("error_correction"))
            for cls in ("onion_natural", "onion_synthetic")
            for row in records[cls].values()
        }
        if surface_meta != onion_meta:
            invalid_matches += 1
            records = defaultdict(dict)
            return

        for corpus in ("surface", "onion"):
            host_key, source_key = cluster_keys[corpus]
            host[corpus].add(effects[corpus], cluster=host_key)
            source[corpus].add(effects[corpus], cluster=source_key)
        complete_matches += 1
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

    def summarize(level: str) -> list[dict[str, Any]]:
        tables = host if level == "host" else source
        difference = IndependentDifference()
        difference.surface = tables["surface"]
        difference.onion = tables["onion"]
        rows = difference.summarize(
            surface_level=f"surface_{level}",
            onion_level=f"onion_{level}",
        )
        _apply_familywise(rows)
        return rows

    host_rows = summarize("host")
    source_rows = summarize("source")
    result = {
        "contrast": "(onion_natural-onion_synthetic)-(surface_natural-surface_synthetic)",
        "core_metrics": list(CORE_METRICS),
        "family_size": DID_FAMILY_SIZE,
        "alpha": DID_ALPHA,
        "complete_matches": complete_matches,
        "invalid_matches": invalid_matches,
        "host_independent_cluster_difference": host_rows,
        "source_file_independent_cluster_difference": source_rows,
        "method": (
            "Match-level natural-minus-null effects after averaging QR masks. Exact cross-corpus "
            "byte-length/QR matching is used for balance only. DiD uncertainty is pairing-invariant: "
            "the one-way CR1 variance of the Onion mean and the one-way CR1 variance of the Surface "
            "mean are added, separately for natural-host and source-file clustering."
        ),
        "interpretation": (
            "This tests whether natural-vs-declared-null QR geometry differs between corpora on "
            "their exact shared support. It is structural, not a semantic or Chomskyan claim."
        ),
    }
    output_path = output_dir / "did_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["output"] = str(output_path)
    return result
