from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Hashable

import numpy as np

from .cluster import CORE_METRICS
from .fileio import open_text

DID_FAMILY_SIZE = len(CORE_METRICS)
DID_ALPHA = 0.05


@dataclass
class GroupTotal:
    count: int = 0
    sums: np.ndarray = field(default_factory=lambda: np.zeros(len(CORE_METRICS), dtype=np.float64))


@dataclass
class TwoWayAccumulator:
    n: int = 0
    sums: np.ndarray = field(default_factory=lambda: np.zeros(len(CORE_METRICS), dtype=np.float64))
    a: dict[Hashable, GroupTotal] = field(default_factory=dict)
    b: dict[Hashable, GroupTotal] = field(default_factory=dict)
    ab: dict[tuple[Hashable, Hashable], GroupTotal] = field(default_factory=dict)

    @staticmethod
    def _add_group(table: dict[Any, GroupTotal], key: Hashable, values: np.ndarray) -> None:
        if key in {None, ""}:
            raise ValueError("Two-way cluster key must be present")
        total = table.get(key)
        if total is None:
            total = GroupTotal()
            table[key] = total
        total.count += 1
        total.sums += values

    def add(self, values: np.ndarray, *, cluster_a: Hashable, cluster_b: Hashable) -> None:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (len(CORE_METRICS),):
            raise ValueError("Two-way effect vector has an unexpected shape")
        self.n += 1
        self.sums += vector
        self._add_group(self.a, cluster_a, vector)
        self._add_group(self.b, cluster_b, vector)
        self._add_group(self.ab, (cluster_a, cluster_b), vector)

    def mean(self) -> np.ndarray:
        return self.sums / self.n if self.n else np.zeros(len(CORE_METRICS), dtype=np.float64)

    def _component_variance(self, table: dict[Any, GroupTotal], mean: np.ndarray) -> np.ndarray:
        groups = len(table)
        if self.n == 0 or groups < 2:
            return np.full(len(CORE_METRICS), np.nan)
        score_squares = np.zeros(len(CORE_METRICS), dtype=np.float64)
        for total in table.values():
            score = total.sums - total.count * mean
            score_squares += np.square(score)
        return (groups / (groups - 1)) * score_squares / (self.n**2)

    @staticmethod
    def _effective_clusters(table: dict[Any, GroupTotal]) -> float:
        counts = np.asarray([total.count for total in table.values()], dtype=np.float64)
        if not counts.size:
            return 0.0
        return float(counts.sum() ** 2 / np.square(counts).sum())

    @staticmethod
    def _max_cluster(table: dict[Any, GroupTotal]) -> int:
        return max((total.count for total in table.values()), default=0)

    def summarize(self, *, level_a: str, level_b: str) -> list[dict[str, Any]]:
        mean = self.mean()
        var_a = self._component_variance(self.a, mean)
        var_b = self._component_variance(self.b, mean)
        var_ab = self._component_variance(self.ab, mean)
        rows = []
        for index, metric in enumerate(CORE_METRICS):
            components_finite = bool(
                np.isfinite(var_a[index])
                and np.isfinite(var_b[index])
                and np.isfinite(var_ab[index])
            )
            raw_variance = (
                float(var_a[index] + var_b[index] - var_ab[index])
                if components_finite
                else None
            )
            negative_variance = bool(raw_variance is not None and raw_variance < 0.0)
            variance = max(0.0, raw_variance) if raw_variance is not None else None
            se = math.sqrt(variance) if variance is not None else None
            rows.append(
                {
                    "metric": metric,
                    "n_matches": self.n,
                    "mean_difference_in_differences": float(mean[index]),
                    "cluster_level_a": level_a,
                    "cluster_level_b": level_b,
                    "cluster_count_a": len(self.a),
                    "cluster_count_b": len(self.b),
                    "intersection_cluster_count": len(self.ab),
                    "effective_cluster_count_a": self._effective_clusters(self.a),
                    "effective_cluster_count_b": self._effective_clusters(self.b),
                    "max_cluster_size_a": self._max_cluster(self.a),
                    "max_cluster_size_b": self._max_cluster(self.b),
                    "variance_component_a": (
                        float(var_a[index]) if np.isfinite(var_a[index]) else None
                    ),
                    "variance_component_b": (
                        float(var_b[index]) if np.isfinite(var_b[index]) else None
                    ),
                    "variance_component_intersection": (
                        float(var_ab[index]) if np.isfinite(var_ab[index]) else None
                    ),
                    "cgm_variance_raw": raw_variance,
                    "cgm_standard_error": se,
                    "negative_cgm_variance": negative_variance,
                    "estimable": bool(
                        se is not None
                        and math.isfinite(se)
                        and se > 0.0
                        and not negative_variance
                    ),
                    "cluster_counts_ge_20": len(self.a) >= 20 and len(self.b) >= 20,
                }
            )
        return rows


def _normal_p(mean: float, se: float) -> float:
    return math.erfc(abs(mean / se) / math.sqrt(2.0))


def _apply_familywise(rows: list[dict[str, Any]], *, alpha: float = DID_ALPHA) -> None:
    if len(rows) != DID_FAMILY_SIZE:
        raise ValueError(f"DiD family must contain exactly {DID_FAMILY_SIZE} core metrics")
    critical = NormalDist().inv_cdf(1.0 - alpha / (2.0 * DID_FAMILY_SIZE))
    estimable_indices = []
    for index, row in enumerate(rows):
        se = row["cgm_standard_error"]
        eligible = bool(row["estimable"] and row["cluster_counts_ge_20"])
        row["eligible_for_confirmatory_claim"] = eligible
        if se is not None and float(se) > 0 and math.isfinite(float(se)):
            mean = float(row["mean_difference_in_differences"])
            se_value = float(se)
            row["p_two_sided_normal"] = _normal_p(mean, se_value)
            row["bonferroni_simultaneous_ci_low"] = mean - critical * se_value
            row["bonferroni_simultaneous_ci_high"] = mean + critical * se_value
            row["bonferroni_critical_value"] = critical
            estimable_indices.append(index)
        else:
            row["p_two_sided_normal"] = None
            row["bonferroni_simultaneous_ci_low"] = None
            row["bonferroni_simultaneous_ci_high"] = None
            row["bonferroni_critical_value"] = critical

    ordered = sorted(
        estimable_indices,
        key=lambda index: float(rows[index]["p_two_sided_normal"]),
    )
    previous = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(
            1.0,
            (DID_FAMILY_SIZE - rank) * float(rows[index]["p_two_sided_normal"]),
        )
        adjusted = max(previous, adjusted)
        rows[index]["p_holm_familywise"] = adjusted
        previous = adjusted
    for index, row in enumerate(rows):
        if index not in estimable_indices:
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
            row["eligible_for_confirmatory_claim"]
            and holm is not None
            and float(holm) <= alpha
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
    """Test whether natural-vs-null QR geometry differs between Onion and Surface.

    Per match and metric:

        (onion_natural - onion_synthetic)
        - (surface_natural - surface_synthetic)

    All configured QR masks are averaged before this contrast. Host uncertainty is
    two-way clustered by the natural Surface host and natural Onion host using the
    Cameron-Gelbach-Miller inclusion-exclusion form V_A + V_B - V_AB. Source-file
    clustering is reported as a parallel sensitivity analysis.
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
    host_acc = TwoWayAccumulator()
    source_acc = TwoWayAccumulator()
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

        means: dict[str, np.ndarray] = {}
        for cls in classes:
            rows = list(records[cls].values())
            means[cls] = np.asarray(
                [
                    sum(float(row[metric]) for row in rows) / len(rows)
                    for metric in CORE_METRICS
                ],
                dtype=np.float64,
            )

        surface_nat = list(records["surface_natural"].values())
        surface_syn = list(records["surface_synthetic"].values())
        onion_nat = list(records["onion_natural"].values())
        onion_syn = list(records["onion_synthetic"].values())

        surface_hosts = {row.get("natural_host_sha256") for row in surface_nat}
        surface_hosts_syn = {row.get("natural_host_sha256") for row in surface_syn}
        onion_hosts = {row.get("natural_host_sha256") for row in onion_nat}
        onion_hosts_syn = {row.get("natural_host_sha256") for row in onion_syn}
        surface_sources = {row.get("natural_source_sha256") for row in surface_nat}
        surface_sources_syn = {row.get("natural_source_sha256") for row in surface_syn}
        onion_sources = {row.get("natural_source_sha256") for row in onion_nat}
        onion_sources_syn = {row.get("natural_source_sha256") for row in onion_syn}

        valid_clusters = (
            len(surface_hosts) == 1
            and surface_hosts == surface_hosts_syn
            and len(onion_hosts) == 1
            and onion_hosts == onion_hosts_syn
            and len(surface_sources) == 1
            and surface_sources == surface_sources_syn
            and len(onion_sources) == 1
            and onion_sources == onion_sources_syn
            and next(iter(surface_hosts)) not in {None, ""}
            and next(iter(onion_hosts)) not in {None, ""}
            and next(iter(surface_sources)) not in {None, ""}
            and next(iter(onion_sources)) not in {None, ""}
        )
        if not valid_clusters:
            invalid_matches += 1
            records = defaultdict(dict)
            return

        did = (
            means["onion_natural"]
            - means["onion_synthetic"]
            - means["surface_natural"]
            + means["surface_synthetic"]
        )
        host_acc.add(
            did,
            cluster_a=str(next(iter(surface_hosts))),
            cluster_b=str(next(iter(onion_hosts))),
        )
        source_acc.add(
            did,
            cluster_a=str(next(iter(surface_sources))),
            cluster_b=str(next(iter(onion_sources))),
        )
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

    host_rows = host_acc.summarize(level_a="surface_host", level_b="onion_host")
    source_rows = source_acc.summarize(
        level_a="surface_source_file",
        level_b="onion_source_file",
    )
    _apply_familywise(host_rows)
    _apply_familywise(source_rows)

    result = {
        "contrast": "(onion_natural-onion_synthetic)-(surface_natural-surface_synthetic)",
        "core_metrics": list(CORE_METRICS),
        "family_size": DID_FAMILY_SIZE,
        "alpha": DID_ALPHA,
        "complete_matches": complete_matches,
        "invalid_matches": invalid_matches,
        "host_two_way_cgm": host_rows,
        "source_file_two_way_cgm": source_rows,
        "method": (
            "Match-level difference-in-differences after averaging all QR masks. Two-way CR1/CGM "
            "variance uses Surface cluster + Onion cluster - Surface×Onion intersection cluster. "
            "Negative finite-sample CGM variance is flagged and is not claim-eligible. Strict "
            "confirmatory pass additionally requires >=20 clusters on both host dimensions, Holm "
            "FWER p<=0.05 and the fixed-four-cell Bonferroni simultaneous interval to exclude zero."
        ),
        "interpretation": (
            "This directly tests whether natural-vs-declared-null QR geometry differs between the "
            "two corpora under exact cross-corpus byte-length matching. It is a structural "
            "difference-in-differences, not a semantic or Chomskyan-grammar measurement by itself."
        ),
    }
    output_path = output_dir / "did_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["output"] = str(output_path)
    return result
