from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

FAMILY_ALPHA = 0.05
PREDECLARED_FAMILY_SIZE = 8  # 2 natural-vs-null contrasts x 4 core metrics


def _normal_two_sided_p(mean: float, se: float) -> float:
    if se <= 0 or not math.isfinite(se):
        raise ValueError("Normal p-value requires a finite positive standard error")
    z = abs(mean / se)
    return math.erfc(z / math.sqrt(2.0))


def _holm_adjust(rows: list[dict[str, Any]], *, family_size: int) -> None:
    estimable = [index for index, row in enumerate(rows) if row["p_two_sided_normal"] is not None]
    ordered = sorted(estimable, key=lambda index: float(rows[index]["p_two_sided_normal"]))
    previous = 0.0
    for rank, index in enumerate(ordered):
        # Use the full preregistered family size even if some cells are currently
        # non-estimable. This is conservative and prevents a failed cell from
        # silently shrinking the multiplicity burden.
        multiplier = max(1, family_size - rank)
        adjusted = min(1.0, multiplier * float(rows[index]["p_two_sided_normal"]))
        adjusted = max(previous, adjusted)
        rows[index]["p_holm_familywise"] = adjusted
        previous = adjusted
    for index, row in enumerate(rows):
        if index not in estimable:
            row["p_holm_familywise"] = None


def _level_rows(clustered: dict[str, Any], level: str, alpha: float) -> list[dict[str, Any]]:
    if level not in {"host", "source"}:
        raise ValueError("level must be host or source")
    key = f"{level}_clustered"
    critical = NormalDist().inv_cdf(1.0 - alpha / (2.0 * PREDECLARED_FAMILY_SIZE))
    rows: list[dict[str, Any]] = []

    for contrast, result in sorted(clustered["contrasts"].items()):
        for source_row in result[key]:
            mean = float(source_row["mean_difference"])
            se_raw = source_row["cr1_standard_error"]
            se = float(se_raw) if se_raw is not None else None
            estimable = se is not None and math.isfinite(se) and se > 0
            p = _normal_two_sided_p(mean, se) if estimable else None
            low = mean - critical * se if estimable else None
            high = mean + critical * se if estimable else None
            rows.append(
                {
                    "contrast": contrast,
                    "metric": source_row["metric"],
                    "cluster_level": level,
                    "n_matches": int(source_row["n_matches"]),
                    "cluster_count": int(source_row["cluster_count"]),
                    "effective_cluster_count": float(source_row["effective_cluster_count"]),
                    "max_cluster_size": int(source_row["max_cluster_size"]),
                    "mean_difference": mean,
                    "cr1_standard_error": se,
                    "p_two_sided_normal": p,
                    "bonferroni_simultaneous_ci_low": low,
                    "bonferroni_simultaneous_ci_high": high,
                    "bonferroni_critical_value": critical,
                    "family_alpha": alpha,
                    "family_size": PREDECLARED_FAMILY_SIZE,
                    "estimable": estimable,
                    "cluster_count_ge_20": bool(source_row["cluster_count_ge_20"]),
                    "eligible_for_confirmatory_claim": bool(
                        estimable and source_row["cluster_count_ge_20"]
                    ),
                }
            )

    if len(rows) != PREDECLARED_FAMILY_SIZE:
        raise ValueError(
            "Confirmatory family changed unexpectedly: "
            f"expected {PREDECLARED_FAMILY_SIZE} cells, found {len(rows)}"
        )
    _holm_adjust(rows, family_size=PREDECLARED_FAMILY_SIZE)
    for row in rows:
        p_holm = row["p_holm_familywise"]
        row["holm_reject_familywise_0_05"] = bool(
            row["eligible_for_confirmatory_claim"] and p_holm is not None and float(p_holm) <= alpha
        )
        low = row["bonferroni_simultaneous_ci_low"]
        high = row["bonferroni_simultaneous_ci_high"]
        row["bonferroni_ci_excludes_zero"] = bool(
            row["eligible_for_confirmatory_claim"]
            and low is not None
            and high is not None
            and (float(low) > 0.0 or float(high) < 0.0)
        )
    return rows


def analyze_familywise_core(
    config: dict[str, Any],
    clustered: dict[str, Any],
    *,
    alpha: float = FAMILY_ALPHA,
) -> dict[str, Any]:
    if not 0 < alpha < 1:
        raise ValueError("familywise alpha must be between 0 and 1")
    result = {
        "alpha": alpha,
        "predeclared_family_size": PREDECLARED_FAMILY_SIZE,
        "family_definition": "2 confirmatory natural-vs-null contrasts x 4 preregistered core metrics",
        "host": _level_rows(clustered, "host", alpha),
        "source": _level_rows(clustered, "source", alpha),
        "interpretation": (
            "Host-level rows are primary. Holm-adjusted normal p-values and Bonferroni simultaneous "
            "normal intervals control the fixed eight-cell family under the same large-cluster "
            "approximation as the CR1 report. Rows with fewer than 20 clusters are not eligible for "
            "a confirmatory claim even if a nominal or adjusted threshold is crossed. Source-file "
            "rows are sensitivity diagnostics, not automatically independent-crawl inference."
        ),
    }
    output_dir = Path(config["outputs"]["directory"])
    output_path = output_dir / "familywise_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["output"] = str(output_path)
    return result
