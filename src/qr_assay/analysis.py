from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fileio import open_text

FULL_METRICS = (
    "density",
    "centroid_x",
    "centroid_y",
    "centroid_radius",
    "radial_mean",
    "radial_std",
    "cov_trace",
    "principal_angle_deg",
    "anisotropy",
    "orientation_cos2",
    "orientation_sin2",
    "transition_h",
    "transition_v",
    "symmetry_rot180",
    "symmetry_horizontal",
    "symmetry_vertical",
)
DATA_METRICS = (
    "data_density",
    "data_centroid_x",
    "data_centroid_y",
    "data_centroid_radius",
    "data_radial_mean",
    "data_radial_std",
    "data_cov_trace",
    "data_principal_angle_deg",
    "data_anisotropy",
    "data_orientation_cos2",
    "data_orientation_sin2",
    "data_transition_h",
    "data_transition_v",
)
GROUP_METRICS = FULL_METRICS + DATA_METRICS
# Principal-axis orientation is axial (modulo 180 degrees). Never linearly
# difference raw angles; use cos(2theta)/sin(2theta) instead.
INFERENCE_METRICS = tuple(metric for metric in GROUP_METRICS if "principal_angle_deg" not in metric)
CLASSES = {
    "surface_natural",
    "onion_natural",
    "surface_synthetic",
    "onion_synthetic",
}
CONTRASTS = {
    "surface_natural-minus-surface_synthetic": ("surface_natural", "surface_synthetic"),
    "onion_natural-minus-onion_synthetic": ("onion_natural", "onion_synthetic"),
    "onion_natural-minus-surface_natural": ("onion_natural", "surface_natural"),
}


@dataclass
class RunningStat:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        value = float(value)
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0,
            "min": self.minimum if self.n else 0.0,
            "max": self.maximum if self.n else 0.0,
        }


def _empty_effects() -> dict[str, dict[str, RunningStat]]:
    return {
        contrast: {metric: RunningStat() for metric in INFERENCE_METRICS} for contrast in CONTRASTS
    }


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


def _group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["payload_class"],
        row.get("group_element"),
        int(row["rotation"]),
        row["reflection"],
        int(row["scale"]),
        bool(row["inverted"]),
        int(row["mask"]),
    )


def _add_group(groups: dict[tuple[Any, ...], dict[str, RunningStat]], row: dict[str, Any]) -> None:
    bucket = groups.setdefault(_group_key(row), {metric: RunningStat() for metric in GROUP_METRICS})
    for metric in GROUP_METRICS:
        bucket[metric].add(float(row[metric]))


def _serialize_groups(
    groups: dict[tuple[Any, ...], dict[str, RunningStat]],
) -> list[dict[str, Any]]:
    result = []
    for key in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
        payload_class, group_element, rotation, reflection, scale, inverted, mask = key
        metrics = groups[key]
        result.append(
            {
                "payload_class": payload_class,
                "group_element": group_element,
                "rotation": rotation,
                "reflection": reflection,
                "scale": scale,
                "inverted": inverted,
                "mask": mask,
                "count": metrics[GROUP_METRICS[0]].n,
                "metrics": {name: stat.as_dict() for name, stat in metrics.items()},
            }
        )
    return result


def _serialize_paired_effects(
    effects: dict[str, dict[str, RunningStat]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contrast, metrics in sorted(effects.items()):
        for metric, stat in sorted(metrics.items()):
            summary = stat.as_dict()
            n = int(summary["n"])
            std = float(summary["std"])
            mean = float(summary["mean"])
            sem = std / math.sqrt(n) if n else 0.0
            z = mean / sem if sem else (math.inf if mean else 0.0)
            p = math.erfc(abs(z) / math.sqrt(2.0)) if math.isfinite(z) else 0.0
            rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "n_matches": n,
                    "mean_difference": mean,
                    "std_difference": std,
                    "standard_error": sem,
                    "ci95_low": mean - 1.96 * sem,
                    "ci95_high": mean + 1.96 * sem,
                    "z_normal_approx": z,
                    "p_two_sided_normal_approx": p,
                    "cohen_dz": mean / std if std else 0.0,
                }
            )
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["p_two_sided_normal_approx"])
    previous = 0.0
    total = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * rows[index]["p_two_sided_normal_approx"])
        adjusted = max(previous, adjusted)
        rows[index]["p_holm"] = adjusted
        previous = adjusted
    return rows


def _payload_stratum(primary_rows: list[dict[str, Any]], bin_width: float) -> tuple[Any, ...]:
    first = primary_rows[0]
    mean_data_density = sum(float(row["base_data_density"]) for row in primary_rows) / len(
        primary_rows
    )
    density_bin = int(mean_data_density / bin_width)
    return (
        int(first["byte_length"]),
        int(first["qr_version"]),
        first["error_correction"],
        density_bin,
    )


def _add_class_contrasts(
    destination: dict[str, dict[str, RunningStat]],
    class_means: dict[str, dict[str, float]],
) -> None:
    for contrast, (left, right) in CONTRASTS.items():
        for metric in INFERENCE_METRICS:
            destination[contrast][metric].add(class_means[left][metric] - class_means[right][metric])


def analyze_features(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["outputs"]["directory"])
    path = output_dir / config["outputs"]["features_file"]
    if not path.exists():
        raise FileNotFoundError(f"Generate features first: {path}")
    bin_width = float(config["analysis"]["density_bin_width"])
    expected_masks = {int(mask) for mask in config["qr"]["masks"]}

    strata_counts: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_groups: dict[tuple[Any, ...], dict[str, RunningStat]] = {}
    paired_effects = _empty_effects()
    paired_effects_density_common_bin = _empty_effects()
    rows = 0
    qc_density_failures = 0
    qc_data_density_failures = 0
    match_violations = 0
    matches = 0
    density_common_bin_matches = 0
    primary_payloads = 0

    current_match_id: int | None = None
    match_records: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    current_payload_key: tuple[int, str, str] | None = None
    current_payload_primary: list[dict[str, Any]] = []

    def finish_payload_for_strata() -> None:
        nonlocal primary_payloads, current_payload_primary
        if not current_payload_primary:
            return
        primary_payloads += 1
        cls = current_payload_primary[0]["payload_class"]
        stratum = _payload_stratum(current_payload_primary, bin_width)
        strata_counts[stratum][cls] += 1
        current_payload_primary = []

    def finish_match() -> None:
        nonlocal matches, density_common_bin_matches, match_violations, match_records
        if current_match_id is None:
            return
        matches += 1
        ok = set(match_records) == CLASSES
        if ok:
            for cls in CLASSES:
                if set(match_records[cls]) != expected_masks:
                    ok = False
                    break
        if not ok:
            match_violations += 1
            match_records = defaultdict(dict)
            return

        class_means: dict[str, dict[str, float]] = {}
        class_density_bins: dict[str, int] = {}
        invariants = {"byte_length": set(), "qr_version": set(), "error_correction": set()}
        for cls in CLASSES:
            rows_for_class = list(match_records[cls].values())
            for name, invariant_values in invariants.items():
                invariant_values.update(row[name] for row in rows_for_class)
            class_means[cls] = {
                metric: sum(float(row[metric]) for row in rows_for_class) / len(rows_for_class)
                for metric in INFERENCE_METRICS
            }
            mean_base_data_density = sum(
                float(row["base_data_density"]) for row in rows_for_class
            ) / len(rows_for_class)
            class_density_bins[cls] = int(mean_base_data_density / bin_width)
        if any(len(values) != 1 for values in invariants.values()):
            match_violations += 1
        else:
            _add_class_contrasts(paired_effects, class_means)
            # Conservative density sensitivity: preserve the original four-way
            # match and keep it only when all four payloads occupy the same
            # mean data/ECC-density bin. This avoids the v0.1 mistake of density
            # balancing classes independently and silently breaking pairs.
            if len(set(class_density_bins.values())) == 1:
                density_common_bin_matches += 1
                _add_class_contrasts(paired_effects_density_common_bin, class_means)
        match_records = defaultdict(dict)

    for row in _iter_jsonl(path):
        rows += 1
        _add_group(all_groups, row)
        expected_density = (
            1.0 - float(row["base_density"]) if row["inverted"] else float(row["base_density"])
        )
        expected_data_density = (
            1.0 - float(row["base_data_density"])
            if row["inverted"]
            else float(row["base_data_density"])
        )
        if abs(float(row["density"]) - expected_density) > 1e-12:
            qc_density_failures += 1
        if abs(float(row["data_density"]) - expected_data_density) > 1e-12:
            qc_data_density_failures += 1
        if not _primary(row):
            continue

        row_match_id = int(row["match_id"])
        if current_match_id is None:
            current_match_id = row_match_id
        elif row_match_id != current_match_id:
            finish_match()
            current_match_id = row_match_id

        cls = row["payload_class"]
        mask = int(row["mask"])
        match_records[cls][mask] = row

        payload_key = (row_match_id, cls, row["payload_sha256"])
        if current_payload_key is None:
            current_payload_key = payload_key
        elif payload_key != current_payload_key:
            finish_payload_for_strata()
            current_payload_key = payload_key
        current_payload_primary.append(row)

    finish_payload_for_strata()
    finish_match()

    # Marginal equal-count density balancing is retained only as a descriptive
    # sensitivity table. It is not a paired estimand because classes can select
    # different match IDs within a stratum. Paired density sensitivity is the
    # common-bin match analysis above.
    quotas = {
        stratum: min(counts.get(cls, 0) for cls in CLASSES)
        for stratum, counts in strata_counts.items()
        if all(counts.get(cls, 0) for cls in CLASSES)
    }
    quotas = {key: value for key, value in quotas.items() if value > 0}

    used: dict[tuple[tuple[Any, ...], str], int] = defaultdict(int)
    selected_per_class: dict[str, int] = defaultdict(int)
    marginal_balanced_groups: dict[tuple[Any, ...], dict[str, RunningStat]] = {}
    payload_buffer: list[dict[str, Any]] = []
    payload_key: tuple[int, str, str] | None = None

    def flush_payload() -> None:
        nonlocal payload_buffer
        if not payload_buffer:
            return
        primary = [row for row in payload_buffer if _primary(row)]
        if not primary:
            payload_buffer = []
            return
        cls = primary[0]["payload_class"]
        stratum = _payload_stratum(primary, bin_width)
        key = (stratum, cls)
        if used[key] < quotas.get(stratum, 0):
            used[key] += 1
            selected_per_class[cls] += 1
            for row in payload_buffer:
                _add_group(marginal_balanced_groups, row)
        payload_buffer = []

    for row in _iter_jsonl(path):
        key = (int(row["match_id"]), row["payload_class"], row["payload_sha256"])
        if payload_key is None:
            payload_key = key
        elif key != payload_key:
            flush_payload()
            payload_key = key
        payload_buffer.append(row)
    flush_payload()

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    source_granularities = {
        "surface": config["sources"]["surface"].get("granularity", "url"),
        "onion": config["sources"]["onion"].get("granularity", "url"),
    }
    result = {
        "rows": rows,
        "payloads": primary_payloads,
        "matches": matches,
        "all_groups": _serialize_groups(all_groups),
        "marginal_density_balanced_groups": _serialize_groups(marginal_balanced_groups),
        # Backward-compatible alias; explicitly labeled as marginal in metadata.
        "balanced_groups": _serialize_groups(marginal_balanced_groups),
        "paired_effects": _serialize_paired_effects(paired_effects),
        "paired_effects_density_common_bin": _serialize_paired_effects(
            paired_effects_density_common_bin
        ),
        "primary_feature_region": "data_ecc_modules_only",
        "inference_unit": "match_id after averaging all configured QR masks",
        "inference_note": (
            "Normal-approximation p-values are secondary descriptive diagnostics. "
            "For URL-granularity publication analyses, use host/crawl-clustered resampling."
        ),
        "source_granularities": source_granularities,
        "balance": {
            "density_bin_width": bin_width,
            "density_basis": "mean base_data_density across masks",
            "paired_common_bin_matches": density_common_bin_matches,
            "paired_common_bin_fraction": density_common_bin_matches / matches if matches else 0.0,
            "marginal_balance_is_pair_preserving": False,
            "eligible_strata": len(quotas),
            "selected_payloads": sum(selected_per_class.values()),
            "selected_per_class": {cls: selected_per_class.get(cls, 0) for cls in sorted(CLASSES)},
        },
        "quality_control": {
            "density_transform_failures": qc_density_failures,
            "data_density_transform_failures": qc_data_density_failures,
            "matched_control_failures": match_violations,
            "expected_masks_per_payload": sorted(expected_masks),
            "passed": qc_density_failures == 0
            and qc_data_density_failures == 0
            and match_violations == 0,
        },
        "features_sha256": digest.hexdigest(),
    }
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def _find_primary(groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        group
        for group in groups
        if group["rotation"] == 0
        and group["reflection"] == "none"
        and group["scale"] == 1
        and not group["inverted"]
    ]


def _effect_table_lines(effects: list[dict[str, Any]], core_metrics: set[str]) -> list[str]:
    lines = []
    for effect in effects:
        if effect["metric"] not in core_metrics:
            continue
        lines.append(
            f"| {effect['contrast']} | {effect['metric']} | {effect['n_matches']:,} | "
            f"{effect['mean_difference']:.6g} | [{effect['ci95_low']:.6g}, {effect['ci95_high']:.6g}] | "
            f"{effect['cohen_dz']:.4f} | {effect['p_holm']:.3g} |"
        )
    return lines


def write_report(
    config: dict[str, Any],
    preparation: dict[str, Any],
    generation: dict[str, Any],
    analysis: dict[str, Any],
) -> Path:
    output_dir = Path(config["outputs"]["directory"])
    report_path = output_dir / config["outputs"]["report_file"]
    primary = _find_primary(analysis["marginal_density_balanced_groups"] or analysis["all_groups"])
    lines = [
        "# QR URL Geometry Assay — run report",
        "",
        "## Outcome",
        "",
        f"- Matched URL quadruples: **{preparation['matched_pairs']:,}**",
        f"- QR masks per payload: **{generation['masks_per_payload']}**",
        f"- QR transform rows: **{generation['rows']:,}**",
        f"- Quality control: **{'PASS' if analysis['quality_control']['passed'] else 'FAIL'}**",
        f"- Synthetic null mode: **{preparation['synthetic_mode']}**",
        f"- Primary feature region: **{analysis['primary_feature_region']}**",
        f"- Inference unit: **{analysis['inference_unit']}**",
        f"- Four-way same-density-bin matches: **{analysis['balance']['paired_common_bin_matches']:,}**",
        "",
        "Rotation/reflection rows are stimulus-equivariance calibration. They are not evidence of neural cyclicity.",
        "",
        "## Marginal density-balanced baseline by mask (descriptive only)",
        "",
        "This table equalizes class counts within density strata but is not pair-preserving. The paired common-bin sensitivity below is the valid density-conditioned paired analysis.",
        "",
        "| payload class | mask | n | full density | data/ECC density | data centroid radius | data radial mean | data cos(2θ) | data sin(2θ) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in sorted(primary, key=lambda item: (item["payload_class"], item["mask"])):
        m = group["metrics"]
        lines.append(
            f"| {group['payload_class']} | {group['mask']} | {group['count']:,} | "
            f"{m['density']['mean']:.6f} | {m['data_density']['mean']:.6f} | "
            f"{m['data_centroid_radius']['mean']:.6f} | {m['data_radial_mean']['mean']:.6f} | "
            f"{m['data_orientation_cos2']['mean']:.6f} | {m['data_orientation_sin2']['mean']:.6f} |"
        )
    core_metrics = {
        "data_radial_mean",
        "data_centroid_radius",
        "data_orientation_cos2",
        "data_orientation_sin2",
    }
    lines.extend(
        [
            "",
            "## Primary paired effects (all matched units, averaged over masks)",
            "",
            "| contrast | metric | matched units | mean difference | 95% CI | Cohen dz | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|",
            *_effect_table_lines(analysis["paired_effects"], core_metrics),
            "",
            "## Density-common-bin paired sensitivity",
            "",
            "Only complete four-way matches whose four payloads fall in the same mean data/ECC-density bin are retained; the original match_id is never broken.",
            "",
            "| contrast | metric | matched units | mean difference | 95% CI | Cohen dz | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|",
            *_effect_table_lines(analysis["paired_effects_density_common_bin"], core_metrics),
            "",
            "**Inference warning.** Normal-approximation p-values are secondary. URL-granularity runs can contain repeated hosts/crawls and require clustered resampling before publication.",
            "",
            "The stage-1 assay measures lexical/sequence structure in encoded QR matrices. It does not measure payload semantics or neural carrier rotation.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
