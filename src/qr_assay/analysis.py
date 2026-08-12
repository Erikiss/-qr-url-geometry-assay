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

METRICS = (
    "density",
    "centroid_x",
    "centroid_y",
    "centroid_radius",
    "radial_mean",
    "radial_std",
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
        int(row["rotation"]),
        row["reflection"],
        int(row["scale"]),
        bool(row["inverted"]),
    )


def _stratum(row: dict[str, Any], bin_width: float) -> tuple[Any, ...]:
    density_bin = int(float(row["base_density"]) / bin_width)
    return (
        int(row["byte_length"]),
        int(row["qr_version"]),
        row["error_correction"],
        int(row["mask"]),
        density_bin,
    )


def _add_group(groups: dict[tuple[Any, ...], dict[str, RunningStat]], row: dict[str, Any]) -> None:
    bucket = groups.setdefault(_group_key(row), {metric: RunningStat() for metric in METRICS})
    for metric in METRICS:
        bucket[metric].add(float(row[metric]))


def _serialize_groups(
    groups: dict[tuple[Any, ...], dict[str, RunningStat]],
) -> list[dict[str, Any]]:
    result = []
    for key in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
        payload_class, rotation, reflection, scale, inverted = key
        metrics = groups[key]
        result.append(
            {
                "payload_class": payload_class,
                "rotation": rotation,
                "reflection": reflection,
                "scale": scale,
                "inverted": inverted,
                "count": metrics[METRICS[0]].n,
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
                    "n": n,
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


def analyze_features(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["outputs"]["directory"])
    path = output_dir / config["outputs"]["features_file"]
    if not path.exists():
        raise FileNotFoundError(f"Generate features first: {path}")
    bin_width = float(config["analysis"]["density_bin_width"])
    classes = {
        "surface_natural",
        "onion_natural",
        "surface_synthetic",
        "onion_synthetic",
    }
    strata_counts: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_groups: dict[tuple[Any, ...], dict[str, RunningStat]] = {}
    rows = 0
    qc_density_failures = 0
    matches = 0
    primary_payloads = 0
    match_violations = 0
    current_match_id: int | None = None
    current_match_records: dict[str, dict[str, Any]] = {}
    paired_effects: dict[str, dict[str, RunningStat]] = {
        "surface_natural-minus-surface_synthetic": {metric: RunningStat() for metric in METRICS},
        "onion_natural-minus-onion_synthetic": {metric: RunningStat() for metric in METRICS},
        "onion_natural-minus-surface_natural": {metric: RunningStat() for metric in METRICS},
    }

    def finish_match() -> None:
        nonlocal matches, match_violations, current_match_records
        if current_match_id is None:
            return
        matches += 1
        if len(current_match_records) != 4:
            match_violations += 1
        else:
            lengths = {int(value["byte_length"]) for value in current_match_records.values()}
            versions = {int(value["qr_version"]) for value in current_match_records.values()}
            masks = {int(value["mask"]) for value in current_match_records.values()}
            if len(lengths) != 1 or len(versions) != 1 or len(masks) != 1:
                match_violations += 1
            contrasts = {
                "surface_natural-minus-surface_synthetic": (
                    "surface_natural",
                    "surface_synthetic",
                ),
                "onion_natural-minus-onion_synthetic": (
                    "onion_natural",
                    "onion_synthetic",
                ),
                "onion_natural-minus-surface_natural": (
                    "onion_natural",
                    "surface_natural",
                ),
            }
            for contrast, (left, right) in contrasts.items():
                for metric in METRICS:
                    paired_effects[contrast][metric].add(
                        float(current_match_records[left][metric])
                        - float(current_match_records[right][metric])
                    )
        current_match_records = {}

    for row in _iter_jsonl(path):
        rows += 1
        _add_group(all_groups, row)
        expected_density = (
            1.0 - float(row["base_density"]) if row["inverted"] else float(row["base_density"])
        )
        if abs(float(row["density"]) - expected_density) > 1e-12:
            qc_density_failures += 1
        if _primary(row):
            primary_payloads += 1
            stratum = _stratum(row, bin_width)
            strata_counts[stratum][row["payload_class"]] += 1
            row_match_id = int(row["match_id"])
            if current_match_id is None:
                current_match_id = row_match_id
            elif row_match_id != current_match_id:
                finish_match()
                current_match_id = row_match_id
            current_match_records[row["payload_class"]] = row
    finish_match()
    quotas = {
        stratum: min(counts.get(cls, 0) for cls in classes)
        for stratum, counts in strata_counts.items()
        if all(counts.get(cls, 0) for cls in classes)
    }
    quotas = {key: value for key, value in quotas.items() if value > 0}
    used: dict[tuple[tuple[Any, ...], str], int] = defaultdict(int)
    selected_per_class: dict[str, int] = defaultdict(int)
    balanced_groups: dict[tuple[Any, ...], dict[str, RunningStat]] = {}
    current_payload: tuple[str, str] | None = None
    accept_current = False
    for row in _iter_jsonl(path):
        cls = row["payload_class"]
        digest = row["payload_sha256"]
        selection_key = (cls, digest)
        if selection_key != current_payload:
            current_payload = selection_key
            stratum = _stratum(row, bin_width)
            key = (stratum, cls)
            if used[key] < quotas.get(stratum, 0):
                used[key] += 1
                selected_per_class[cls] += 1
                accept_current = True
            else:
                accept_current = False
        if accept_current:
            _add_group(balanced_groups, row)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result = {
        "rows": rows,
        "payloads": primary_payloads,
        "matches": matches,
        "all_groups": _serialize_groups(all_groups),
        "balanced_groups": _serialize_groups(balanced_groups),
        "paired_effects": _serialize_paired_effects(paired_effects),
        "balance": {
            "density_bin_width": bin_width,
            "eligible_strata": len(quotas),
            "selected_payloads": sum(selected_per_class.values()),
            "selected_per_class": {cls: selected_per_class.get(cls, 0) for cls in sorted(classes)},
        },
        "quality_control": {
            "density_transform_failures": qc_density_failures,
            "matched_control_failures": match_violations,
            "passed": qc_density_failures == 0 and match_violations == 0,
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


def write_report(
    config: dict[str, Any],
    preparation: dict[str, Any],
    generation: dict[str, Any],
    analysis: dict[str, Any],
) -> Path:
    output_dir = Path(config["outputs"]["directory"])
    report_path = output_dir / config["outputs"]["report_file"]
    primary = _find_primary(analysis["balanced_groups"] or analysis["all_groups"])
    lines = [
        "# QR URL Geometry Assay — run report",
        "",
        "## Outcome",
        "",
        f"- Matched URL quadruples: **{preparation['matched_pairs']:,}**",
        f"- QR transform rows: **{generation['rows']:,}**",
        f"- CPU workers: **{generation['workers']}**",
        f"- Generation time: **{generation['elapsed_seconds'] / 60:.2f} min**",
        f"- Quality control: **{'PASS' if analysis['quality_control']['passed'] else 'FAIL'}**",
        f"- Density-balanced payloads: **{analysis['balance']['selected_payloads']:,}**",
        "",
        "## Three geometric observables",
        "",
        "1. **Radial change:** `radial_mean`, `radial_std`, and covariance trace.",
        "2. **Translation:** normalized black-module centroid `(centroid_x, centroid_y)`.",
        "3. **Rotation/cyclicity:** `principal_angle_deg`, anisotropy, and the full 0°/90°/180°/270° orbit.",
        "",
        "## Controlled baseline (0°, normal polarity)",
        "",
        "| payload class | n | density | centroid radius | radial mean | orient cos(2θ) | orient sin(2θ) | anisotropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in sorted(primary, key=lambda item: item["payload_class"]):
        m = group["metrics"]
        lines.append(
            f"| {group['payload_class']} | {group['count']:,} | {m['density']['mean']:.6f} | "
            f"{m['centroid_radius']['mean']:.6f} | {m['radial_mean']['mean']:.6f} | "
            f"{m['orientation_cos2']['mean']:.6f} | {m['orientation_sin2']['mean']:.6f} | "
            f"{m['anisotropy']['mean']:.6f} |"
        )
    core_metrics = {
        "radial_mean",
        "centroid_radius",
        "orientation_cos2",
        "orientation_sin2",
    }
    core_effects = [
        effect for effect in analysis["paired_effects"] if effect["metric"] in core_metrics
    ]
    lines.extend(
        [
            "",
            "## Paired effects",
            "",
            "| contrast (left minus right) | metric | n | mean difference | 95% CI | Cohen dz | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|",
            *[
                f"| {effect['contrast']} | {effect['metric']} | {effect['n']:,} | "
                f"{effect['mean_difference']:.6g} | [{effect['ci95_low']:.6g}, {effect['ci95_high']:.6g}] | "
                f"{effect['cohen_dz']:.4f} | {effect['p_holm']:.3g} |"
                for effect in core_effects
            ],
            "",
            (
                "Paired effects use all byte-length/version/ECC/mask matches before the additional "
                "density-bin intersection; the controlled baseline table uses the density-balanced subset."
            ),
            "",
            "## Controls",
            "",
            (
                "All four payload classes are paired on UTF-8 byte length, QR version, error-correction "
                "level, byte-mode encoding, and mask. The balanced table additionally intersects "
                "black-module-density bins."
            ),
            "",
            f"- Density/inversion identity failures: {analysis['quality_control']['density_transform_failures']}",
            f"- Matched length/version/mask failures: {analysis['quality_control']['matched_control_failures']}",
            f"- Feature file SHA-256: `{analysis['features_sha256']}`",
            "",
            "The run uses URL strings only. It does not connect to onion services or download page content.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
