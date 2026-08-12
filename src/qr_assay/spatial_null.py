from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cluster import CONFIRMATORY_CONTRASTS, CORE_METRICS, ClusterAccumulator
from .codeword import REGION_FEATURE_KEYS
from .fileio import open_text
from .geometry import codeword_region_masks, geometry_features, make_qr, unmask_data_modules

REGIONS = ("data_codeword", "rs_ecc")


def _vector(matrix: np.ndarray, region: np.ndarray) -> np.ndarray:
    features = geometry_features(matrix, region)
    return np.asarray([float(features[key]) for key in REGION_FEATURE_KEYS], dtype=np.float64)


def _rng_seed(base_seed: int, match_id: int, pair_group: str, region: str, mask: int) -> int:
    material = f"{base_seed}:{match_id}:{pair_group}:{region}:{mask}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def spatial_residual(
    matrix: np.ndarray,
    region: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed geometry minus random-placement expectation for exactly the same bits."""
    if permutations < 1:
        raise ValueError("spatial-null permutations must be positive")
    data = np.asarray(matrix, dtype=np.uint8)
    region_bool = np.asarray(region, dtype=bool)
    if data.shape != region_bool.shape:
        raise ValueError("spatial null region must match matrix shape")
    bits = data[region_bool].copy()
    observed = _vector(data, region_bool)
    rng = np.random.default_rng(seed)
    null_values = np.zeros((permutations, len(REGION_FEATURE_KEYS)), dtype=np.float64)
    permuted = data.copy()
    for index in range(permutations):
        permuted[region_bool] = rng.permutation(bits)
        # Density is exactly preserved by construction. The four reported
        # residual metrics focus on spatial placement rather than bit count.
        null_values[index] = _vector(permuted, region_bool)
    null_mean = null_values.mean(axis=0)
    null_sd = null_values.std(axis=0, ddof=1) if permutations > 1 else np.zeros_like(null_mean)
    return observed - null_mean, null_sd


def _rename(rows: list[dict[str, Any]], region: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item = dict(row)
        index = CORE_METRICS.index(str(item["metric"]))
        item["metric"] = f"{region}_placement_residual_{REGION_FEATURE_KEYS[index]}"
        result.append(item)
    return result


def analyze_spatial_null(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["outputs"]["directory"])
    payload_path = output_dir / config["outputs"]["payloads_file"]
    if not payload_path.exists():
        raise FileNotFoundError(f"Prepare payloads first: {payload_path}")

    analysis_cfg = config.get("analysis", {})
    permutations = int(analysis_cfg.get("spatial_null_permutations", 32))
    diagnostic_mask = int(analysis_cfg.get("spatial_null_mask", 0))
    if permutations < 1:
        raise ValueError("analysis.spatial_null_permutations must be positive")
    if diagnostic_mask < 0 or diagnostic_mask > 7:
        raise ValueError("analysis.spatial_null_mask must be in 0..7")
    ecc = str(config["qr"]["error_correction"]).upper()
    base_seed = int(config.get("seed", 267010))

    accumulators = {
        region: {contrast: ClusterAccumulator() for contrast in CONFIRMATORY_CONTRASTS}
        for region in REGIONS
    }
    null_sd_sums = {
        region: np.zeros(len(REGION_FEATURE_KEYS), dtype=np.float64) for region in REGIONS
    }
    null_sd_count = {region: 0 for region in REGIONS}
    current_match_id: int | None = None
    records: dict[str, dict[str, Any]] = {}
    complete_matches = 0
    invalid_matches = 0
    qrs_encoded = 0

    def finish_match() -> None:
        nonlocal records, complete_matches, invalid_matches, qrs_encoded
        if current_match_id is None:
            return
        needed = {name for pair in CONFIRMATORY_CONTRASTS.values() for name in pair}
        if set(records) != needed:
            invalid_matches += 1
            records = {}
            return

        residuals: dict[str, dict[str, np.ndarray]] = {}
        versions = set()
        for payload_class, record in records.items():
            payload = record.get("payload")
            if payload is None:
                raise ValueError("Spatial-null diagnostics require outputs.store_payload_text=true")
            matrix, version = make_qr(
                str(payload),
                error_correction=ecc,
                mask=diagnostic_mask,
                border=0,
            )
            qrs_encoded += 1
            versions.add(version)
            unmasked = unmask_data_modules(matrix, version, diagnostic_mask)
            data_region, ecc_region, _ = codeword_region_masks(version, ecc)
            region_masks = {"data_codeword": data_region, "rs_ecc": ecc_region}
            residuals[payload_class] = {}
            pair_group = "surface" if payload_class.startswith("surface_") else "onion"
            for region_name, region_mask in region_masks.items():
                residual, null_sd = spatial_residual(
                    unmasked,
                    region_mask,
                    permutations=permutations,
                    # Natural and synthetic members of one corpus/match use the
                    # same permutation stream. This common-random-number design
                    # removes avoidable Monte-Carlo noise from the paired effect.
                    seed=_rng_seed(
                        base_seed,
                        int(current_match_id),
                        pair_group,
                        region_name,
                        diagnostic_mask,
                    ),
                )
                residuals[payload_class][region_name] = residual
                null_sd_sums[region_name][:] += null_sd
                null_sd_count[region_name] += 1

        if len(versions) != 1:
            invalid_matches += 1
            records = {}
            return

        complete_matches += 1
        for contrast, (natural_class, synthetic_class) in CONFIRMATORY_CONTRASTS.items():
            natural = records[natural_class]
            synthetic = records[synthetic_class]
            natural_host = natural.get("natural_host_sha256")
            natural_source = natural.get("natural_source_sha256")
            if (
                not natural_host
                or not natural_source
                or natural_host != synthetic.get("natural_host_sha256")
                or natural_source != synthetic.get("natural_source_sha256")
            ):
                invalid_matches += 1
                continue
            for region_name in REGIONS:
                accumulators[region_name][contrast].add(
                    residuals[natural_class][region_name]
                    - residuals[synthetic_class][region_name],
                    host=str(natural_host),
                    source=str(natural_source),
                )
        records = {}

    with open_text(payload_path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            match_id = int(record["match_id"])
            if current_match_id is None:
                current_match_id = match_id
            elif match_id != current_match_id:
                finish_match()
                current_match_id = match_id
            records[record["payload_class"]] = record
    finish_match()

    results: dict[str, Any] = {}
    for region_name in REGIONS:
        results[region_name] = {}
        for contrast, accumulator in accumulators[region_name].items():
            results[region_name][contrast] = {
                "host_clustered": _rename(accumulator.summarize("host"), region_name),
                "source_clustered": _rename(accumulator.summarize("source"), region_name),
            }

    mean_null_sd = {
        region: {
            feature: float(null_sd_sums[region][index] / null_sd_count[region])
            if null_sd_count[region]
            else 0.0
            for index, feature in enumerate(REGION_FEATURE_KEYS)
        }
        for region in REGIONS
    }
    result = {
        "diagnostic_mask": diagnostic_mask,
        "mask_removed_before_geometry": True,
        "permutations_per_payload_region": permutations,
        "common_random_numbers_within_natural_synthetic_pair": True,
        "complete_matches": complete_matches,
        "invalid_matches": invalid_matches,
        "qrs_encoded": qrs_encoded,
        "null_preserves": [
            "exact region coordinates",
            "exact bit multiset and therefore exact one-density",
            "QR version and ECC level",
        ],
        "null_destroys": "the assignment of the existing unmasked bits to positions within each region",
        "mean_within_payload_null_sd": mean_null_sd,
        "interpretation": (
            "The reported effect is a natural-vs-structural-null contrast of observed-minus-random-"
            "placement geometry. A surviving effect is evidence of placement-specific 2D structure "
            "beyond exact bit composition, not by itself evidence of semantic or emergent geometry."
        ),
        "results": results,
    }
    output_path = output_dir / "spatial_null_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["output"] = str(output_path)
    return result
