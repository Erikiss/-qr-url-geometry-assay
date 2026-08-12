from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .cluster import CONFIRMATORY_CONTRASTS, CORE_METRICS, ClusterAccumulator
from .codeword import REGION_FEATURE_KEYS
from .config import load_config
from .fileio import open_text
from .geometry import codeword_region_masks, make_qr, unmask_data_modules
from .spatial_null import REGIONS, spatial_residual


def _batch_seed(
    base_seed: int,
    batch: int,
    match_id: int,
    pair_group: str,
    region: str,
    mask: int,
) -> int:
    material = f"{base_seed}:mc-batch:{batch}:{match_id}:{pair_group}:{region}:{mask}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _load_pilot_matches(path: Path, limit: int) -> list[dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    with open_text(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            match_id = int(row["match_id"])
            if match_id >= limit:
                continue
            grouped[match_id][row["payload_class"]] = row
    needed = {name for pair in CONFIRMATORY_CONTRASTS.values() for name in pair}
    result = []
    for match_id in sorted(grouped):
        records = grouped[match_id]
        if set(records) != needed:
            raise ValueError(f"Pilot match {match_id} does not contain all four payload classes")
        result.append(records)
    if len(result) < limit:
        raise ValueError(f"Requested {limit} pilot matches, found only {len(result)} complete matches")
    return result


def _placement_residuals_for_batch(
    records: dict[str, dict[str, Any]],
    *,
    base_seed: int,
    batch: int,
    match_id: int,
    permutations: int,
    mask: int,
    ecc: str,
) -> dict[str, dict[str, np.ndarray]]:
    residuals: dict[str, dict[str, np.ndarray]] = {}
    versions = set()
    for payload_class, record in records.items():
        payload = record.get("payload")
        if payload is None:
            raise ValueError("Convergence assay requires outputs.store_payload_text=true")
        matrix, version = make_qr(str(payload), error_correction=ecc, mask=mask, border=0)
        versions.add(version)
        unmasked = unmask_data_modules(matrix, version, mask)
        data_region, ecc_region, _ = codeword_region_masks(version, ecc)
        region_masks = {"data_codeword": data_region, "rs_ecc": ecc_region}
        pair_group = "surface" if payload_class.startswith("surface_") else "onion"
        residuals[payload_class] = {}
        for region_name, region_mask in region_masks.items():
            residual, _ = spatial_residual(
                unmasked,
                region_mask,
                permutations=permutations,
                seed=_batch_seed(
                    base_seed,
                    batch,
                    match_id,
                    pair_group,
                    region_name,
                    mask,
                ),
            )
            residuals[payload_class][region_name] = residual
    if len(versions) != 1:
        raise ValueError(f"Pilot match {match_id} produced mismatched QR versions")
    return residuals


def _metric_name(index: int, region: str) -> str:
    return f"{region}_placement_residual_{REGION_FEATURE_KEYS[index]}"


def _effect_key(region: str, contrast: str, metric_index: int) -> tuple[str, str, int]:
    return region, contrast, metric_index


def run_convergence(
    config: dict[str, Any],
    *,
    pilot_matches: int = 250,
    k_values: tuple[int, ...] = (16, 32, 64, 128),
    batches: int = 8,
    mc_fraction_of_sampling_se: float = 0.10,
    drift_fraction_of_sampling_se: float = 0.25,
) -> dict[str, Any]:
    """Choose a spatial-null permutation count by a locked precision rule.

    For each K and each independent Monte-Carlo batch, the same pilot matches are
    evaluated. Natural/synthetic members still share common random placements
    within a batch. At the largest K, host-cluster CR1 SE estimates observational
    sampling uncertainty. A candidate K is accepted only if, for every estimable
    region/contrast/metric:

    1. batch-to-batch Monte-Carlo SD at K <= `mc_fraction_of_sampling_se` * host CR1 SE;
    2. absolute drift of the batch-mean estimate from the previous K <=
       `drift_fraction_of_sampling_se` * host CR1 SE.

    The smallest K satisfying both rules is selected. Metrics with fewer than two
    host clusters or exactly zero sampling SE are reported but cannot certify K.
    """
    if pilot_matches < 2:
        raise ValueError("pilot_matches must be at least 2")
    if batches < 2:
        raise ValueError("batches must be at least 2")
    k_values = tuple(sorted({int(value) for value in k_values}))
    if not k_values or k_values[0] < 1:
        raise ValueError("k_values must contain positive integers")
    if mc_fraction_of_sampling_se <= 0 or drift_fraction_of_sampling_se <= 0:
        raise ValueError("convergence fractions must be positive")

    output_dir = Path(config["outputs"]["directory"])
    payload_path = output_dir / config["outputs"]["payloads_file"]
    if not payload_path.exists():
        raise FileNotFoundError(f"Prepare payloads first: {payload_path}")
    records_by_match = _load_pilot_matches(payload_path, pilot_matches)
    mask = int(config.get("analysis", {}).get("spatial_null_mask", 0))
    ecc = str(config["qr"]["error_correction"]).upper()
    base_seed = int(config.get("seed", 267010))

    # batch_effects[K][batch][key] -> list of match-level effects
    batch_effects: dict[int, dict[int, dict[tuple[str, str, int], list[float]]]] = {
        k: {batch: defaultdict(list) for batch in range(batches)} for k in k_values
    }
    # Host-cluster reference uses batch 0 at the maximum K. MC variation is
    # separately measured across batches; using one batch avoids averaging away
    # the very randomness we are trying to quantify.
    reference_accumulators = {
        region: {contrast: ClusterAccumulator() for contrast in CONFIRMATORY_CONTRASTS}
        for region in REGIONS
    }
    max_k = k_values[-1]

    for match_id, records in enumerate(records_by_match):
        for batch in range(batches):
            for k in k_values:
                residuals = _placement_residuals_for_batch(
                    records,
                    base_seed=base_seed,
                    batch=batch,
                    match_id=match_id,
                    permutations=k,
                    mask=mask,
                    ecc=ecc,
                )
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
                        raise ValueError(
                            f"Pilot match {match_id} has inconsistent natural cluster metadata"
                        )
                    for region in REGIONS:
                        vector = residuals[natural_class][region] - residuals[synthetic_class][region]
                        for metric_index, value in enumerate(vector):
                            batch_effects[k][batch][
                                _effect_key(region, contrast, metric_index)
                            ].append(float(value))
                        if batch == 0 and k == max_k:
                            reference_accumulators[region][contrast].add(
                                vector,
                                host=str(natural_host),
                                source=str(natural_source),
                            )

    reference_se: dict[tuple[str, str, int], float | None] = {}
    reference_cluster_count: dict[tuple[str, str, int], int] = {}
    for region in REGIONS:
        for contrast, accumulator in reference_accumulators[region].items():
            rows = accumulator.summarize("host")
            for metric_index, row in enumerate(rows):
                key = _effect_key(region, contrast, metric_index)
                reference_se[key] = row["cr1_standard_error"]
                reference_cluster_count[key] = int(row["cluster_count"])

    summaries: list[dict[str, Any]] = []
    previous_mean: dict[tuple[str, str, int], float] = {}
    accepted_by_k = {k: True for k in k_values}
    certifiable_by_k = {k: True for k in k_values}

    for k in k_values:
        all_keys = sorted(batch_effects[k][0])
        for key in all_keys:
            region, contrast, metric_index = key
            batch_means = np.asarray(
                [np.mean(batch_effects[k][batch][key]) for batch in range(batches)],
                dtype=np.float64,
            )
            mean_effect = float(batch_means.mean())
            mc_sd = float(batch_means.std(ddof=1))
            sampling_se = reference_se[key]
            cluster_count = reference_cluster_count[key]
            drift = (
                abs(mean_effect - previous_mean[key]) if key in previous_mean else None
            )
            certifiable = (
                sampling_se is not None
                and math.isfinite(float(sampling_se))
                and float(sampling_se) > 0
                and cluster_count >= 2
            )
            mc_limit = (
                mc_fraction_of_sampling_se * float(sampling_se) if certifiable else None
            )
            drift_limit = (
                drift_fraction_of_sampling_se * float(sampling_se) if certifiable else None
            )
            mc_ok = bool(certifiable and mc_sd <= float(mc_limit))
            # The first K has no previous-K drift comparison and therefore cannot
            # be selected even if MC SD is already small.
            drift_ok = bool(
                certifiable
                and drift is not None
                and drift <= float(drift_limit)
            )
            if not certifiable:
                certifiable_by_k[k] = False
            if not (mc_ok and drift_ok):
                accepted_by_k[k] = False
            summaries.append(
                {
                    "k": k,
                    "region": region,
                    "contrast": contrast,
                    "metric": _metric_name(metric_index, region),
                    "pilot_matches": pilot_matches,
                    "host_cluster_count_reference": cluster_count,
                    "batch_count": batches,
                    "batch_mean_effect": mean_effect,
                    "monte_carlo_sd_of_effect": mc_sd,
                    "host_cr1_sampling_se_reference": sampling_se,
                    "drift_from_previous_k": drift,
                    "mc_limit": mc_limit,
                    "drift_limit": drift_limit,
                    "certifiable": certifiable,
                    "mc_ok": mc_ok,
                    "drift_ok": drift_ok,
                }
            )
            previous_mean[key] = mean_effect

    selected_k = next(
        (
            k
            for k in k_values[1:]
            if certifiable_by_k[k] and accepted_by_k[k]
        ),
        None,
    )
    result = {
        "pilot_matches": pilot_matches,
        "k_values": list(k_values),
        "batches": batches,
        "diagnostic_mask": mask,
        "error_correction": ecc,
        "criterion": {
            "mc_sd_le_fraction_of_host_cr1_se": mc_fraction_of_sampling_se,
            "drift_le_fraction_of_host_cr1_se": drift_fraction_of_sampling_se,
            "all_estimable_region_contrast_metric_cells_must_pass": True,
            "first_k_cannot_be_selected_without_a_drift_comparison": True,
        },
        "accepted_by_k": {str(k): accepted_by_k[k] for k in k_values},
        "certifiable_by_k": {str(k): certifiable_by_k[k] for k in k_values},
        "selected_k": selected_k,
        "status": "PASS" if selected_k is not None else "NO_K_CERTIFIED",
        "interpretation": (
            "This pilot chooses computational effort, not scientific significance. Failure to "
            "certify K means increase the K ladder and/or pilot host diversity; it is not evidence "
            "for or against a spatial-placement effect."
        ),
        "cells": summaries,
    }
    output_path = output_dir / "spatial_null_convergence.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["output"] = str(output_path)
    return result


def _parse_k_values(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("K list must contain at least one integer")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the exact-bit spatial-null Monte-Carlo permutation count."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-matches", type=int, default=250)
    parser.add_argument("--k", type=_parse_k_values, default=(16, 32, 64, 128))
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--mc-fraction", type=float, default=0.10)
    parser.add_argument("--drift-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    # This module reads the already prepared payload artifact. It never re-samples
    # the corpus while changing K.
    result = run_convergence(
        copy.deepcopy(config),
        pilot_matches=args.pilot_matches,
        k_values=args.k,
        batches=args.batches,
        mc_fraction_of_sampling_se=args.mc_fraction,
        drift_fraction_of_sampling_se=args.drift_fraction,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
