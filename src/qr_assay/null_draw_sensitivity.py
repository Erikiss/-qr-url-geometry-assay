from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .cluster import CORE_METRICS, ClusterAccumulator
from .config import load_config
from .did import IndependentDifference
from .fileio import open_text
from .geometry import data_module_mask, geometry_features, make_qr
from .synthetic import grammar_matched

CORPORA = ("surface", "onion")


def _read_natural_matches(path: Path, limit: int) -> list[dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    with open_text(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("synthetic")):
                continue
            match_id = int(row["match_id"])
            grouped[match_id][str(row["grammar"])] = row
    result = []
    for match_id in sorted(grouped):
        records = grouped[match_id]
        if set(records) != set(CORPORA):
            raise ValueError(f"Natural pilot match {match_id} does not contain both corpora")
        result.append(records)
        if len(result) >= limit:
            break
    if len(result) < limit:
        raise ValueError(f"Requested {limit} pilot matches, found only {len(result)}")
    return result


def _payload_core_vector(payload: str, config: dict[str, Any]) -> tuple[np.ndarray, int]:
    ecc = str(config["qr"]["error_correction"]).upper()
    vectors = []
    versions = set()
    for mask in config["qr"]["masks"]:
        matrix, version = make_qr(
            payload,
            error_correction=ecc,
            mask=int(mask),
            border=0,
        )
        versions.add(version)
        features = geometry_features(matrix, data_module_mask(version))
        vectors.append(
            np.asarray(
                [
                    float(features["radial_mean"]),
                    float(features["centroid_radius"]),
                    float(features["orientation_cos2"]),
                    float(features["orientation_sin2"]),
                ],
                dtype=np.float64,
            )
        )
    if len(versions) != 1:
        raise AssertionError("One payload changed QR version across mask patterns")
    return np.mean(np.stack(vectors, axis=0), axis=0), next(iter(versions))


def _derive_seeds(base_seed: int, draws: int) -> tuple[int, ...]:
    if draws < 2:
        raise ValueError("null draw sensitivity requires at least two seeds")
    seeds = []
    for index in range(draws):
        material = f"{base_seed}:structural-null-draw:{index}".encode("utf-8")
        seeds.append(int.from_bytes(hashlib.sha256(material).digest()[:4], "big"))
    return tuple(seeds)


def _summary_rows(
    seed_means: dict[str, list[np.ndarray]],
    reference_rows: dict[str, list[dict[str, Any]]],
    *,
    mc_fraction: float,
    drift_fraction: float,
    zero_tolerance: float,
) -> list[dict[str, Any]]:
    rows = []
    for family, values in seed_means.items():
        matrix = np.stack(values, axis=0)
        mean_across_draws = matrix.mean(axis=0)
        draw_sd = matrix.std(axis=0, ddof=1)
        max_drift = np.max(np.abs(matrix - mean_across_draws), axis=0)
        for metric_index, metric in enumerate(CORE_METRICS):
            reference = reference_rows[family][metric_index]
            if family == "did":
                sampling_se = reference["independent_cluster_standard_error"]
                reference_mean = reference["mean_difference_in_differences"]
                clusters_ok = bool(reference["cluster_counts_ge_20"])
            else:
                sampling_se = reference["cr1_standard_error"]
                reference_mean = reference["mean_difference"]
                clusters_ok = bool(reference["cluster_count_ge_20"])
            se = float(sampling_se) if sampling_se is not None else None
            ordinary = bool(se is not None and math.isfinite(se) and se > zero_tolerance)
            stable_zero = bool(
                se is not None
                and abs(se) <= zero_tolerance
                and abs(float(reference_mean)) <= zero_tolerance
                and float(draw_sd[metric_index]) <= zero_tolerance
                and float(max_drift[metric_index]) <= zero_tolerance
            )
            if ordinary:
                sd_limit = mc_fraction * se
                drift_limit = drift_fraction * se
                sd_ok = float(draw_sd[metric_index]) <= sd_limit
                drift_ok = float(max_drift[metric_index]) <= drift_limit
            elif stable_zero:
                sd_limit = 0.0
                drift_limit = 0.0
                sd_ok = True
                drift_ok = True
            else:
                sd_limit = None
                drift_limit = None
                sd_ok = False
                drift_ok = False
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "null_draw_mean_effect": float(mean_across_draws[metric_index]),
                    "null_draw_sd_of_effect": float(draw_sd[metric_index]),
                    "max_abs_drift_from_draw_mean": float(max_drift[metric_index]),
                    "reference_sampling_se": se,
                    "reference_mean_effect": float(reference_mean),
                    "reference_has_20_clusters": clusters_ok,
                    "ordinary_estimable": ordinary,
                    "degenerate_stable_zero": stable_zero,
                    "sd_limit": sd_limit,
                    "drift_limit": drift_limit,
                    "sd_ok": sd_ok,
                    "drift_ok": drift_ok,
                    "cell_pass": bool((ordinary or stable_zero) and sd_ok and drift_ok),
                }
            )
    return rows


def run_null_draw_sensitivity(
    config: dict[str, Any],
    *,
    pilot_matches: int = 250,
    draws: int = 8,
    mc_fraction_of_sampling_se: float = 0.10,
    drift_fraction_of_sampling_se: float = 0.25,
    zero_tolerance: float = 1e-14,
) -> dict[str, Any]:
    """Measure how much one random structural-null draw moves the scientific estimates.

    The natural pilot sample is read from the already prepared payload artifact and
    never re-sampled. Only the structural-null RNG seed changes. Reference sampling
    uncertainty is computed after averaging all tested null draws *within each
    observational unit*, so null-draw replications never manufacture a larger N.
    """
    if pilot_matches < 2:
        raise ValueError("pilot_matches must be at least 2")
    if mc_fraction_of_sampling_se <= 0 or drift_fraction_of_sampling_se <= 0:
        raise ValueError("sensitivity fractions must be positive")
    output_dir = Path(config["outputs"]["directory"])
    payload_path = output_dir / config["outputs"]["payloads_file"]
    if not payload_path.exists():
        raise FileNotFoundError(f"Prepare payloads first: {payload_path}")
    pilot = _read_natural_matches(payload_path, pilot_matches)
    null_mode = str(config["sampling"].get("synthetic_mode", "grammar_random"))
    base_seed = int(config.get("seed", 267010))
    seeds = _derive_seeds(base_seed, draws)

    natural_vectors: dict[tuple[int, str], np.ndarray] = {}
    natural_versions: dict[tuple[int, str], int] = {}
    for match_id, records in enumerate(pilot):
        for corpus in CORPORA:
            payload = str(records[corpus]["payload"])
            vector, version = _payload_core_vector(payload, config)
            natural_vectors[(match_id, corpus)] = vector
            natural_versions[(match_id, corpus)] = version

    # effects[seed_index][match_id][corpus]
    effects: list[list[dict[str, np.ndarray]]] = []
    for null_seed in seeds:
        seed_effects = []
        for match_id, records in enumerate(pilot):
            match_effects: dict[str, np.ndarray] = {}
            for corpus in CORPORA:
                natural_payload = str(records[corpus]["payload"])
                synthetic = grammar_matched(
                    natural_payload,
                    seed=null_seed,
                    match_id=match_id,
                    corpus=corpus,
                    mode=null_mode,
                )
                synthetic_vector, synthetic_version = _payload_core_vector(synthetic, config)
                if synthetic_version != natural_versions[(match_id, corpus)]:
                    raise AssertionError("Structural null changed QR version despite byte matching")
                match_effects[corpus] = natural_vectors[(match_id, corpus)] - synthetic_vector
            seed_effects.append(match_effects)
        effects.append(seed_effects)

    seed_means: dict[str, list[np.ndarray]] = {"surface": [], "onion": [], "did": []}
    for seed_effects in effects:
        surface_matrix = np.stack([row["surface"] for row in seed_effects], axis=0)
        onion_matrix = np.stack([row["onion"] for row in seed_effects], axis=0)
        seed_means["surface"].append(surface_matrix.mean(axis=0))
        seed_means["onion"].append(onion_matrix.mean(axis=0))
        seed_means["did"].append(onion_matrix.mean(axis=0) - surface_matrix.mean(axis=0))

    # Average null-draw effects inside each observational unit, then estimate the
    # ordinary host-cluster sampling uncertainty on those unit-level averages.
    surface_reference = ClusterAccumulator()
    onion_reference = ClusterAccumulator()
    did_reference = IndependentDifference()
    for match_id, records in enumerate(pilot):
        surface_mean = np.mean(
            np.stack([seed_effects[match_id]["surface"] for seed_effects in effects], axis=0),
            axis=0,
        )
        onion_mean = np.mean(
            np.stack([seed_effects[match_id]["onion"] for seed_effects in effects], axis=0),
            axis=0,
        )
        surface_host = str(records["surface"].get("natural_host_sha256"))
        onion_host = str(records["onion"].get("natural_host_sha256"))
        surface_source = str(records["surface"].get("natural_source_sha256"))
        onion_source = str(records["onion"].get("natural_source_sha256"))
        if None in {
            records["surface"].get("natural_host_sha256"),
            records["onion"].get("natural_host_sha256"),
            records["surface"].get("natural_source_sha256"),
            records["onion"].get("natural_source_sha256"),
        }:
            raise ValueError(f"Pilot match {match_id} lacks natural cluster metadata")
        surface_reference.add(surface_mean, host=surface_host, source=surface_source)
        onion_reference.add(onion_mean, host=onion_host, source=onion_source)
        did_reference.add_surface(surface_mean, cluster=surface_host)
        did_reference.add_onion(onion_mean, cluster=onion_host)

    reference_rows = {
        "surface": surface_reference.summarize("host"),
        "onion": onion_reference.summarize("host"),
        "did": did_reference.summarize(
            surface_level="surface_host",
            onion_level="onion_host",
        ),
    }
    cells = _summary_rows(
        seed_means,
        reference_rows,
        mc_fraction=mc_fraction_of_sampling_se,
        drift_fraction=drift_fraction_of_sampling_se,
        zero_tolerance=zero_tolerance,
    )
    status = "PASS" if all(cell["cell_pass"] for cell in cells) else "UNSTABLE_SINGLE_DRAW"
    result = {
        "pilot_matches": pilot_matches,
        "null_mode": null_mode,
        "draw_count": draws,
        "null_seeds": list(seeds),
        "qr_masks_averaged_per_payload": [int(mask) for mask in config["qr"]["masks"]],
        "criterion": {
            "null_draw_sd_le_fraction_of_host_sampling_se": mc_fraction_of_sampling_se,
            "max_draw_drift_le_fraction_of_host_sampling_se": drift_fraction_of_sampling_se,
            "zero_tolerance": zero_tolerance,
            "all_12_cells_must_pass": True,
        },
        "status": status,
        "single_draw_acceptable_on_pilot": status == "PASS",
        "interpretation": (
            "This is a numerical/randomization-stability pilot, not a significance test. If it "
            "returns UNSTABLE_SINGLE_DRAW, the production design must average multiple structural-"
            "null draws per natural payload (or otherwise integrate over the declared null) rather "
            "than selecting a favorable seed. The natural pilot sample is fixed across draws."
        ),
        "cells": cells,
    }
    output_path = output_dir / "null_draw_sensitivity.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["output"] = str(output_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test sensitivity of stage-1 estimates to the structural-null RNG draw."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-matches", type=int, default=250)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--mc-fraction", type=float, default=0.10)
    parser.add_argument("--drift-fraction", type=float, default=0.25)
    parser.add_argument("--zero-tolerance", type=float, default=1e-14)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = run_null_draw_sensitivity(
        config,
        pilot_matches=args.pilot_matches,
        draws=args.draws,
        mc_fraction_of_sampling_se=args.mc_fraction,
        drift_fraction_of_sampling_se=args.drift_fraction,
        zero_tolerance=args.zero_tolerance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
