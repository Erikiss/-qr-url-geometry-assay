from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .cluster import CORE_METRICS, CONFIRMATORY_CONTRASTS, ClusterAccumulator
from .fileio import open_text
from .geometry import codeword_region_masks, geometry_features, make_qr, unmask_data_modules

REGION_FEATURE_KEYS = (
    "radial_mean",
    "centroid_radius",
    "orientation_cos2",
    "orientation_sin2",
)
REGIONS = ("data_codeword", "rs_ecc")


def _read_payloads(path: Path):
    with open_text(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _rename_region_metrics(rows: list[dict[str, Any]], region: str) -> list[dict[str, Any]]:
    renamed = []
    for row in rows:
        item = dict(row)
        index = CORE_METRICS.index(item["metric"])
        item["metric"] = f"{region}_{REGION_FEATURE_KEYS[index]}"
        renamed.append(item)
    return renamed


def analyze_codeword_regions(config: dict[str, Any]) -> dict[str, Any]:
    """Localize sequence-geometry effects to unmasked QR codeword regions.

    One explicitly chosen QR mask is generated and then deterministically removed
    before geometry is measured. This avoids treating the QR masking function as
    a source of payload geometry. Data-codeword modules and Reed-Solomon parity
    modules are analyzed separately.
    """
    output_dir = Path(config["outputs"]["directory"])
    payload_path = output_dir / config["outputs"]["payloads_file"]
    if not payload_path.exists():
        raise FileNotFoundError(f"Prepare payloads first: {payload_path}")
    ecc = str(config["qr"]["error_correction"]).upper()
    diagnostic_mask = int(config.get("analysis", {}).get("codeword_mask", 0))
    if diagnostic_mask < 0 or diagnostic_mask > 7:
        raise ValueError("analysis.codeword_mask must be in 0..7")

    accumulators = {
        region: {contrast: ClusterAccumulator() for contrast in CONFIRMATORY_CONTRASTS}
        for region in REGIONS
    }
    current_match_id: int | None = None
    records: dict[str, dict[str, Any]] = {}
    complete_matches = 0
    invalid_matches = 0
    qrs_encoded = 0

    def finish_match() -> None:
        nonlocal records, complete_matches, invalid_matches, qrs_encoded
        if current_match_id is None:
            return
        needed_classes = {name for pair in CONFIRMATORY_CONTRASTS.values() for name in pair}
        if set(records) != needed_classes:
            invalid_matches += 1
            records = {}
            return

        class_features: dict[str, dict[str, np.ndarray]] = {}
        versions = set()
        for cls, record in records.items():
            payload = record.get("payload")
            if payload is None:
                raise ValueError("Codeword diagnostics require outputs.store_payload_text=true")
            matrix, version = make_qr(
                payload,
                error_correction=ecc,
                mask=diagnostic_mask,
                border=0,
            )
            qrs_encoded += 1
            versions.add(version)
            unmasked = unmask_data_modules(matrix, version, diagnostic_mask)
            data_region, ecc_region, _ = codeword_region_masks(version, ecc)
            regions = {"data_codeword": data_region, "rs_ecc": ecc_region}
            class_features[cls] = {}
            for region_name, region_mask in regions.items():
                features = geometry_features(unmasked, region_mask)
                class_features[cls][region_name] = np.asarray(
                    [float(features[key]) for key in REGION_FEATURE_KEYS],
                    dtype=np.float64,
                )
        if len(versions) != 1:
            invalid_matches += 1
            records = {}
            return

        complete_matches += 1
        for contrast, (natural_class, synthetic_class) in CONFIRMATORY_CONTRASTS.items():
            natural = records[natural_class]
            synthetic = records[synthetic_class]
            if (
                natural.get("natural_host_sha256") != synthetic.get("natural_host_sha256")
                or natural.get("natural_source_sha256")
                != synthetic.get("natural_source_sha256")
            ):
                invalid_matches += 1
                continue
            for region_name in REGIONS:
                values = (
                    class_features[natural_class][region_name]
                    - class_features[synthetic_class][region_name]
                )
                accumulators[region_name][contrast].add(
                    values,
                    host=str(natural.get("natural_host_sha256")),
                    source=str(natural.get("natural_source_sha256")),
                )
        records = {}

    for record in _read_payloads(payload_path):
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
                "host_clustered": _rename_region_metrics(
                    accumulator.summarize("host"), region_name
                ),
                "source_clustered": _rename_region_metrics(
                    accumulator.summarize("source"), region_name
                ),
            }

    result = {
        "diagnostic_mask": diagnostic_mask,
        "mask_removed_before_geometry": True,
        "error_correction": ecc,
        "complete_matches": complete_matches,
        "invalid_matches": invalid_matches,
        "qrs_encoded": qrs_encoded,
        "regions": {
            "data_codeword": (
                "mode/count framing + payload bytes + terminator/alignment padding + QR pad "
                "codewords; not pure payload"
            ),
            "rs_ecc": "Reed-Solomon error-correction codeword modules",
        },
        "results": results,
    }
    output_path = output_dir / "codeword_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["output"] = str(output_path)
    return result
