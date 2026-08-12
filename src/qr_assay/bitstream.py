from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cluster import CONFIRMATORY_CONTRASTS, CORE_METRICS, ClusterAccumulator
from .fileio import open_text
from .geometry import codeword_region_masks, data_module_mask, make_qr, unmask_data_modules

BIT_FEATURE_KEYS = (
    "one_density",
    "transition_rate",
    "mean_run_length",
    "max_run_fraction",
)
BITSTREAMS = ("raw_payload", "data_codeword", "rs_ecc")


def bit_features(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n = int(values.size)
    if n == 0:
        return np.zeros(len(BIT_FEATURE_KEYS), dtype=np.float64)
    density = float(values.mean())
    transitions = float(np.mean(values[1:] != values[:-1])) if n > 1 else 0.0

    if n == 1:
        runs = np.array([1], dtype=np.int64)
    else:
        change_points = np.flatnonzero(values[1:] != values[:-1]) + 1
        boundaries = np.concatenate(([0], change_points, [n]))
        runs = np.diff(boundaries)
    mean_run = float(runs.mean())
    max_run_fraction = float(runs.max() / n)
    return np.asarray(
        [density, transitions, mean_run, max_run_fraction],
        dtype=np.float64,
    )


def _mapped_bits_in_order(matrix: np.ndarray, version: int) -> np.ndarray:
    """Replay python-qrcode's map_data zig-zag and return free modules in bit order."""
    data = np.asarray(matrix, dtype=np.uint8)
    free = data_module_mask(int(version)).astype(bool)
    if data.shape != free.shape:
        raise ValueError("mapped bitstream requires a border-free QR matrix")

    modules_count = free.shape[0]
    inc = -1
    row = modules_count - 1
    values: list[int] = []
    for col in range(modules_count - 1, 0, -2):
        if col <= 6:
            col -= 1
        col_range = (col, col - 1)
        while True:
            for current_col in col_range:
                if free[row, current_col]:
                    values.append(int(data[row, current_col]))
            row += inc
            if row < 0 or modules_count <= row:
                row -= inc
                inc = -inc
                break
    result = np.asarray(values, dtype=np.uint8)
    if result.size != int(free.sum()):
        raise AssertionError("Mapped bitstream traversal did not consume all free modules")
    return result


def _payload_bitstreams(payload: str, *, ecc: str, mask: int) -> dict[str, np.ndarray]:
    raw_bytes = np.frombuffer(payload.encode("utf-8"), dtype=np.uint8)
    raw_bits = np.unpackbits(raw_bytes)

    matrix, version = make_qr(payload, error_correction=ecc, mask=mask, border=0)
    unmasked = unmask_data_modules(matrix, version, mask)
    mapped = _mapped_bits_in_order(unmasked, version)
    data_region, ecc_region, remainder_region = codeword_region_masks(version, ecc)
    data_count = int(data_region.sum())
    ecc_count = int(ecc_region.sum())
    remainder_count = int(remainder_region.sum())
    if mapped.size != data_count + ecc_count + remainder_count:
        raise AssertionError("Bitstream split does not match QR region partition")
    return {
        "raw_payload": raw_bits,
        "data_codeword": mapped[:data_count],
        "rs_ecc": mapped[data_count : data_count + ecc_count],
    }


def _renamed_summary(rows: list[dict[str, Any]], stream: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item = dict(row)
        index = CORE_METRICS.index(str(item["metric"]))
        item["metric"] = f"{stream}_{BIT_FEATURE_KEYS[index]}"
        result.append(item)
    return result


def analyze_bitstream_baseline(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["outputs"]["directory"])
    payload_path = output_dir / config["outputs"]["payloads_file"]
    if not payload_path.exists():
        raise FileNotFoundError(f"Prepare payloads first: {payload_path}")
    ecc = str(config["qr"]["error_correction"]).upper()
    mask = int(config.get("analysis", {}).get("bitstream_mask", 0))
    if mask < 0 or mask > 7:
        raise ValueError("analysis.bitstream_mask must be in 0..7")

    accumulators = {
        stream: {contrast: ClusterAccumulator() for contrast in CONFIRMATORY_CONTRASTS}
        for stream in BITSTREAMS
    }
    current_match_id: int | None = None
    records: dict[str, dict[str, Any]] = {}
    complete_matches = 0
    invalid_matches = 0

    def finish_match() -> None:
        nonlocal records, complete_matches, invalid_matches
        if current_match_id is None:
            return
        needed = {name for pair in CONFIRMATORY_CONTRASTS.values() for name in pair}
        if set(records) != needed:
            invalid_matches += 1
            records = {}
            return

        features: dict[str, dict[str, np.ndarray]] = {}
        for cls, record in records.items():
            payload = record.get("payload")
            if payload is None:
                raise ValueError("Bitstream diagnostics require outputs.store_payload_text=true")
            streams = _payload_bitstreams(str(payload), ecc=ecc, mask=mask)
            features[cls] = {name: bit_features(bits) for name, bits in streams.items()}

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
            for stream in BITSTREAMS:
                accumulators[stream][contrast].add(
                    features[natural_class][stream] - features[synthetic_class][stream],
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
    for stream in BITSTREAMS:
        results[stream] = {}
        for contrast, accumulator in accumulators[stream].items():
            results[stream][contrast] = {
                "host_clustered": _renamed_summary(accumulator.summarize("host"), stream),
                "source_clustered": _renamed_summary(accumulator.summarize("source"), stream),
            }

    result = {
        "diagnostic_mask": mask,
        "qr_mask_removed_before_mapped_bitstream_extraction": True,
        "error_correction": ecc,
        "complete_matches": complete_matches,
        "invalid_matches": invalid_matches,
        "features": list(BIT_FEATURE_KEYS),
        "streams": {
            "raw_payload": "UTF-8 payload bytes before QR framing",
            "data_codeword": (
                "QR data-codeword bits in encoder mapping order before spatial interpretation; "
                "includes framing and padding"
            ),
            "rs_ecc": "Reed-Solomon parity bits before spatial interpretation",
        },
        "interpretation": (
            "A difference already present here is not evidence that 2D QR placement created the "
            "underlying distinction. These are negative-control baselines, not a sufficient test "
            "that a surviving 2D statistic is emergent."
        ),
        "results": results,
    }
    output_path = output_dir / "bitstream_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["output"] = str(output_path)
    return result
