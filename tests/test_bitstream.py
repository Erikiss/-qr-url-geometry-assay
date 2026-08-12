import json
from pathlib import Path

import numpy as np

from qr_assay.bitstream import analyze_bitstream_baseline, bit_features
from qr_assay.fileio import open_text


def test_bit_features_separate_order_from_density():
    alternating = bit_features(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8))
    grouped = bit_features(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8))
    assert alternating[0] == grouped[0] == 0.5
    assert alternating[1] > grouped[1]
    assert alternating[2] < grouped[2]
    assert alternating[3] < grouped[3]


def test_identical_payload_null_is_zero_in_all_bitstream_baselines(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    payload_path = output / "payloads.jsonl.gz"
    payload = "/same/path/query?value=12345"
    rows = [
        {
            "match_id": 0,
            "payload_class": "surface_natural",
            "payload": payload,
            "natural_host_sha256": "surface-host",
            "natural_source_sha256": "surface-source",
        },
        {
            "match_id": 0,
            "payload_class": "surface_synthetic",
            "payload": payload,
            "natural_host_sha256": "surface-host",
            "natural_source_sha256": "surface-source",
        },
        {
            "match_id": 0,
            "payload_class": "onion_natural",
            "payload": payload,
            "natural_host_sha256": "onion-host",
            "natural_source_sha256": "onion-source",
        },
        {
            "match_id": 0,
            "payload_class": "onion_synthetic",
            "payload": payload,
            "natural_host_sha256": "onion-host",
            "natural_source_sha256": "onion-source",
        },
    ]
    with open_text(payload_path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    config = {
        "outputs": {"directory": str(output), "payloads_file": payload_path.name},
        "qr": {"error_correction": "M"},
        "analysis": {"bitstream_mask": 7},
    }
    result = analyze_bitstream_baseline(config)
    assert result["complete_matches"] == 1
    assert result["invalid_matches"] == 0
    for stream in ("raw_payload", "data_codeword", "rs_ecc"):
        for contrast in result["results"][stream].values():
            assert all(row["mean_difference"] == 0.0 for row in contrast["host_clustered"])
