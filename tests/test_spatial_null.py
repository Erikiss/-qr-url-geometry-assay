import json
from pathlib import Path

import numpy as np

from qr_assay.fileio import open_text
from qr_assay.spatial_null import analyze_spatial_null, spatial_residual


def test_degenerate_exact_bit_multiset_has_zero_spatial_residual():
    matrix = np.ones((5, 5), dtype=np.uint8)
    region = np.ones_like(matrix, dtype=np.uint8)
    residual, null_sd = spatial_residual(matrix, region, permutations=8, seed=1)
    assert np.allclose(residual, 0.0)
    assert np.allclose(null_sd, 0.0)


def test_identical_payload_null_has_exact_zero_placement_contrast(tmp_path: Path):
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
        "seed": 17,
        "outputs": {"directory": str(output), "payloads_file": payload_path.name},
        "qr": {"error_correction": "M"},
        "analysis": {"spatial_null_mask": 3, "spatial_null_permutations": 4},
    }
    result = analyze_spatial_null(config)
    assert result["complete_matches"] == 1
    assert result["invalid_matches"] == 0
    assert result["permutations_per_payload_region"] == 4
    assert result["common_random_numbers_within_natural_synthetic_pair"] is True
    for region in ("data_codeword", "rs_ecc"):
        for contrast in result["results"][region].values():
            assert all(row["n_matches"] == 1 for row in contrast["host_clustered"])
            assert all(row["mean_difference"] == 0.0 for row in contrast["host_clustered"])
