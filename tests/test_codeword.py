import json
from pathlib import Path

from qr_assay.codeword import analyze_codeword_regions
from qr_assay.fileio import open_text


def test_identical_payload_null_is_zero_in_data_and_ecc_regions(tmp_path: Path):
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
            "natural_source_sha256": "surface-crawl",
        },
        {
            "match_id": 0,
            "payload_class": "surface_synthetic",
            "payload": payload,
            "natural_host_sha256": "surface-host",
            "natural_source_sha256": "surface-crawl",
        },
        {
            "match_id": 0,
            "payload_class": "onion_natural",
            "payload": payload,
            "natural_host_sha256": "onion-host",
            "natural_source_sha256": "onion-crawl",
        },
        {
            "match_id": 0,
            "payload_class": "onion_synthetic",
            "payload": payload,
            "natural_host_sha256": "onion-host",
            "natural_source_sha256": "onion-crawl",
        },
    ]
    with open_text(payload_path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    config = {
        "outputs": {"directory": str(output), "payloads_file": payload_path.name},
        "qr": {"error_correction": "M"},
        "analysis": {"codeword_mask": 5},
    }
    result = analyze_codeword_regions(config)
    assert result["complete_matches"] == 1
    assert result["invalid_matches"] == 0
    assert result["qrs_encoded"] == 4
    assert result["mask_removed_before_geometry"] is True

    for region in ("data_codeword", "rs_ecc"):
        for contrast in result["results"][region].values():
            host_rows = contrast["host_clustered"]
            assert all(row["mean_difference"] == 0.0 for row in host_rows)
