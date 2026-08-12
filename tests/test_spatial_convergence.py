import json
from pathlib import Path

from qr_assay.fileio import open_text
from qr_assay.spatial_convergence import run_convergence


def test_zero_sampling_uncertainty_cannot_fake_convergence_pass(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    payload_path = output / "payloads.jsonl.gz"
    payload = "/same/path/query?value=12345"

    rows = []
    for match_id in range(2):
        rows.extend(
            [
                {
                    "match_id": match_id,
                    "payload_class": "surface_natural",
                    "payload": payload,
                    "natural_host_sha256": f"surface-host-{match_id}",
                    "natural_source_sha256": "surface-source",
                },
                {
                    "match_id": match_id,
                    "payload_class": "surface_synthetic",
                    "payload": payload,
                    "natural_host_sha256": f"surface-host-{match_id}",
                    "natural_source_sha256": "surface-source",
                },
                {
                    "match_id": match_id,
                    "payload_class": "onion_natural",
                    "payload": payload,
                    "natural_host_sha256": f"onion-host-{match_id}",
                    "natural_source_sha256": "onion-source",
                },
                {
                    "match_id": match_id,
                    "payload_class": "onion_synthetic",
                    "payload": payload,
                    "natural_host_sha256": f"onion-host-{match_id}",
                    "natural_source_sha256": "onion-source",
                },
            ]
        )

    with open_text(payload_path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    config = {
        "seed": 17,
        "outputs": {"directory": str(output), "payloads_file": payload_path.name},
        "qr": {"error_correction": "M"},
        "analysis": {"spatial_null_mask": 0},
    }
    result = run_convergence(
        config,
        pilot_matches=2,
        k_values=(2, 4),
        batches=2,
    )

    assert result["status"] == "NO_K_CERTIFIED"
    assert result["selected_k"] is None
    assert result["accepted_by_k"] == {"2": False, "4": False}
    assert result["certifiable_by_k"] == {"2": False, "4": False}
    assert all(not cell["certifiable"] for cell in result["cells"])
    assert Path(result["output"]).exists()
