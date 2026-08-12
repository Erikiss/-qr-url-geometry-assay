import json
from pathlib import Path

import numpy as np

from qr_assay.did import SideAccumulator, analyze_difference_in_differences
from qr_assay.fileio import open_text


def test_unique_clusters_reduce_to_ordinary_mean_se():
    accumulator = SideAccumulator()
    values = [-1.0, -0.5, 0.5, 1.0]
    for index, value in enumerate(values):
        accumulator.add(np.full(4, value, dtype=np.float64), cluster=f"host-{index}")

    expected_var = np.var(values, ddof=1) / len(values)
    assert all(np.isclose(value, expected_var) for value in accumulator.variance())
    assert len(accumulator.clusters) == 4


def test_repeated_clusters_remain_few_clusters():
    accumulator = SideAccumulator()
    for index in range(200):
        value = 1.0 if index % 2 == 0 else -1.0
        accumulator.add(np.full(4, value, dtype=np.float64), cluster=f"host-{index % 2}")
    assert accumulator.n == 200
    assert len(accumulator.clusters) == 2
    assert accumulator.effective_clusters() == 2.0


def test_difference_in_differences_averages_masks_before_contrast(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    features_path = output / "features.jsonl.gz"
    metrics = (
        "data_radial_mean",
        "data_centroid_radius",
        "data_orientation_cos2",
        "data_orientation_sin2",
    )

    rows = []
    for match_id in range(2):
        classes = {
            "surface_natural": 1.0,
            "surface_synthetic": 0.0,
            "onion_natural": 3.0,
            "onion_synthetic": 0.0,
        }
        for payload_class, base in classes.items():
            corpus = "surface" if payload_class.startswith("surface_") else "onion"
            for mask in (0, 1):
                row = {
                    "match_id": match_id,
                    "payload_class": payload_class,
                    "mask": mask,
                    "rotation": 0,
                    "reflection": "none",
                    "scale": 1,
                    "inverted": False,
                    "byte_length": 50,
                    "qr_version": 4,
                    "error_correction": "M",
                    "natural_host_sha256": f"{corpus}-host-{match_id}",
                    "natural_source_sha256": f"{corpus}-source-{match_id}",
                }
                for metric in metrics:
                    row[metric] = base + 0.25 * mask
                rows.append(row)

    with open_text(features_path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    config = {
        "outputs": {"directory": str(output), "features_file": features_path.name},
        "qr": {"masks": [0, 1]},
    }
    result = analyze_difference_in_differences(config)

    assert result["complete_matches"] == 2
    assert result["invalid_matches"] == 0
    assert result["family_size"] == 4
    rows = result["host_independent_cluster_difference"]
    assert len(rows) == 4
    for row in rows:
        assert np.isclose(row["mean_difference_in_differences"], 2.0)
        assert row["surface_cluster_count"] == 2
        assert row["onion_cluster_count"] == 2
        assert row["eligible_for_confirmatory_claim"] is False
        assert row["strict_confirmatory_pass"] is False
    assert Path(result["output"]).exists()
