import json
import math
from pathlib import Path

import numpy as np

from qr_assay.did import TwoWayAccumulator, analyze_difference_in_differences
from qr_assay.fileio import open_text


def test_unique_two_way_clusters_reduce_to_ordinary_mean_se():
    accumulator = TwoWayAccumulator()
    values = [-1.0, -0.5, 0.5, 1.0]
    for index, value in enumerate(values):
        accumulator.add(
            np.full(4, value, dtype=np.float64),
            cluster_a=f"surface-{index}",
            cluster_b=f"onion-{index}",
        )

    expected = np.std(values, ddof=1) / math.sqrt(len(values))
    rows = accumulator.summarize(level_a="surface_host", level_b="onion_host")
    assert all(row["cluster_count_a"] == 4 for row in rows)
    assert all(row["cluster_count_b"] == 4 for row in rows)
    assert all(row["intersection_cluster_count"] == 4 for row in rows)
    assert all(np.isclose(row["cgm_standard_error"], expected) for row in rows)


def test_repeated_two_way_clusters_do_not_become_claim_eligible():
    accumulator = TwoWayAccumulator()
    for index in range(200):
        value = 1.0 if index % 2 == 0 else -1.0
        accumulator.add(
            np.full(4, value, dtype=np.float64),
            cluster_a=f"surface-{index % 2}",
            cluster_b=f"onion-{(index // 2) % 2}",
        )

    rows = accumulator.summarize(level_a="surface_host", level_b="onion_host")
    assert all(row["n_matches"] == 200 for row in rows)
    assert all(row["cluster_count_a"] == 2 for row in rows)
    assert all(row["cluster_count_b"] == 2 for row in rows)
    assert all(not row["cluster_counts_ge_20"] for row in rows)


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
                    "natural_host_sha256": f"{corpus}-host-{match_id}",
                    "natural_source_sha256": f"{corpus}-source-{match_id}",
                }
                # Add the same mask nuisance to every class. It must disappear
                # after within-payload mask averaging and the paired DiD.
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
    assert len(result["host_two_way_cgm"]) == 4
    for row in result["host_two_way_cgm"]:
        assert np.isclose(row["mean_difference_in_differences"], 2.0)
        assert row["cluster_count_a"] == 2
        assert row["cluster_count_b"] == 2
        assert row["eligible_for_confirmatory_claim"] is False
        assert row["strict_confirmatory_pass"] is False
    assert Path(result["output"]).exists()
