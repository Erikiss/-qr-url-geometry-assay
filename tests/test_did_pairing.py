import json
from pathlib import Path

import numpy as np

from qr_assay.did import analyze_difference_in_differences
from qr_assay.fileio import open_text

METRICS = (
    "data_radial_mean",
    "data_centroid_radius",
    "data_orientation_cos2",
    "data_orientation_sin2",
)


def _write_design(path: Path, onion_assignment: list[int]) -> None:
    surface_effects = [0.0, 1.0, 2.0, 3.0]
    onion_effects = [4.0, 5.0, 6.0, 7.0]
    rows = []
    for match_id in range(4):
        onion_index = onion_assignment[match_id]
        classes = {
            "surface_natural": (surface_effects[match_id], f"surface-{match_id}"),
            "surface_synthetic": (0.0, f"surface-{match_id}"),
            "onion_natural": (onion_effects[onion_index], f"onion-{onion_index}"),
            "onion_synthetic": (0.0, f"onion-{onion_index}"),
        }
        for payload_class, (base, host) in classes.items():
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
                    "natural_host_sha256": host,
                    "natural_source_sha256": f"{corpus}-source-{host}",
                }
                for metric in METRICS:
                    row[metric] = base + 0.1 * mask
                rows.append(row)
    with open_text(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _run(output: Path, assignment: list[int]):
    output.mkdir()
    path = output / "features.jsonl.gz"
    _write_design(path, assignment)
    config = {
        "outputs": {"directory": str(output), "features_file": path.name},
        "qr": {"masks": [0, 1]},
    }
    return analyze_difference_in_differences(config)


def test_cross_corpus_repairing_does_not_change_did_or_se(tmp_path: Path):
    first = _run(tmp_path / "first", [0, 1, 2, 3])
    second = _run(tmp_path / "second", [2, 0, 3, 1])

    first_rows = first["host_independent_cluster_difference"]
    second_rows = second["host_independent_cluster_difference"]
    assert [row["metric"] for row in first_rows] == [row["metric"] for row in second_rows]
    for left, right in zip(first_rows, second_rows, strict=True):
        assert np.isclose(
            left["mean_difference_in_differences"],
            right["mean_difference_in_differences"],
        )
        assert np.isclose(
            left["independent_cluster_standard_error"],
            right["independent_cluster_standard_error"],
        )
