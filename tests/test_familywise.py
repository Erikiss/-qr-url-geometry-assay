from pathlib import Path

from qr_assay.familywise import PREDECLARED_FAMILY_SIZE, analyze_familywise_core


def _row(metric, mean, se, clusters=30):
    return {
        "metric": metric,
        "n_matches": 100,
        "cluster_level": "host",
        "cluster_count": clusters,
        "effective_cluster_count": float(clusters),
        "max_cluster_size": 5,
        "mean_difference": mean,
        "cr1_standard_error": se,
        "ci95_low_normal_cr1": mean - 1.96 * se,
        "ci95_high_normal_cr1": mean + 1.96 * se,
        "estimable": True,
        "cluster_count_ge_20": clusters >= 20,
    }


def test_familywise_analysis_uses_fixed_eight_cell_family(tmp_path: Path):
    metrics = [
        "data_radial_mean",
        "data_centroid_radius",
        "data_orientation_cos2",
        "data_orientation_sin2",
    ]
    contrasts = {}
    for contrast_index, contrast in enumerate(("surface", "onion")):
        host_rows = []
        source_rows = []
        for metric_index, metric in enumerate(metrics):
            strong = contrast_index == 0 and metric_index == 0
            row = _row(metric, 8.0 if strong else 0.0, 1.0)
            host_rows.append(row)
            source_rows.append(dict(row))
        contrasts[contrast] = {
            "host_clustered": host_rows,
            "source_clustered": source_rows,
        }

    config = {"outputs": {"directory": str(tmp_path)}}
    result = analyze_familywise_core(config, {"contrasts": contrasts})

    assert result["predeclared_family_size"] == PREDECLARED_FAMILY_SIZE == 8
    assert len(result["host"]) == 8
    strong = next(
        row
        for row in result["host"]
        if row["contrast"] == "surface" and row["metric"] == "data_radial_mean"
    )
    assert strong["holm_reject_familywise_0_05"] is True
    assert strong["bonferroni_ci_excludes_zero"] is True
    nulls = [row for row in result["host"] if row is not strong]
    assert all(not row["holm_reject_familywise_0_05"] for row in nulls)
    assert all(not row["bonferroni_ci_excludes_zero"] for row in nulls)
    assert Path(result["output"]).exists()


def test_low_cluster_count_never_becomes_confirmatory_claim(tmp_path: Path):
    metrics = [
        "data_radial_mean",
        "data_centroid_radius",
        "data_orientation_cos2",
        "data_orientation_sin2",
    ]
    rows = [_row(metric, 20.0, 1.0, clusters=5) for metric in metrics]
    contrasts = {
        "surface": {"host_clustered": rows, "source_clustered": [dict(row) for row in rows]},
        "onion": {
            "host_clustered": [dict(row) for row in rows],
            "source_clustered": [dict(row) for row in rows],
        },
    }
    config = {"outputs": {"directory": str(tmp_path)}}
    result = analyze_familywise_core(config, {"contrasts": contrasts})
    assert all(not row["eligible_for_confirmatory_claim"] for row in result["host"])
    assert all(not row["holm_reject_familywise_0_05"] for row in result["host"])
    assert all(not row["bonferroni_ci_excludes_zero"] for row in result["host"])
