import json
from pathlib import Path

import yaml

from qr_assay.config import load_config
from qr_assay.runner import run_all


def test_end_to_end(tmp_path: Path):
    base32_a = "a" * 56
    base32_b = "b" * 56
    onions = [f"http://{base32_a}.onion/", f"http://{base32_b}.onion/x"]
    surfaces = []
    for index, onion in enumerate(onions):
        path = "/" if index == 0 else "/x"
        fixed = len(("https://" + ".invalid" + path).encode())
        label = "s" * (len(onion.encode()) - fixed)
        surfaces.append(f"https://{label}.invalid{path}")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "surface.txt").write_text("\n".join(surfaces), encoding="utf-8")
    (raw / "onion.txt").write_text("\n".join(onions), encoding="utf-8")
    config_path = tmp_path / "run.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sources": {
                    "surface": {
                        "paths": [str(raw / "surface.txt")],
                        "deduplicate": True,
                    },
                    "onion": {"paths": [str(raw / "onion.txt")], "deduplicate": True},
                },
                "sampling": {"target_pairs": 2, "reservoir_per_length": 10},
                "transforms": {
                    "rotations": [0, 90],
                    "inversions": [False, True],
                    "reflections": ["none"],
                    "scales": [1],
                },
                "execution": {"workers": 1, "chunk_size": 2},
                "outputs": {
                    "directory": str(tmp_path / "results"),
                    "png_examples_per_class": 0,
                },
                "analysis": {"density_bin_width": 0.1},
            }
        ),
        encoding="utf-8",
    )
    manifest = run_all(load_config(config_path))
    assert manifest["preparation"]["matched_pairs"] == 2
    assert manifest["generation"]["rows"] == 2 * 4 * 2 * 2
    assert manifest["analysis_summary"]["quality_control"]["passed"]
    assert (tmp_path / "results" / "report.md").exists()
    with (tmp_path / "results" / "analysis.json").open(encoding="utf-8") as handle:
        analysis = json.load(handle)
    assert len(analysis["paired_effects"]) > 0
    assert all("p_holm" in effect for effect in analysis["paired_effects"])
    with (tmp_path / "results" / "run_manifest.json").open(encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["config_sha256"] == manifest["config_sha256"]
