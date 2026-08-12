import json
import random
from pathlib import Path

import yaml

from qr_assay.config import load_config
from qr_assay.runner import run_all
from qr_assay.synthetic import synthetic_onion_label


def test_end_to_end(tmp_path: Path):
    rng = random.Random(11)
    labels = [synthetic_onion_label(56, rng) for _ in range(2)]
    onions = [f"http://{labels[0]}.onion/", f"http://{labels[1]}.onion/x"]
    surfaces = []
    for index, onion in enumerate(onions):
        onion_stripped = onion.split("://", 1)[1]
        path = "/" if index == 0 else "/x"
        fixed = len(("s.invalid" + path).encode())
        label = "s" * (len(onion_stripped.encode()) - fixed + 1)
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
                        "granularity": "url",
                    },
                    "onion": {
                        "paths": [str(raw / "onion.txt")],
                        "deduplicate": True,
                        "granularity": "url",
                    },
                },
                "sampling": {
                    "target_pairs": 2,
                    "reservoir_per_length": 10,
                    "scheme_policy": "strip",
                },
                "qr": {"masks": [0, 1, 2]},
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
    # 2 matches x 4 payload classes x 3 masks x 2 rotations x 2 polarities
    assert manifest["generation"]["rows"] == 2 * 4 * 3 * 2 * 2
    assert manifest["generation"]["masks_per_payload"] == 3
    assert manifest["analysis_summary"]["quality_control"]["passed"]
    assert (tmp_path / "results" / "report.md").exists()
    assert (tmp_path / "results" / "confirmatory_report.md").exists()
    assert Path(manifest["report"]).name == "confirmatory_report.md"
    assert Path(manifest["descriptive_report"]).name == "report.md"
    with (tmp_path / "results" / "confirmatory_report.md").open(encoding="utf-8") as handle:
        confirmatory_text = handle.read()
    assert "Confirmatory natural-vs-null effects" in confirmatory_text
    assert "Host-clustered" in confirmatory_text or "host-clustered" in confirmatory_text
    with (tmp_path / "results" / "analysis.json").open(encoding="utf-8") as handle:
        analysis = json.load(handle)
    assert len(analysis["paired_effects"]) > 0
    assert all("p_holm" in effect for effect in analysis["paired_effects"])
    assert all(effect["n_matches"] == 2 for effect in analysis["paired_effects"])
    with (tmp_path / "results" / "run_manifest.json").open(encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["config_sha256"] == manifest["config_sha256"]
