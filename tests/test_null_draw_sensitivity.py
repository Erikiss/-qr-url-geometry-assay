import hashlib
import json
from pathlib import Path

from qr_assay.fileio import open_text
from qr_assay.null_draw_sensitivity import run_null_draw_sensitivity


def _natural(match_id, corpus, payload):
    encoded = payload.encode("utf-8")
    return {
        "match_id": match_id,
        "payload_class": f"{corpus}_natural",
        "grammar": corpus,
        "synthetic": False,
        "payload": payload,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "byte_length": len(encoded),
        "natural_host_sha256": f"{corpus}-host-{match_id}",
        "natural_source_sha256": f"{corpus}-source-{match_id}",
    }


def test_degenerate_null_draws_are_numerically_stable(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    path = output / "payloads.jsonl.gz"
    rows = []
    for match_id in range(2):
        rows.append(_natural(match_id, "surface", "/aaaa/1111"))
        rows.append(_natural(match_id, "onion", "/aaaa/1111"))
    with open_text(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    config = {
        "seed": 17,
        "outputs": {"directory": str(output), "payloads_file": path.name},
        "sampling": {"synthetic_mode": "token_shuffle"},
        "qr": {"error_correction": "M", "masks": [0]},
    }
    result = run_null_draw_sensitivity(
        config,
        pilot_matches=2,
        draws=2,
    )

    assert result["status"] == "PASS"
    assert result["single_draw_acceptable_on_pilot"] is True
    assert len(result["cells"]) == 12
    assert all(cell["degenerate_stable_zero"] for cell in result["cells"])
    assert all(cell["cell_pass"] for cell in result["cells"])
    assert Path(result["output"]).exists()
