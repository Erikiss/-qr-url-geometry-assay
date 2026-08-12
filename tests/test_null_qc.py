import hashlib
import json
from pathlib import Path

from qr_assay.fileio import open_text
from qr_assay.null_qc import analyze_null_qc


def _record(match_id, payload_class, grammar, synthetic, payload):
    encoded = payload.encode("utf-8")
    return {
        "match_id": match_id,
        "payload_class": payload_class,
        "grammar": grammar,
        "synthetic": synthetic,
        "payload": payload,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "byte_length": len(encoded),
    }


def test_null_qc_reports_unchanged_controls_and_unicode(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    path = output / "payloads.jsonl.gz"
    rows = [
        _record(0, "surface_natural", "surface", False, "/aaaa/1111"),
        _record(0, "surface_synthetic", "surface", True, "/aaaa/1111"),
        _record(0, "onion_natural", "onion", False, "/café"),
        _record(0, "onion_synthetic", "onion", True, "/café"),
    ]
    with open_text(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "outputs": {"directory": str(output), "payloads_file": path.name},
        "sampling": {"synthetic_mode": "token_shuffle"},
    }
    result = analyze_null_qc(config)

    assert result["passed_hard_invariants"] is True
    assert result["corpora"]["surface"]["unchanged_fraction"] == 1.0
    assert result["corpora"]["onion"]["unchanged_fraction"] == 1.0
    assert result["corpora"]["surface"]["non_ascii_natural_fraction"] == 0.0
    assert result["corpora"]["onion"]["non_ascii_natural_fraction"] == 1.0
