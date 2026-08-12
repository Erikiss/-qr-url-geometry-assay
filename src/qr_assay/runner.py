from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from . import __version__
from .analysis import analyze_features, write_report
from .cluster import analyze_clustered_core
from .codeword import analyze_codeword_regions
from .config import config_sha256
from .generate import generate_features
from .null_qc import analyze_null_qc
from .sampling import prepare_payloads


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _package_versions() -> dict[str, str | None]:
    packages = ("qr-url-geometry-assay", "qrcode", "numpy", "Pillow", "PyYAML")
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def run_all(config: dict[str, Any]) -> dict[str, Any]:
    started = datetime.now(UTC)
    start = time.perf_counter()
    preparation = prepare_payloads(config)
    null_qc = analyze_null_qc(config)
    generation = generate_features(config)
    analysis = analyze_features(config)
    clustered = analyze_clustered_core(config)
    codeword = (
        analyze_codeword_regions(config)
        if bool(config.get("analysis", {}).get("codeword_diagnostics", False))
        else None
    )
    report_path = write_report(config, preparation, generation, analysis)
    output_dir = Path(config["outputs"]["directory"])
    manifest = {
        "assay_version": __version__,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "config_sha256": config_sha256(config),
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "git_commit": _git_commit(),
            "packages": _package_versions(),
        },
        "preparation": preparation,
        "null_qc": {
            "output": null_qc["output"],
            "synthetic_mode": null_qc["synthetic_mode"],
            "passed_hard_invariants": null_qc["passed_hard_invariants"],
            "corpora": null_qc["corpora"],
        },
        "generation": generation,
        "analysis_summary": {
            "rows": analysis["rows"],
            "balance": analysis["balance"],
            "quality_control": analysis["quality_control"],
            "features_sha256": analysis["features_sha256"],
        },
        "cluster_analysis": {
            "output": clustered["output"],
            "core_metrics": clustered["core_metrics"],
            "complete_matches_seen": clustered["complete_matches_seen"],
            "invalid_matches": clustered["invalid_matches"],
            "method": clustered["method"],
        },
        "codeword_analysis": (
            {
                "output": codeword["output"],
                "complete_matches": codeword["complete_matches"],
                "invalid_matches": codeword["invalid_matches"],
                "qrs_encoded": codeword["qrs_encoded"],
            }
            if codeword is not None
            else None
        ),
        "report": str(report_path),
    }
    manifest_path = output_dir / config["outputs"]["manifest_file"]
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
