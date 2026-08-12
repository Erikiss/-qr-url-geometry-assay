from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 267010,
    "sources": {
        "surface": {
            "paths": [],
            "scan_limit": None,
            "deduplicate": False,
            "hash_inputs": True,
        },
        "onion": {
            "paths": [],
            "scan_limit": None,
            "deduplicate": True,
            "granularity": "url",
            "hash_inputs": True,
        },
    },
    "sampling": {
        "target_pairs": 10_000,
        "reservoir_per_length": 5_000,
        "min_bytes": 16,
        "max_bytes": 512,
    },
    "qr": {
        "error_correction": "M",
        "masks": list(range(8)),
        "border": 0,
    },
    "transforms": {
        "rotations": [0, 90, 180, 270],
        "inversions": [False, True],
        "reflections": ["none"],
        "scales": [1],
    },
    "execution": {
        "workers": 0,
        "chunk_size": 128,
    },
    "outputs": {
        "directory": "results/standard",
        "payloads_file": "payloads.jsonl.gz",
        "features_file": "features.jsonl.gz",
        "report_file": "report.md",
        "manifest_file": "run_manifest.json",
        "store_payload_text": True,
        "png_examples_per_class": 2,
    },
    "analysis": {
        "density_bin_width": 0.005,
    },
}


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid override value: {value!r}") from exc


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        cursor: dict[str, Any] = result
        keys = dotted.split(".")
        for key in keys[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                child = {}
                cursor[key] = child
            cursor = child
        cursor[keys[-1]] = _parse_scalar(raw)
    return result


def _resolve_paths(config: dict[str, Any], base_dir: Path) -> None:
    for corpus in ("surface", "onion"):
        paths = config["sources"][corpus].get("paths", [])
        config["sources"][corpus]["paths"] = [
            str((base_dir / p).resolve()) if not Path(p).is_absolute() else str(Path(p))
            for p in paths
        ]
    out = Path(config["outputs"]["directory"])
    if not out.is_absolute():
        config["outputs"]["directory"] = str((base_dir / out).resolve())


def validate(config: dict[str, Any]) -> None:
    target = int(config["sampling"]["target_pairs"])
    if target < 1:
        raise ConfigError("sampling.target_pairs must be positive")
    if int(config["sampling"]["reservoir_per_length"]) < 1:
        raise ConfigError("sampling.reservoir_per_length must be positive")
    rotations = config["transforms"]["rotations"]
    if not rotations or any(int(x) % 90 for x in rotations):
        raise ConfigError("transforms.rotations must contain multiples of 90")
    reflections = set(config["transforms"]["reflections"])
    valid_reflections = {"none", "horizontal", "vertical", "diagonal", "anti_diagonal"}
    if not reflections <= valid_reflections:
        raise ConfigError(f"Unknown reflection: {sorted(reflections - valid_reflections)}")
    if any(int(scale) < 1 for scale in config["transforms"]["scales"]):
        raise ConfigError("transforms.scales must be positive integers")
    if 0 not in [int(x) for x in rotations]:
        raise ConfigError("transforms.rotations must include 0 for the controlled baseline")
    if "none" not in reflections or 1 not in [int(x) for x in config["transforms"]["scales"]]:
        raise ConfigError("transforms must include reflection=none and scale=1")
    if False not in [bool(x) for x in config["transforms"]["inversions"]]:
        raise ConfigError("transforms.inversions must include false")
    masks = [int(x) for x in config["qr"]["masks"]]
    if not masks or any(x < 0 or x > 7 for x in masks):
        raise ConfigError("qr.masks must be a non-empty subset of 0..7")
    if str(config["qr"]["error_correction"]).upper() not in {"L", "M", "Q", "H"}:
        raise ConfigError("qr.error_correction must be L, M, Q, or H")
    if config["sources"]["onion"].get("granularity", "url") not in {"url", "origin"}:
        raise ConfigError("sources.onion.granularity must be url or origin")


def load_config(path: str | os.PathLike[str], overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULTS, loaded)
    config = apply_overrides(config, overrides)
    _resolve_paths(
        config,
        config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent,
    )
    validate(config)
    return config


def canonical_json(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def effective_workers(config: dict[str, Any]) -> int:
    configured = int(config["execution"].get("workers", 0))
    return configured if configured > 0 else max(1, os.cpu_count() or 1)
