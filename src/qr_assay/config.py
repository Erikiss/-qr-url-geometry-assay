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
            "granularity": "url",
            "hash_inputs": True,
        },
        "onion": {
            "paths": [],
            "scan_limit": None,
            "deduplicate": True,
            "granularity": "url",
            "hash_inputs": True,
            # v2 Onion services are a historical format with a radically different
            # hostname length. Do not mix them into the confirmatory v3 population.
            "versions": [3],
        },
    },
    "sampling": {
        "target_pairs": 10_000,
        "reservoir_per_length": 5_000,
        "min_bytes": 16,
        "max_bytes": 512,
        "scheme_policy": "strip",
        "synthetic_mode": "grammar_random",
        # Weight matched lengths by their shared empirical support rather than
        # giving rare and common byte lengths equal influence.
        "length_weighting": "overlap",
    },
    "qr": {
        "error_correction": "M",
        "masks": list(range(8)),
        "border": 0,
    },
    "transforms": {
        "group": "factorial",
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
    if config["sampling"].get("scheme_policy", "strip") not in {"strip", "https", "preserve"}:
        raise ConfigError("sampling.scheme_policy must be strip, https, or preserve")
    if config["sampling"].get("synthetic_mode", "grammar_random") not in {
        "token_shuffle",
        "class_permute",
        "grammar_random",
    }:
        raise ConfigError(
            "sampling.synthetic_mode must be token_shuffle, class_permute, or grammar_random"
        )
    if config["sampling"].get("length_weighting", "overlap") not in {"overlap", "equal_length"}:
        raise ConfigError("sampling.length_weighting must be overlap or equal_length")

    valid_granularities = {"url", "origin", "path_query"}
    for corpus in ("surface", "onion"):
        granularity = config["sources"][corpus].get("granularity", "url")
        if granularity not in valid_granularities:
            raise ConfigError(
                f"sources.{corpus}.granularity must be url, origin, or path_query"
            )

    onion_versions = [int(value) for value in config["sources"]["onion"].get("versions", [3])]
    if not onion_versions or not set(onion_versions) <= {2, 3}:
        raise ConfigError("sources.onion.versions must be a non-empty subset of [2, 3]")
    if len(set(onion_versions)) != len(onion_versions):
        raise ConfigError("sources.onion.versions must not contain duplicates")

    group = str(config["transforms"].get("group", "factorial"))
    if group not in {"factorial", "d4"}:
        raise ConfigError("transforms.group must be factorial or d4")
    rotations = config["transforms"]["rotations"]
    if not rotations or any(int(x) % 90 for x in rotations):
        raise ConfigError("transforms.rotations must contain multiples of 90")
    reflections = set(config["transforms"]["reflections"])
    valid_reflections = {"none", "horizontal", "vertical", "diagonal", "anti_diagonal"}
    if not reflections <= valid_reflections:
        raise ConfigError(f"Unknown reflection: {sorted(reflections - valid_reflections)}")
    if any(int(scale) < 1 for scale in config["transforms"]["scales"]):
        raise ConfigError("transforms.scales must be positive integers")
    if group == "factorial":
        if 0 not in [int(x) for x in rotations]:
            raise ConfigError("transforms.rotations must include 0 for the controlled baseline")
        if "none" not in reflections:
            raise ConfigError("factorial transforms must include reflection=none")
    if 1 not in [int(x) for x in config["transforms"]["scales"]]:
        raise ConfigError("transforms.scales must include 1")
    if False not in [bool(x) for x in config["transforms"]["inversions"]]:
        raise ConfigError("transforms.inversions must include false")

    masks = [int(x) for x in config["qr"]["masks"]]
    if not masks or any(x < 0 or x > 7 for x in masks):
        raise ConfigError("qr.masks must be a non-empty subset of 0..7")
    if len(set(masks)) != len(masks):
        raise ConfigError("qr.masks must not contain duplicates")
    if str(config["qr"]["error_correction"]).upper() not in {"L", "M", "Q", "H"}:
        raise ConfigError("qr.error_correction must be L, M, Q, or H")


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
