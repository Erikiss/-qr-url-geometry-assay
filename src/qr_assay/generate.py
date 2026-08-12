from __future__ import annotations

import json
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

from PIL import Image

from .config import effective_workers
from .fileio import open_text
from .geometry import (
    data_module_mask,
    geometry_features,
    make_qr,
    transform_grid,
    transform_matrix,
)


def _chunks(iterator: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items() if key != "matrix_sha256"}


def _process_chunk(
    args: tuple[list[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    records, config = args
    output: list[dict[str, Any]] = []
    ecc = str(config["qr"]["error_correction"]).upper()
    border = int(config["qr"].get("border", 0))
    if border != 0:
        raise ValueError("Scientific feature generation requires qr.border=0")
    masks = [int(mask) for mask in config["qr"]["masks"]]
    transforms = list(transform_grid(config))
    for record in records:
        payload = record.get("payload")
        if payload is None:
            raise ValueError("Generation requires outputs.store_payload_text=true")
        for mask in masks:
            base, version = make_qr(payload, error_correction=ecc, mask=mask, border=border)
            base_region = data_module_mask(version)
            if base_region.shape != base.shape:
                raise AssertionError("QR data-region mask shape mismatch")
            base_features = geometry_features(base)
            base_data_features = geometry_features(base, base_region)
            for transform in transforms:
                matrix = transform_matrix(base, **transform)
                region_transform = dict(transform)
                region_transform["inverted"] = False
                region = transform_matrix(base_region, **region_transform)
                features = geometry_features(matrix)
                data_features = geometry_features(matrix, region)
                row = {
                    "match_id": record["match_id"],
                    "payload_class": record["payload_class"],
                    "grammar": record["grammar"],
                    "synthetic": record["synthetic"],
                    "synthetic_mode": record.get("synthetic_mode"),
                    "source_sha256": record.get("source_sha256"),
                    "natural_source_sha256": record.get("natural_source_sha256"),
                    "natural_host_sha256": record.get("natural_host_sha256"),
                    "host_sha256": record.get("host_sha256"),
                    "onion_version": record.get("onion_version"),
                    "payload_sha256": record["payload_sha256"],
                    "byte_length": record["byte_length"],
                    "qr_version": version,
                    "error_correction": ecc,
                    "mask": mask,
                    "base_density": base_features["density"],
                    "base_data_density": base_data_features["density"],
                    **transform,
                    **features,
                    **_prefixed("data_", data_features),
                }
                output.append(row)
    return output


def _save_png(matrix, destination: Path, quiet_zone: int = 4, module_pixels: int = 8) -> None:
    import numpy as np

    padded = np.pad(matrix, quiet_zone, constant_values=0)
    pixels = np.where(padded, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").resize(
        (pixels.shape[1] * module_pixels, pixels.shape[0] * module_pixels),
        Image.Resampling.NEAREST,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def _write_examples(config: dict[str, Any], payload_path: Path, output_dir: Path) -> int:
    limit = int(config["outputs"].get("png_examples_per_class", 0))
    if limit <= 0:
        return 0
    counts: dict[str, int] = {}
    written = 0
    ecc = str(config["qr"]["error_correction"]).upper()
    example_mask = int(config["qr"]["masks"][0])
    for record in _read_jsonl(payload_path):
        cls = record["payload_class"]
        if counts.get(cls, 0) >= limit:
            continue
        matrix, _ = make_qr(record["payload"], error_correction=ecc, mask=example_mask, border=0)
        path = (
            output_dir
            / "examples"
            / cls
            / f"match-{record['match_id']:06d}-mask-{example_mask}.png"
        )
        _save_png(matrix, path)
        counts[cls] = counts.get(cls, 0) + 1
        written += 1
        if len(counts) == 4 and all(value >= limit for value in counts.values()):
            break
    return written


def generate_features(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["outputs"]["directory"])
    payload_path = output_dir / config["outputs"]["payloads_file"]
    features_path = output_dir / config["outputs"]["features_file"]
    if not payload_path.exists():
        raise FileNotFoundError(f"Prepare payloads first: {payload_path}")
    workers = effective_workers(config)
    chunk_size = int(config["execution"].get("chunk_size", 128))
    start = time.perf_counter()
    rows = 0
    chunks = 0
    iterator = _chunks(_read_jsonl(payload_path), chunk_size)
    with open_text(features_path, "wt") as output:
        if workers == 1:
            results = map(_process_chunk, ((chunk, config) for chunk in iterator))
            for result in results:
                chunks += 1
                for row in result:
                    output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                    rows += 1
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                args = ((chunk, config) for chunk in iterator)
                for result in executor.map(_process_chunk, args, chunksize=1):
                    chunks += 1
                    for row in result:
                        output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                        rows += 1
    examples = _write_examples(config, payload_path, output_dir)
    elapsed = time.perf_counter() - start
    return {
        "features_path": str(features_path),
        "rows": rows,
        "chunks": chunks,
        "workers": workers,
        "masks_per_payload": len(config["qr"]["masks"]),
        "elapsed_seconds": elapsed,
        "rows_per_second": rows / elapsed if elapsed else 0.0,
        "png_examples": examples,
        "feature_regions": ["full_matrix", "data_ecc_modules_only"],
    }
