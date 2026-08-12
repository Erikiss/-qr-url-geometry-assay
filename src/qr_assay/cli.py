from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from .analysis import analyze_features
from .config import DEFAULTS, effective_workers, load_config
from .generate import _process_chunk, generate_features
from .runner import run_all
from .sampling import prepare_payloads
from .sources import acquire_manifest
from .synthetic import synthetic_onion_label


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML run configuration")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key; may be repeated",
    )


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _demo_payloads(output_dir: Path, pairs: int, seed: int) -> dict[str, Any]:
    """Create safe fixtures whose *scheme-stripped* payload lengths match."""
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = output_dir / "surface.txt"
    onion_path = output_dir / "onion.txt"
    rng = random.Random(seed)
    surface_rows: list[str] = []
    onion_rows: list[str] = []
    for index in range(pairs):
        path = f"/p/{index:06d}" if index % 3 else "/"
        onion_host = synthetic_onion_label(56, rng) + ".onion"
        onion = f"http://{onion_host}{path}"
        onion_stripped_bytes = len((onion_host + path).encode("utf-8"))
        fixed = len((".invalid" + path).encode("utf-8"))
        label_length = onion_stripped_bytes - fixed
        label = "".join(
            rng.choice(string.ascii_lowercase + string.digits) for _ in range(label_length)
        )
        surface = f"https://{label}.invalid{path}"
        assert len(surface.split("://", 1)[1].encode("utf-8")) == onion_stripped_bytes
        surface_rows.append(surface)
        onion_rows.append(onion)
    surface_path.write_text("\n".join(surface_rows) + "\n", encoding="utf-8")
    onion_path.write_text("\n".join(onion_rows) + "\n", encoding="utf-8")
    return {
        "surface": str(surface_path),
        "onion": str(onion_path),
        "pairs": pairs,
        "demo_only": True,
    }


def _benchmark(count: int, workers: int) -> dict[str, Any]:
    rng = random.Random(267010)
    records = []
    for index in range(count):
        payload = synthetic_onion_label(56, rng) + ".onion/"
        records.append(
            {
                "match_id": index,
                "payload_class": "onion_natural",
                "grammar": "onion",
                "synthetic": False,
                "synthetic_mode": None,
                "payload": payload,
                "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "byte_length": len(payload.encode()),
            }
        )
    config = __import__("copy").deepcopy(DEFAULTS)
    config["execution"]["workers"] = workers
    workers = effective_workers(config)
    chunk_size = 64
    chunks = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]
    start = time.perf_counter()
    rows = 0
    if workers == 1:
        for chunk in chunks:
            rows += len(_process_chunk((chunk, config)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_process_chunk, ((chunk, config) for chunk in chunks)):
                rows += len(result)
    elapsed = time.perf_counter() - start
    masks = len(config["qr"]["masks"])
    base_qr_codes = count * masks
    return {
        "payloads": count,
        "masks_per_payload": masks,
        "base_qr_codes": base_qr_codes,
        "transform_rows": rows,
        "workers": workers,
        "elapsed_seconds": elapsed,
        "base_qr_per_second": base_qr_codes / elapsed,
        "estimated_seconds_for_1m_base_qr": 1_000_000 * elapsed / base_qr_codes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qr-assay",
        description="CPU-only QR geometry assay for natural and structural-null URL corpora.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Normalize, deduplicate, match, and synthesize payloads"
    )
    _common_config(prepare)
    generate = sub.add_parser("generate", help="Generate QR matrices and geometric features")
    _common_config(generate)
    analyze = sub.add_parser("analyze", help="Run controlled/balanced statistical summaries")
    _common_config(analyze)
    run = sub.add_parser("run", help="Run prepare, generate, and analyze")
    _common_config(run)
    demo = sub.add_parser("demo-data", help="Create safe synthetic fixtures for smoke testing only")
    demo.add_argument("--output-dir", default="data/raw/demo")
    demo.add_argument("--pairs", type=int, default=250)
    demo.add_argument("--seed", type=int, default=267010)
    acquire = sub.add_parser(
        "acquire", help="Download declared clearnet source files with optional hashes"
    )
    acquire.add_argument("--manifest", required=True)
    acquire.add_argument("--output-dir", required=True)
    benchmark = sub.add_parser("benchmark", help="Measure this machine and estimate full-run time")
    benchmark.add_argument("--count", type=int, default=1000)
    benchmark.add_argument("--workers", type=int, default=0, help="0 uses all logical CPUs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo-data":
            _print(_demo_payloads(Path(args.output_dir), args.pairs, args.seed))
        elif args.command == "acquire":
            _print(acquire_manifest(args.manifest, args.output_dir))
        elif args.command == "benchmark":
            _print(_benchmark(args.count, args.workers))
        else:
            config = load_config(args.config, args.set)
            if args.command == "prepare":
                _print(prepare_payloads(config))
            elif args.command == "generate":
                _print(generate_features(config))
            elif args.command == "analyze":
                _print(analyze_features(config))
            elif args.command == "run":
                _print(run_all(config))
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
