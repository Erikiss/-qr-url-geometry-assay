# QR URL Geometry Assay

A reproducible, CPU-only experiment for separating QR geometry caused by URL
structure from geometry caused by payload semantics.

The repository implements the complete three-corpus design:

1. **Natural surface URLs** — streamed from DomainsProject or compatible files.
2. **Natural onion URLs** — merged and deduplicated across any number of current
   and historical crawl exports.
3. **Grammar-matched synthetic URLs** — random tokens with the same delimiters,
   character classes, UTF-8 byte length, and surface/onion grammar.

The synthetic family has a surface-matched and an onion-matched arm, so every
matched unit contains four payload classes:

`surface_natural`, `onion_natural`, `surface_synthetic`, `onion_synthetic`.

## What is measured

The preregistered core is a `payload class × 4 rotations × 2 polarities`
factorial design. It records three families of geometric observables:

1. **Radial movement** — radial moments and covariance trace.
2. **Translation** — the normalized black-module centroid.
3. **Rotation/cyclicity** — principal-axis angle, anisotropy, and the complete
   0°/90°/180°/270° orbit.

Horizontal, vertical, diagonal, and anti-diagonal reflection plus integer
nearest-neighbor scaling are implemented in `configs/extended.yml`.

Every matched unit is controlled on UTF-8 byte length, QR version,
error-correction level, forced byte-mode encoding, and mask pattern. The
analysis then intersects black-module-density bins across all four classes.
Bit inversion is the exact involution `Q' = 1 - Q`; it is not geometric
inversion.

## Five-minute smoke test

Python 3.11+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
qr-assay demo-data --output-dir data/raw/demo --pairs 250
qr-assay run --config configs/smoke.yml
```

The demo inputs are synthetic fixtures for testing the pipeline. They must not
be used as scientific data.

## Real data

Place one or more files/directories under:

```text
data/raw/surface/
data/raw/onion/
```

Accepted inputs include plain text, CSV/JSONL-like text, `.gz`, `.xz`, `.bz2`,
and `.zip`. Surface lines may be domains or URLs. The onion extractor finds v2
and v3 `.onion` URLs inside arbitrary crawl-export lines. No page content is
used.

DomainsProject publishes multi-billion-domain lists; its documented unpacked
format is one domain per line. Stream a subset or the full files directly into
`data/raw/surface/` rather than loading them into RAM.

```bash
qr-assay acquire \
  --manifest configs/sources.example.yml \
  --output-dir data/raw/onion

qr-assay run --config configs/standard.yml
```

Add every historical/current onion crawl export to `data/raw/onion/`. The
normalizer deduplicates them before reservoir sampling. The code deliberately
does **not** connect to Tor, probe services, or download onion page content.

`sources.onion.granularity` selects the research unit. `url` retains observed
paths and queries; `origin` deduplicates by natural onion host and canonicalizes
it to `http://<host>.onion/`. The exhaustive config uses `origin`, so “all
natural onions” means every unique onion site in the pinned source snapshots,
not repeated crawl paths.

## RunPod CPU Pod (on demand)

No GPU is required. On a CPU Pod with Docker:

```bash
git clone YOUR_REPOSITORY_URL qr-url-geometry-assay
cd qr-url-geometry-assay
docker build -t qr-assay .

docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  qr-assay run --config configs/standard.yml
```

Use a persistent/network volume only for the raw corpora and `results/`. The
container itself can be ephemeral. `execution.workers: 0` uses all logical
CPUs visible in the container.

Before paying for a large machine, benchmark the exact CPU flavor:

```bash
qr-assay benchmark --count 10000 --workers 0
```

The command reports measured base-QR throughput and a projection for one
million base QR codes. Final wall time also includes source scanning and two
streaming analysis passes.

### Planning ranges

| profile | matched units | base QR codes | transform rows | suggested CPU | planning time |
|---|---:|---:|---:|---:|---:|
| smoke | 100 | 400 | 3,200 | laptop | under 2 min |
| standard | 100,000 | 400,000 | 3.2 million | 16–32 vCPU | roughly 10–45 min |
| exhaustive core | 1.5 million | 6 million | 48 million | 32–64 vCPU | roughly 1–3 h |

These are deliberately broad until `qr-assay benchmark` has run on the chosen
CPU. On the development runner (9 logical CPUs), a complete 10,000-match run
produced 320,000 rows in 27.5 seconds; raw QR generation projected one million
base codes in 4.9 minutes. Real source scanning and compressed JSONL I/O make
the larger planning ranges more appropriate. The exhaustive run is usually
limited more by I/O than by QR arithmetic. Gzip level 1 reduced a measured
308 MB feature file to 47 MB, putting a 1.5-million-match core run in the
single-digit-gigabyte range for similar payload lengths.

## Commands

```text
qr-assay prepare    normalize, deduplicate, pair, and synthesize payloads
qr-assay generate   generate QR matrices and stream geometric features
qr-assay analyze    create controlled and density-balanced summaries
qr-assay run        execute all three stages and write a run manifest
qr-assay benchmark  measure the current CPU
qr-assay acquire    fetch declared clearnet source files with SHA-256 checks
```

Any YAML value can be overridden without editing the file:

```bash
qr-assay run --config configs/standard.yml \
  --set sampling.target_pairs=250000 \
  --set execution.workers=48
```

## Outputs

Each run directory contains:

- `payloads.jsonl.gz` — matched payload metadata and provenance;
- `features.jsonl.gz` — one row per QR transform;
- `analysis.json` — raw and density-balanced group summaries;
- `report.md` — concise human-readable result;
- `run_manifest.json` — full config, environment, hashes, timings, and QC;
- `examples/` — a few PNGs for visual inspection.

Raw corpora and generated results are gitignored. Publish source manifests,
checksums, configuration, code commit, and the run manifest—not live crawl
content—unless the upstream dataset license explicitly permits redistribution.

## Reproducibility contract

- RNG seed is fixed in each config.
- Sampling is reservoir-based and bounded in memory.
- All four classes share match ID, byte length, QR version, ECC, and mask.
- QR byte mode is forced to avoid library-dependent segment optimization.
- Every feature file is SHA-256 hashed.
- Every declared raw source file is inventoried by path, byte size, mtime, and
  SHA-256 (`hash_inputs: false` is available only for exploratory scans).
- A run fails QC if inversion density or matched controls disagree.
- Natural-vs-synthetic and onion-vs-surface effects are computed within
  `match_id`, with 95% intervals, standardized paired effects, and Holm-adjusted
  normal-approximation p-values. Effect sizes and intervals remain primary.

See [docs/protocol.md](docs/protocol.md) and
[docs/data-sources.md](docs/data-sources.md) for the preregistered comparisons
and provenance rules. [docs/source-catalog.md](docs/source-catalog.md) separates
automatically reproducible current lists from historical research corpora that
must be requested from their authors.
