# QR URL Geometry Assay

A reproducible, CPU-only stage-1 assay for separating QR geometry caused by natural URL byte/sequence structure from geometry expected under declared structural nulls.

> **Scope:** this repository does not measure semantics, attention heads, experts, or neural carrier rotation. A QR encoder sees bytes. Stage 1 locks down the stimulus family and its geometric nulls before a later model/checkpoint assay is attached.

## Corpus design

Each `match_id` contains four byte-length-matched payloads:

1. `surface_natural`
2. `onion_natural`
3. `surface_synthetic`
4. `onion_synthetic`

The confirmatory full run uses **origin vs origin** and strips schemes before matching, removing the arbitrary `https://` versus `http://` prefix difference. The path/query experiment uses **URL vs URL**. Mixed granularity is exploratory only.

Onion sources are unions of pinned current/historical crawl exports. “All onions” means every unique onion unit in the declared snapshots, not an enumerable ground truth for Tor.

## Structural-null ladder

Run the same design under progressively weaker nulls:

```text
token_shuffle   preserves token character multisets where meaningful
class_permute   preserves global ASCII unigram counts within character classes
grammar_random  preserves delimiters and lower/upper/digit positions only
```

For a v3 onion hostname, all modes substitute a fresh checksum-valid v3 onion address; the cryptographic host is not treated as a natural-language token. Path/query material still follows the selected null.

Select a null without editing YAML:

```bash
qr-assay run --config configs/standard.yml --set sampling.synthetic_mode=token_shuffle
qr-assay run --config configs/standard.yml --set sampling.synthetic_mode=class_permute
qr-assay run --config configs/standard.yml --set sampling.synthetic_mode=grammar_random
```

A difference is therefore a **lexical/sequence-structure effect relative to the declared null**, not a semantic effect.

## Mask-factorial QR design

Every payload is encoded under every configured QR mask. With the default masks `0..7`, mask is a within-payload nuisance factor rather than a between-payload assignment.

The primary transformed calibration is:

```text
4 rotations × 2 polarities
```

so each payload produces `8 masks × 8 transform states = 64` feature rows in the core design. Paired inference first averages the primary 0°/normal geometry over masks and then contributes **one observation per match_id**. Masks and deterministic transforms are not counted as independent replicates.

The extended design uses the exact eight elements of the square symmetry group `D4` rather than a redundant rotation × reflection product.

Bit inversion is the exact involution `Q' = 1 - Q`; it is not geometric circle inversion.

## What is measured

The stage-1 feature families are:

- **radial/stretching-like:** radial mean, radial standard deviation, covariance trace;
- **translation-like:** normalized black-module centroid and centroid radius;
- **axial orientation:** anisotropy plus `cos(2θ)` / `sin(2θ)`;
- **texture/symmetry controls:** horizontal/vertical transition rates and reflection/180° similarity.

`principal_angle_deg` is descriptive only because principal-axis direction is axial modulo 180°. Inferential comparisons use the doubled-angle embedding.

The 0°→90°→180°→270° orbit is **equivariance calibration**. Because the code explicitly rotates the matrix, cyclicity there is true by construction and is not evidence for cyclic dynamics in a transformer.

## Five-minute smoke test

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
qr-assay demo-data --output-dir data/raw/demo --pairs 250
qr-assay run --config configs/smoke.yml
```

Demo inputs are safe synthetic fixtures and must not be used as scientific data.

## Real data

Place source files under:

```text
data/raw/surface/
data/raw/onion/
```

Accepted inputs include plain text, CSV/JSONL-like text, `.gz`, `.xz`, `.bz2`, and `.zip`. Surface lines may be domains or URLs. The onion extractor finds v2/v3 `.onion` URLs inside arbitrary crawl-export lines. No onion page content is fetched.

DomainsProject is the intended large surface source. Add every licensed historical/current onion crawl export to `data/raw/onion/`; the pipeline normalizes and deduplicates before sampling.

```bash
qr-assay acquire --manifest configs/sources.example.yml --output-dir data/raw/onion
qr-assay run --config configs/standard.yml
```

See `docs/data-sources.md` and `docs/source-catalog.md` for provenance rules and the precise meaning of “all”.

## Confirmatory vs exploratory runs

`configs/full.yml` is the cleanest confirmatory corpus comparison:

```text
surface granularity = origin
onion granularity   = origin
scheme policy       = strip
masks               = all 8
```

`configs/standard.yml` retains paths/queries in both corpora and is more vulnerable to host/crawl dependence. Publication inference for URL-granularity data should use host/crawl-clustered resampling; the built-in normal-approximation p-values are secondary diagnostics only.

## Compute warning after hardening

Mask factorialization increases the row count by 8× relative to the original v0.1 design.

| profile | matches | payloads | base QRs (8 masks) | core transform rows |
|---|---:|---:|---:|---:|
| smoke | 100 | 400 | 3,200 | 25,600 |
| standard | 100,000 | 400,000 | 3.2 million | 25.6 million |
| full target | 1.5 million | 6 million | 48 million | 384 million |

Do **not** launch the full target from the old v0.1 timing estimates. Benchmark the exact machine and run the falsification battery first. At full scale, feature I/O can dominate QR arithmetic.

```bash
qr-assay benchmark --count 10000 --workers 0
```

## Outputs

Each run directory contains:

- `payloads.jsonl.gz` — matched payload metadata/provenance;
- `features.jsonl.gz` — one row per QR mask × transform;
- `analysis.json` — unbalanced and density-balanced summaries;
- `report.md` — concise human-readable result;
- `run_manifest.json` — full config, environment, hashes, timings and QC;
- `examples/` — a few PNGs for visual inspection.

Raw corpora and generated results are gitignored. Publish source manifests, checksums, configuration, code commit and run manifest—not live crawl content—unless upstream licensing explicitly permits redistribution.

## Reproducibility contract

- fixed RNG seed in every config;
- exact source inventory and SHA-256 when enabled;
- bounded-memory reservoir sampling;
- symmetric surface/onion granularity controls;
- explicit scheme policy;
- forced QR byte mode;
- every payload under every configured mask;
- feature-file SHA-256;
- density/inversion/matched-mask QC gates;
- paired effects computed once per `match_id` after mask averaging;
- axial orientation represented with `cos(2θ)` and `sin(2θ)`;
- effect sizes and intervals are primary; p-values are secondary.

## Before a million-scale run

Run small planted worlds first. The assay should stay null when both corpora come from the same process and recover deliberately planted unigram, ordering, mask and orientation differences. The required falsification battery is specified in `docs/protocol.md`.

Only after stage 1 survives those checks should a stage-2 model assay attach continuous head/expert carrier vectors and test actual neural translation, stretching, rotation, hysteresis or cyclicity.
