# Experimental protocol

## Scope and estimand

This repository is **stage 1: stimulus geometry calibration**. It asks whether the QR matrices produced from natural URL strings retain detectable lexical/sequence structure after the obvious QR encoding confounds are controlled.

It does **not** measure payload semantics, neural representations, attention-head carriers, or carrier rotation. The QR encoder only sees bytes. Claims about semantics or internal model dynamics belong to a later model/checkpoint assay.

The primary confirmatory estimand is conditional:

> among surface and onion payloads matched on byte length, QR version, ECC and then averaged across all eight QR masks, how much geometric structure remains in natural strings relative to a declared structural null?

The full run uses `origin` granularity for **both** surface and onion sources and strips schemes before matching. The URL/path experiment is separate and uses `url` granularity for both corpora.

## Unit of analysis

One `match_id` contains four payloads with identical UTF-8 byte length:

| class | observed/generated | role |
|---|---|---|
| `surface_natural` | observed | surface natural |
| `onion_natural` | observed | onion natural |
| `surface_synthetic` | generated from its natural partner | declared surface null |
| `onion_synthetic` | generated from its natural partner | declared onion null |

All configured QR masks are generated for **every payload**. Mask is therefore a within-payload nuisance factor, not a between-payload assignment. Paired inference first averages the primary geometry over masks and then computes one contrast per `match_id`; masks must never be counted as independent replicates.

## Structural-null ladder

`samping.synthetic_mode` selects one null per run. Publication claims should be repeated across the ladder rather than relying on one synthetic family:

1. `token_shuffle` — preserves token-level character multisets where meaningful;
2. `class_permute` — preserves global ASCII unigram counts within lower/upper/digit classes while destroying positions;
3. `grammar_random` — preserves delimiters and character-class positions but resamples token characters.

For a v3 onion hostname, all three modes replace the cryptographic host label with another checksum-valid v3 address. That host is not a natural-language token, so the linguistic null ladder intentionally collapses at origin level. Path/query material still follows the selected null mode.

A natural-vs-synthetic difference may therefore be called a **lexical/sequence-structure effect relative to the declared null**. It must not be called semantic without a model-based semantic assay.

## Payload canonicalization

Granularity and scheme treatment are explicit factors:

- `sources.surface.granularity`: `origin` or `url`;
- `sources.onion.granularity`: `origin` or `url`;
- `sampling.scheme_policy`: `strip`, `https`, or `preserve`.

The confirmatory full run uses `origin × origin` and `scheme_policy: strip`, removing the arbitrary `https://` versus `http://` prefix difference. The path/query experiment uses `url × url`; mixed granularity is exploratory only.

## QR encoding controls

Every matched unit is controlled on:

- UTF-8 byte length;
- forced byte-mode encoding (`optimize=0`);
- error-correction level (`M` by default);
- QR version;
- all configured masks, factorialized within payload.

The density-balanced analysis additionally intersects classes on `(byte length, QR version, ECC, mean-density bin)`, where mean density is averaged across masks for the payload.

## Geometric calibration factors

The core design uses:

- rotation: `0°, 90°, 180°, 270°`;
- polarity: `normal, bit-inverted`.

These transformed rows are **equivariance/QC calibration**, not evidence of spontaneous cyclic dynamics. A deterministic `np.rot90` orbit is cyclic by construction.

The extended design uses the exact eight elements of the square symmetry group `D4` rather than a redundant rotation × reflection cross-product:

`r0, r90, r180, r270, mh, mv, md, ma`.

Scale is configured separately. Bit inversion is the exact involution `Q' = 1 - Q`; it is not geometric circle inversion.

## Observable families

### Radial / stretching-like

- `radial_mean`;
- `radial_std`;
- covariance trace.

### Translation-like

- normalized black-module centroid `(centroid_x, centroid_y)`;
- centroid radius.

### Axial orientation

- anisotropy;
- `orientation_cos2`;
- `orientation_sin2`.

`principal_angle_deg` is descriptive only. Principal-axis orientation is axial modulo 180°, so linear angle subtraction is invalid around the wrap boundary. Inferential comparisons use the doubled-angle embedding.

### Texture / symmetry controls

- horizontal and vertical transition rates;
- 180° rotational similarity;
- horizontal and vertical reflection similarity.

## Inference

The primary paired inference unit is **one `match_id` after averaging the primary geometry over all configured QR masks**. Masks and deterministic rotations are repeated measures, not independent observations.

The repository still reports normal-approximation intervals and Holm-adjusted p-values as secondary diagnostics. For publication:

- effect sizes and intervals are primary;
- URL-granularity runs require host/crawl-clustered resampling because repeated paths from one host are not independent;
- the origin-level run is the cleaner confirmatory analysis;
- a large `n` must not turn a numerically tiny effect into a mechanistic claim.

## Falsification and QC

A run is invalid if any of these occur:

1. a `match_id` lacks any of the four payload classes;
2. a payload lacks any configured mask;
3. byte length, QR version, or ECC differ inside a match;
4. normal rotations/reflections/scales change module density;
5. inverted density is not exactly `1 - base_density` at module resolution;
6. a source file or effective configuration is missing from the manifest;
7. corpus comparison is reported without unbalanced and density-balanced counts.

## Required falsification battery before the million-scale run

Before treating a full-corpus result as scientific evidence, run small synthetic worlds with known ground truth:

- exact-null world: two corpora generated by the same process, expected effect 0;
- unigram-only world: matched grammar with deliberately shifted character frequencies;
- ordering world: equal character counts with different local sequence order;
- orientation-QC world: known asymmetric matrices transformed by exact `D4` elements;
- mask-stability world: the same payload under all eight masks.

The assay should identify the planted distinction and stay null when none was planted.

## Boundary to stage 2

A later neural assay should consume these locked stimuli and record, per model/checkpoint, a continuous carrier vector such as head/expert causal contribution. Only then can one test whether stimulus rotation, reflection, inversion, or training-time relocation induces translation, stretching, rotation, hysteresis, or cyclicity in the model's internal carrier space.
