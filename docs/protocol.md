# Experimental protocol

## Scope

This repository is **stage 1: stimulus geometry calibration**. It asks whether QR matrices produced from URL-derived byte strings retain detectable lexical/sequence structure after obvious QR-encoding confounds are controlled.

It does **not** measure payload semantics, neural representations, attention-head carriers, carrier rotation, or spontaneous cyclicity. The QR encoder sees bytes. Claims about model semantics or internal dynamics belong to a later model/checkpoint assay.

## Two primary experimental arms

### A. Origin / cryptographic-host control

The origin arm uses `origin × origin`, strips schemes, and restricts onion input to v3 by default.

Its direct surface-versus-onion contrast is **not a Chomsky/grammar test**. A v3 onion hostname is a self-authenticating cryptographic identifier, whereas a surface domain is generally human-selected. This arm characterizes that deliberately strong structural contrast and supplies a useful cryptographic-host control.

At v3 origin level, `onion_natural` versus a newly generated checksum-valid v3 onion is expected to be a demanding negative/control: the linguistic null ladder largely collapses because the host label itself is not a natural-language token.

### B. Host-neutral path/query grammar arm

`configs/grammar.yml` uses `path_query × path_query`. Hosts and schemes are removed before matching and QR encoding, leaving only observed path/query material. Natural parent host and crawl hashes remain as cluster metadata.

This is the arm relevant to the grammar/sequence-structure hypothesis. It requires observed full URLs on both sides; a host-only source such as DomainsProject cannot supply the surface grammar population.

The confirmatory estimand is:

> among observed surface and v3-onion path/query payloads in their shared UTF-8 byte-length support, under overlap-weighted matching and after averaging all eight QR masks, how much QR geometry remains associated with natural sequence structure relative to a declared structural null?

## Unit of analysis

One `match_id` contains four payloads with identical UTF-8 byte length:

| class | observed/generated | role |
|---|---|---|
| `surface_natural` | observed | surface natural |
| `onion_natural` | observed | onion natural |
| `surface_synthetic` | generated from its natural partner | declared surface null |
| `onion_synthetic` | generated from its natural partner | declared onion null |

All configured QR masks are generated for **every payload**. Mask is a within-payload nuisance factor, not a between-payload assignment. Paired inference averages geometry over masks before computing one contrast per `match_id`; masks must never be counted as independent replicates.

Synthetic controls inherit `natural_host_sha256` and `natural_source_sha256` from their observed parent. This preserves the correct dependency structure for later host/crawl-clustered inference.

## Structural-null ladder

`sampling.synthetic_mode` selects one null per run. Claims should be repeated across the ladder rather than relying on one synthetic family:

1. `token_shuffle` — preserves token-level character multisets where meaningful;
2. `class_permute` — preserves global ASCII unigram counts within lower/upper/digit classes while destroying positions;
3. `grammar_random` — preserves delimiters and character-class positions but resamples token characters.

For v3 onion host labels, all three modes replace the cryptographic label with another checksum-valid v3 address. At **origin** level the linguistic ladder therefore intentionally collapses. In the **path_query** arm the host is absent, so the selected null operates on path/query material normally.

A natural-vs-synthetic difference may be called a **lexical/sequence-structure effect relative to the declared null**. It must not be called semantic without a model-based semantic assay.

## Onion version rule

Confirmatory runs use `sources.onion.versions: [3]`. Legacy v2 addresses remain parseable only for explicitly labeled historical sensitivity analyses.

A v3 candidate must have the correct 56-character Base32 form, decoded version byte, and self-authenticating checksum. A random 56-character Base32 string is not accepted merely because its syntax resembles v3.

## Payload canonicalization

Granularity and scheme treatment are explicit factors:

- `url` — host plus path/query;
- `origin` — host only;
- `path_query` — path/query only, host removed;
- `sampling.scheme_policy` — `strip`, `https`, or `preserve` for host-bearing arms.

Mixed surface/onion granularities are exploratory only.

For `path_query`, deduplication uses `(natural host, path/query)` rather than suffix alone. Identical suffixes repeated on the same host collapse; the same suffix observed independently on different hosts remains as separate clustered observations.

## Matching and population weighting

Exact UTF-8 byte-length matching is mandatory. It changes the estimand to the shared observed support and must be reported as such.

The default `sampling.length_weighting: overlap` assigns target mass to byte-length stratum `l` proportional to:

```text
min(n_surface,l, n_onion,l)
```

subject to retained reservoir capacity. This avoids the previous equal-length round-robin behavior, which could greatly over-weight rare lengths.

`length_weighting: equal_length` is retained only as an explicit sensitivity estimand. The manifest records empirical overlap, available retained pairs and selected counts by byte length.

## QR encoding controls

Every matched unit is controlled on:

- UTF-8 byte length;
- **explicit QR byte mode** via `QRData(..., mode=MODE_8BIT_BYTE)`;
- error-correction level (`M` by default);
- QR version;
- all configured masks, factorialized within payload.

`optimize=0` alone is not treated as sufficient mode control because the encoder can otherwise still choose numeric/alphanumeric mode from the content.

## Feature regions

Every QR currently produces two primary parallel feature batteries:

1. **full matrix** — finder, timing, alignment, format/version information and mapped modules together;
2. **data/ECC/remainder modules only** — fixed QR function patterns removed before geometry is computed.

The mapped-module region is primary; the whole matrix is a sensitivity/control analysis because fixed finder/timing structures can dominate low-order moments while carrying no payload-specific information.

### Data-codeword versus Reed–Solomon decomposition

The implementation now also contains a tested structural decomposition of mapped modules into:

- data-codeword modules;
- Reed–Solomon ECC-codeword modules;
- remainder modules.

This follows the pinned `python-qrcode==8.2` codeword interleaving and `map_data` traversal. CI verifies that the three regions are disjoint and exactly partition the free mapped-module region across ECC levels.

The data-codeword region must **not** be mislabeled “pure payload”: it also contains mode/count framing, terminator/alignment padding and QR pad codewords. The decomposition is currently an instrumented diagnostic; it should be promoted to a primary feature battery only after the compact diagnostic arm is implemented without multiplying the full 48M-row run unnecessarily.

The code also tests that undoing QR mask functions on the mapped region recovers the same underlying mapped bitstream for one payload across all eight masks.

## Confirmatory versus deterministic calibration transforms

The corpus-scale confirmatory arms use only the **identity image state**. Rotation, reflection and bit inversion are deterministic matrix operations and are not inferential replicates.

A separate, much smaller equivariance calibration uses:

- rotations `0°, 90°, 180°, 270°`;
- exact square-symmetry group `D4 = {r0, r90, r180, r270, mh, mv, md, ma}` where configured;
- polarity `normal` versus exact bit complement `Q' = 1 - Q`.

A deterministic `np.rot90` orbit is cyclic by construction. It is only a calibration of feature equivariance, never evidence of spontaneous neural cyclicity.

Nearest-neighbour pixel scaling is deferred from stage 1. Repeating QR modules by a factor of two is mathematically tautological at the matrix level; image-scale sensitivity becomes meaningful when a vision/model stage introduces a nontrivial renderer, tokenizer or encoder.

## Observable families

The predeclared core metrics are computed in the mapped data/ECC/remainder region; corresponding full-matrix quantities are controls.

### Radial / stretching-like

- `data_radial_mean`;
- `data_radial_std`;
- `data_cov_trace`.

### Translation-like

- normalized centroid `(data_centroid_x, data_centroid_y)`;
- `data_centroid_radius`.

### Axial orientation

- `data_anisotropy`;
- `data_orientation_cos2`;
- `data_orientation_sin2`.

`data_principal_angle_deg` is descriptive only. Principal-axis orientation is axial modulo 180°, so linear angle subtraction is invalid around the wrap boundary. Inferential comparisons use the doubled-angle embedding.

### Texture controls

- `data_transition_h`;
- `data_transition_v`.

Whole-matrix symmetry metrics are retained as calibration controls.

## Inference

The primary paired inference unit is **one `match_id` after averaging identity-state geometry over all configured QR masks**.

The two natural-vs-null contrasts are confirmatory for sequence-structure claims:

- `surface_natural - surface_synthetic`;
- `onion_natural - onion_synthetic`.

The direct `onion_natural - surface_natural` contrast is secondary because its interpretation depends strongly on which experimental arm generated the payloads.

Normal-approximation intervals and Holm-adjusted p-values are secondary diagnostics. For `path_query`/URL data, publication inference must account for repeated observations from the same natural host and crawl/source. Large raw `n` must not convert a numerically trivial effect into a mechanistic claim.

## Falsification and QC

A run is invalid if any of these occur:

1. a `match_id` lacks any of the four payload classes;
2. a payload lacks any configured mask;
3. byte length, QR version or ECC differ inside a match;
4. ordinary rotations/reflections change density when they mathematically should not;
5. inversion does not complement the appropriate density exactly;
6. the eight QR masks fail to recover the same underlying mapped bitstream after unmasking;
7. data-codeword, ECC-codeword and remainder regions overlap or fail to partition mapped modules;
8. a v3 onion hostname fails checksum/version validation;
9. a source file or effective configuration is missing from the manifest;
10. overlap weighting or reservoir truncation cannot be reconstructed from the manifest.

## Required falsification battery before a million-scale run

Small synthetic worlds must precede the expensive corpus run:

- **exact-null world:** two corpora generated by the same process; expected effect 0;
- **unigram-only world:** same grammar with deliberately shifted character frequencies;
- **ordering world:** equal character counts with different local sequence order;
- **D4 calibration world:** known asymmetric matrices transformed by exact group elements;
- **mask-stability world:** same payload under all eight masks and unmasking;
- **function-pattern world:** alter only fixed function modules and verify mapped-region metrics remain unchanged;
- **cluster world:** duplicate effects within hosts and verify clustered uncertainty widens while the point estimate does not spuriously gain independent sample size;
- **length-weight world:** construct known 90:10 shared byte-length support and verify overlap matching preserves approximately 90:10 rather than 50:50.

The assay should identify planted distinctions, stay null when none was planted, and fail loudly when an invariant is broken.

## Boundary to stage 2

A later neural assay should consume locked stimuli and record, per model/checkpoint, a continuous carrier vector such as head/expert causal contribution. Only then can one test whether stimulus transformations or training-time relocation induce translation, stretching, rotation, hysteresis or cyclicity in the model's internal carrier space.
