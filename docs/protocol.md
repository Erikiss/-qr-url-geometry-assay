# Experimental protocol

## Question

Does QR module geometry retain a detectable signature of natural URL structure,
and is that signature different for ordinary surface URLs, natural onion URLs,
and grammar-matched random controls after obvious size/mask confounds are held
constant?

This first-stage repository is deliberately CPU-only. It does not train a
neural network and makes no claim about internal attention carriers. Its job is
to establish the QR stimulus family and the geometric nulls cleanly before a
later model/checkpoint carrier assay is attached.

## Unit of analysis

One `match_id` contains four payloads with identical UTF-8 byte length:

| class | observed/generated | grammar |
|---|---|---|
| `surface_natural` | observed | surface URL |
| `onion_natural` | observed | onion URL |
| `surface_synthetic` | generated from its natural partner | surface URL |
| `onion_synthetic` | generated from its natural partner | onion URL |

The synthetic generator retains schemes, delimiter positions, case classes,
digit positions, URL punctuation, `.onion`, and total byte length. Tokens are
replaced using a deterministic seed. Synthetic v3 onion labels retain the
specified version byte and checksum relation, so “synthetic” does not merely
mean a malformed 56-character hostname. Synthetic addresses are never
resolved.

## Primary factorial design

For each payload:

- rotation: `0°, 90°, 180°, 270°`;
- polarity: `normal, bit-inverted`.

This creates eight transform rows per base QR. The extended, separately
configured design adds five reflection states and two integer scales. Keeping
it separate prevents the confirmatory core from becoming a 48-row expansion.

## Locked controls

Within each match:

- UTF-8 byte length;
- QR version;
- error correction (`M` by default);
- byte-mode encoding (`optimize=0`);
- mask pattern, assigned evenly over masks 0–7.

For the reported balanced analysis, retain equal counts from the intersection
of `(byte length, QR version, ECC, mask, black-density bin)` strata across all
four classes. Report both the unfiltered and balanced estimates.

## Three observable families

### 1. Radial

Measure mean and standard deviation of black-module radius around the symbol
center plus the second-moment covariance. A rotation should preserve normalized
radial quantities; a systematic corpus effect after balancing is the signal.

### 2. Translation

Measure the normalized centroid of black modules. Under a known 90° rotation,
the centroid vector should rotate around the symbol center. A corpus effect is
tested on centroid radius and paired x/y orbits.

### 3. Rotation/cyclicity

Measure principal-axis angle, anisotropy, transition rates, and explicit
0°→90°→180°→270° orbits. Angle is meaningful only alongside anisotropy; nearly
isotropic matrices must not be treated as having a stable orientation.

## Falsification and QC

The run is invalid if any of these occur:

1. any match has unequal byte length, QR version, ECC, or mask;
2. normal transforms change density;
3. inverted density is not exactly `1 - base_density` at module resolution;
4. a source file or effective configuration is missing from the manifest;
5. corpus comparison is reported without the unbalanced and balanced counts.

The experiment does not interpret a classifier's accuracy as semantics. It
first reports exact geometric statistics and controls. A future predictive
model must split by source/crawl and by registrable/onion host to prevent URL
leakage.
