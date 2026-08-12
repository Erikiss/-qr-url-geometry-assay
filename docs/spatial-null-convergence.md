# Spatial-null Monte Carlo convergence

The exact-bit spatial-placement null contains a computational parameter, the number of random placements `K`. `K` is **not** a scientific treatment and must not be chosen because a preferred result appears at one value.

## Locked pilot

Run the convergence assay only after `prepare` has produced the locked payload artifact for the intended spatial-null corpus. The convergence assay reads that artifact directly and does not re-sample URLs while changing `K`.

Recommended first pilot:

```bash
python -m qr_assay.spatial_convergence \
  --config configs/spatial_null.yml \
  --pilot-matches 250 \
  --k 16,32,64,128 \
  --batches 8 \
  --mc-fraction 0.10 \
  --drift-fraction 0.25
```

The same pilot matches are used at every `K`. Independent Monte Carlo batches are used to estimate numerical Monte Carlo variability. Within each batch, the natural and synthetic member of a paired contrast still use common random placements.

## Acceptance rule

For every estimable combination of:

- region (`data_codeword`, `rs_ecc`),
- confirmatory contrast (`surface natural-null`, `onion natural-null`),
- core spatial metric,

let `SE_host` be the host-cluster CR1 sampling standard error computed on the largest tested `K` in batch 0.

A candidate `K` is acceptable only if **all** estimable cells satisfy both:

```text
Monte-Carlo SD of the batch-level effect estimate <= 0.10 * SE_host
```

and

```text
|mean effect at K - mean effect at previous K| <= 0.25 * SE_host
```

The first tested `K` cannot be selected because it has no previous-K drift comparison. The smallest later `K` satisfying both conditions in every estimable cell is selected.

If a cell has fewer than two host clusters or exactly zero host sampling SE, that cell cannot certify `K`. This prevents a degenerate exact-null pilot from producing a false convergence pass merely because all numbers happen to be zero.

## If no K is certified

`NO_K_CERTIFIED` is a valid calibration result. It is **not** evidence for or against the scientific spatial-placement effect.

Proceed in this order:

1. increase the `K` ladder, for example to `32,64,128,256`;
2. if cells remain non-estimable, increase pilot host diversity rather than merely duplicating URLs from the same host;
3. do not relax the precision fractions after inspecting the effect direction;
4. record the complete failed and successful convergence artifacts with the final run.

## Why the criterion uses sampling uncertainty

The goal is not to make Monte Carlo error literally zero. It is to make numerical random-placement error small relative to the uncertainty from the observational units that define the scientific estimate. A `K` that is adequate for a large, noisy effect may therefore differ from a `K` required for a very precisely estimated effect.

The convergence assay chooses **computational precision only**. Statistical or mechanistic significance is evaluated later from the locked scientific analysis, never from whether a particular `K` converged quickly.
