# Data-source and provenance rules

## Two different surface populations

Do not reuse one source merely because both arms concern URLs.

### Origin / cryptographic-host control

DomainsProject is the intended large surface source for the origin-level control. Its public files are host/domain strings, which is exactly the required unit for this arm.

For the origin comparison, surface and onion use the **same unit: `origin`**. Bare surface domains become host-level payloads, while onion crawl paths are collapsed to their onion host. `sampling.scheme_policy: strip` removes the arbitrary `https://` versus `http://` prefix.

This comparison must not be described as a grammar test. A v3 onion hostname is a self-authenticating cryptographic identifier, so surface-origin versus onion-origin primarily contrasts human-selected domain strings with cryptographic host identifiers. `onion_natural` versus a checksum-valid synthetic v3 onion at origin level is useful as a cryptographic-host negative/control.

Record the exact DomainsProject snapshot, filenames, upstream license and checksums. The collection is too large to vendor here.

### Host-neutral path/query grammar arm

`configs/grammar.yml` requires a **different surface source containing observed full URLs**, stored under `data/raw/surface_urls` by default. A hostname-only/domain dump is invalid input because it contains no observed natural path or query structure.

Both corpora use `granularity: path_query`. The QR payload is therefore only:

```text
/path/components?query=values
```

The surface domain and `.onion` host are removed before byte-length matching and QR encoding. Their hashes are retained separately as cluster/provenance metadata. This makes the grammar arm ask about suffix sequence structure rather than the trivial visible difference between ordinary DNS names and v3 onion identifiers.

The surface-URL source must be a pinned crawl/index export with explicit snapshot date, license and checksum. Do not synthesize paths from DomainsProject hostnames and call them natural surface URLs.

## Natural onion corpus

Combine multiple historical and current crawl exports. Accepted inputs include CSV, JSONL, logs, compressed text or arbitrary text containing onion URLs.

For every source record, retain:

- project/paper and canonical source URL;
- snapshot/crawl dates;
- file SHA-256;
- whether entries are hosts only or full paths;
- license/redistribution terms;
- known filtering policy.

The pipeline reads URL strings only. It must not fetch the referenced onion service, test liveness or ingest page bodies. Automated acquisition is restricted to ordinary clearnet HTTPS endpoints declared in the source manifest.

### Onion version rule

The confirmatory population is **v3 only** (`sources.onion.versions: [3]`). v2 addresses are still parseable for historical sensitivity work but are excluded by default because their 16-character hostname format is structurally different from the 56-character v3 format. Mixing v2 and v3 would confound service generation with address format/era.

A 56-character candidate is accepted only if its decoded version byte and self-authenticating v3 checksum are valid.

## Deduplication

The hardened pipeline separates three cases:

- **origin:** canonicalize immediately to host/origin and deduplicate the effective payload with a 128-bit BLAKE2b key;
- **url:** deduplicate normalized full URL strings when enabled; multi-million unions use a SQLite-backed key table by default;
- **path_query:** deduplicate on `(natural host, effective path/query payload)`, not on suffix alone. Repeated `/login` records from one host collapse, while the same `/login` suffix on independent hosts remains as separate clustered observations.

The original v0.1 64-bit helper is retained only for backward compatibility and is not used for exhaustive deduplication.

## Matching and the estimand

Exact UTF-8 byte-length matching is a deliberate conditional design. The default `sampling.length_weighting: overlap` allocates matched pairs at length `l` in proportion to:

```text
min(n_surface,l, n_onion,l)
```

subject to the retained per-length reservoir capacity. This estimates the empirical shared-support population instead of giving a rare byte length the same influence as a common one.

`length_weighting: equal_length` remains an explicit sensitivity mode only. It deliberately changes the estimand toward equal weighting of byte-length strata.

The manifest records empirical overlap, retained reservoir capacity, and selected counts for every matched byte length. If reservoir caps bind, that must be visible before interpretation.

An origin-level result therefore means:

> difference between surface and v3-onion origin QR geometry among payloads in their observed shared byte-length support, under the declared overlap weighting.

A grammar-arm result means:

> difference in host-neutral path/query QR geometry among observed surface and v3-onion suffixes in their shared byte-length support, relative to the declared structural null.

Neither is an unconditional population estimate of “the surface web versus Tor”.

## Sampling and the meaning of “all”

`configs/full.yml` scans every provided source and asks for up to 1.5 million matched **origin pairs**. If fewer supported pairs are available, the manifest records `complete: false` and the achieved count.

“All onion sites” therefore means every unique valid v3 onion hostname observed in the pinned, declared source snapshots—not an enumerable ground truth for the Tor network. Onion services cannot be exhaustively enumerated from the protocol itself.

The host-neutral grammar arm is intentionally not “all sites”: it samples observed path/query records and must retain host/crawl cluster IDs because one service can contribute many suffixes.

## Snapshot lock

`qr-assay acquire` records SHA-256 for every declared download when provided. During `prepare`, every raw input is inventoried again by path, byte size, modification time and SHA-256 unless hashing is explicitly disabled for exploratory scans.

Keep the acquisition record, effective YAML, git commit and `run_manifest.json` with every report. Synthetic rows inherit the natural parent host and source hashes so later clustered inference cannot accidentally treat generated controls as independent sources.
