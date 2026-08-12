# Data-source and provenance rules

## Surface corpus

DomainsProject is the intended large source. Its public files are host/domain strings; the pipeline normalizes them and can treat them as either `origin` or `url` units.

For the confirmatory full comparison, surface and onion use the **same unit: `origin`**. Bare surface domains therefore become host-level payloads, while onion crawl paths are collapsed to their onion host. The QR payload then follows the configured `sampling.scheme_policy`; the confirmatory run uses `strip` so `https://` versus `http://` cannot create the result.

Record the exact snapshot, filenames, upstream license and checksums. The DomainsProject collection is too large to vendor here.

## Natural onion corpus

Combine multiple historical and current crawl exports. Accepted inputs include CSV, JSONL, logs, compressed text or arbitrary text containing v2/v3 onion URLs.

For every source record:

- project/paper and canonical source URL;
- snapshot/crawl dates;
- file SHA-256;
- whether entries are hosts only or full paths;
- license/redistribution terms;
- known filtering policy.

The pipeline reads URL strings only. It must not fetch the referenced onion service, test liveness or ingest page bodies. Automated acquisition is restricted to ordinary clearnet HTTPS endpoints declared in the source manifest.

## Deduplication

The v0.2 hardening separates two cases:

- **origin granularity:** raw crawl paths are not globally deduplicated first; they are immediately canonicalized to the selected host/origin payload and deduplicated with a 128-bit BLAKE2b key. This avoids storing millions of redundant path URLs merely to throw them away later.
- **url granularity:** if source deduplication is enabled, normalized URL strings are deduplicated with 128-bit keys. Multi-million crawl unions use a SQLite-backed key table by default rather than a Python in-memory set.

The original v0.1 64-bit hash key is retained only for backward compatibility and is not used for exhaustive deduplication.

Surface canonicalization is also deduplicated after the requested granularity/scheme policy. DomainsProject is already expected to be unique, so this should not alter its intended population while protecting mixed-source runs.

## Sampling and the meaning of “all”

`configs/full.yml` scans every provided source and asks for up to 1.5 million **matched origin pairs**. If fewer byte-length-matched surface/onion origins exist, the manifest records `complete: false` and the achieved count.

“All onion sites” therefore means every unique onion hostname observed in the pinned, declared source snapshots—not an enumerable ground truth for the Tor network. Onion services cannot be exhaustively enumerated from the protocol itself.

Use `granularity: url` only for the separate path/query experiment. Because one host may contribute many paths, URL-granularity publication inference needs host/crawl-clustered resampling; it is not exchangeable at the raw URL-row level.

## Matching changes the estimand

Exact byte-length matching is a deliberate conditional design. An origin-level result means:

> difference between surface and onion QR geometry **among payloads that share the observed matched byte-length support**.

It is not an unconditional population estimate of “the surface web versus Tor”. Report the source counts before matching, the overlap support and the achieved matched sample.

## Snapshot lock

`qr-assay acquire` records SHA-256 for every declared download when provided. During `prepare`, every raw input is inventoried again by path, byte size, modification time and SHA-256 (unless hashing is explicitly disabled for exploratory scans). Keep the acquisition record, effective YAML, git commit and `run_manifest.json` with every report.
