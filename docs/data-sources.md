# Data-source and provenance rules

## Surface corpus

DomainsProject is the intended large source. Its public repository documents
plain-text, one-domain-per-line files after Git LFS/XZ unpacking. This pipeline
canonicalizes a bare domain to `https://<domain>/` and streams compressed or
uncompressed inputs without loading the corpus into memory.

Record the exact snapshot, filenames, upstream license, and checksums. The
current DomainsProject collection is too large to vendor in this repository.

## Natural onion corpus

Combine multiple historical and current crawl exports, then deduplicate the URL
strings. Accepted sources include CSV, JSONL, logs, or text embedded with v2/v3
onion URLs. The included example manifest starts with the curated
`real-world-onion-sites` list; it is a pipeline test/source stratum, not a
substitute for broad historical crawls.

For each source, record:

- project/paper and canonical URL;
- snapshot or crawl dates;
- file SHA-256;
- whether entries are hosts only or full paths;
- license/redistribution terms;
- known filtering policy.

The pipeline reads URL strings only. It must not fetch the referenced onion
service, test liveness, or ingest page bodies. Source acquisition must occur
over an ordinary clearnet HTTPS endpoint declared in the manifest.

## Deduplication

Onion sources are deduplicated after canonicalization because crawl overlap is
expected. Surface deduplication defaults off for DomainsProject because the
upstream lists are sorted/unique and a global in-memory hash set would be
wasteful; enable it for mixed surface sources.

## Sampling and the meaning of “all”

`configs/full.yml` scans every provided onion source and asks for up to 1.5
million matched URL units. If fewer byte-length-matched surface/onion pairs are
available, the manifest records `complete: false` and the achieved count.

“All onion URLs” therefore means all URLs in the pinned, declared source
snapshots—not an enumerable ground truth for the Tor network. Onion services
cannot be exhaustively enumerated from the protocol itself.

For the site-level exhaustive run, `granularity: origin` instead means every
unique natural onion hostname observed in those snapshots, regardless of how
many crawl paths point into that site. Use `granularity: url` for the separate
path/query grammar experiment.
