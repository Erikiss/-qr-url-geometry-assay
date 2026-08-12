# Natural URL source catalog

The word **all** is scoped to pinned, declared source snapshots. Onion v3 was
designed to resist enumeration, so no public list is a ground truth for the Tor
network.

## Automatically acquirable sources

| source | role | contents | acquisition |
|---|---|---|---|
| [DomainsProject](https://domainsproject.org/dataset) | natural surface control | apex domains/hostnames, one per line | obtain the chosen upstream snapshot, then place files in `data/raw/surface/` |
| [Real-World Onion Sites](https://github.com/alecmuffett/real-world-onion-sites) `master.csv` | current natural onion stratum | curated mainstream/social-good onion URLs | `configs/sources.example.yml` |
| Real-World Onion Sites CT log | historical/current certificate stratum | onion names observed in certificate transparency | `configs/sources.example.yml` |
| [SecureDrop directory](https://securedrop.org/api/v1/directory/) | current natural onion stratum | news-organization submission services | `configs/sources.example.yml` |
| [Tor Project service list](https://onion.torproject.org/) | current + explicitly marked legacy stratum | services operated by the Tor Project | `configs/sources.example.yml` |

All automated onion acquisitions are ordinary clearnet HTTPS downloads. The
extractor only retains URL strings; it never follows an onion URL.

## Historical research corpora requiring manual access

These studies are part of the provenance plan but are not silently mirrored or
scraped because a stable, license-compatible bulk URL download was not verified.

| corpus/study | reported scale | access status | repository action |
|---|---:|---|---|
| [DUTA-10K / ToRank](https://gvis.unileon.es/datasets-duta-10k/) | 10,367 unique onion addresses | available from GVIS, currently upon request | after approval, place the received export in `data/raw/onion/duta10k/` and retain its license/checksum |
| [Sanchez-Rola et al.](https://www.eurecom.fr/en/publication/5152/download/sec-publi-5152.pdf), *A Comprehensive Structure and Privacy Analysis of Tor Hidden Services* | about 1.5M URLs across 7,257 onion domains | paper verified; no stable public bulk URL artifact verified | import an author-provided export only |
| [Dizzy](https://doi.org/10.1145/3600160.3600167), *Large-Scale Crawling and Analysis of Onion Services* | 63.2M pages across 39,536 onion domains | paper verified; no stable public bulk URL artifact verified | import an author-provided URL list only, not page bodies |
| [Ahmia](https://github.com/ahmia) | continuously updated search index | open-source crawler/index; no stable public bulk snapshot included | accept a documented index export, never scrape query results as if exhaustive |

For a publication run, list excluded/request-pending corpora in the report. Do
not substitute unversioned “dark web links” pages, live-site probes, or active
Tor crawling: they change the sampling frame and introduce content/safety
confounds.

## Snapshot lock

`qr-assay acquire` records SHA-256 for every download. During `prepare`, every
raw input is inventoried again by path, byte size, mtime, and SHA-256. Keep both
the acquisition output and `run_manifest.json` with the report.
