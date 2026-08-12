from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.6g}"


def _cluster_table(contrasts: dict[str, Any], level: str) -> list[str]:
    lines = [
        (
            "| contrast | metric | matches | clusters | effective clusters | max cluster | "
            "mean Δ | CR1 95% CI | stable-count flag |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    key = f"{level}_clustered"
    for contrast, result in sorted(contrasts.items()):
        for row in result[key]:
            low = row["ci95_low_normal_cr1"]
            high = row["ci95_high_normal_cr1"]
            interval = (
                f"[{_fmt(low)}, {_fmt(high)}]" if low is not None and high is not None else "NA"
            )
            lines.append(
                f"| {contrast} | {row['metric']} | {row['n_matches']:,} | "
                f"{row['cluster_count']:,} | {row['effective_cluster_count']:.2f} | "
                f"{row['max_cluster_size']:,} | {_fmt(row['mean_difference'])} | {interval} | "
                f"{'OK' if row['cluster_count_ge_20'] else 'LOW CLUSTER COUNT'} |"
            )
    return lines


def write_confirmatory_report(
    config: dict[str, Any],
    preparation: dict[str, Any],
    null_qc: dict[str, Any],
    clustered: dict[str, Any],
    *,
    descriptive_report: Path,
    codeword: dict[str, Any] | None = None,
    bitstream: dict[str, Any] | None = None,
    spatial_null: dict[str, Any] | None = None,
) -> Path:
    """Write the primary human-readable report without promoting naive row-wise p-values."""
    output_dir = Path(config["outputs"]["directory"])
    report_path = output_dir / "confirmatory_report.md"
    surface_granularity = str(config["sources"]["surface"].get("granularity", "url"))
    onion_granularity = str(config["sources"]["onion"].get("granularity", "url"))
    support = preparation.get("matching", {})
    selected_by_length = support.get("selected_by_length", {})
    support_text = ", ".join(
        f"{length} B: {count:,}" for length, count in selected_by_length.items()
    )
    if not support_text:
        support_text = "not available"

    lines = [
        "# QR URL Geometry Assay — confirmatory report",
        "",
        (
            "This is the primary run-level inference report. The legacy/descriptive report is "
            "retained for QC and sensitivity inspection, but its normal-approximation p-values "
            "are not the publication-level inferential target."
        ),
        "",
        "## Locked estimand",
        "",
        f"- Surface granularity: **{surface_granularity}**",
        f"- Onion granularity: **{onion_granularity}**",
        f"- Matched quadruples: **{preparation['matched_pairs']:,}**",
        f"- Length weighting: **{support.get('length_weighting', 'unknown')}**",
        f"- Selected shared byte-length support: **{support_text}**",
        f"- Synthetic null: **{preparation['synthetic_mode']}**",
        (
            f"- QR masks per payload: **{len(config['qr']['masks'])}**, averaged before the "
            "match-level effect"
        ),
        "",
    ]

    if surface_granularity == onion_granularity == "origin":
        lines.extend(
            [
                (
                    "**Origin-arm interpretation.** This is a conditional hostname/"
                    "cryptographic-identifier control on the byte lengths represented in both "
                    "corpora. It is not an unconditional ‘surface web versus Tor’ estimate and "
                    "it is not the host-neutral grammar assay."
                ),
                "",
            ]
        )
    elif surface_granularity == onion_granularity == "path_query":
        lines.extend(
            [
                (
                    "**Grammar-arm interpretation.** Hosts and schemes were removed before "
                    "matching/encoding. Natural host/source hashes remain only as dependency/"
                    "provenance clusters."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Structural-null QC",
            "",
            f"Hard null invariants: **{'PASS' if null_qc['passed_hard_invariants'] else 'FAIL'}**",
            "",
            (
                "| corpus | pairs | unchanged controls | unchanged fraction | unique synthetic "
                "fraction | non-ASCII natural fraction |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for corpus, row in sorted(null_qc["corpora"].items()):
        lines.append(
            f"| {corpus} | {row['pairs_compared']:,} | {row['unchanged_controls']:,} | "
            f"{row['unchanged_fraction']:.4f} | {row['unique_synthetic_fraction']:.4f} | "
            f"{row['non_ascii_natural_fraction']:.4f} |"
        )

    lines.extend(
        [
            "",
            (
                "Unchanged controls are retained. A high unchanged fraction means the declared "
                "null is weak or degenerate for that corpus; it is not silently repaired by "
                "switching null families."
            ),
            "",
            "## Confirmatory natural-vs-null effects — host-clustered",
            "",
            (
                "Effects are computed after averaging all configured QR masks within each "
                "payload. CR1 uncertainty is clustered by the natural parent host. No row-wise "
                "mask pseudoreplication is used."
            ),
            "",
            *_cluster_table(clustered["contrasts"], "host"),
            "",
            "## Source-file-cluster sensitivity",
            "",
            (
                "This second uncertainty calculation clusters by the content hash of the natural "
                "source file. A file is not automatically equivalent to an independent crawl; "
                "if only one or a few source files are supplied, the source-level interval is "
                "non-estimable or flagged as low-cluster-count."
            ),
            "",
            *_cluster_table(clustered["contrasts"], "source"),
            "",
            (
                "The direct `onion_natural - surface_natural` contrast is intentionally omitted "
                "from one-way clustered confirmatory inference because the two sides carry "
                "different host/source clusters. It remains descriptive unless a separate "
                "multi-way procedure is preregistered."
            ),
            "",
        ]
    )

    if bitstream is not None:
        lines.extend(
            [
                "## Pre-spatial bitstream negative control",
                "",
                f"Diagnostic file: `{Path(bitstream['output']).name}`",
                "",
                (
                    "Raw UTF-8 bits, QR data-codeword bits and Reed–Solomon parity bits are "
                    "analyzed before interpreting their 2D placement. A natural-vs-null "
                    "distinction already present in these streams cannot be attributed to the QR "
                    "spatial map itself."
                ),
                "",
            ]
        )
    if spatial_null is not None:
        lines.extend(
            [
                "## Exact-bit spatial-placement null",
                "",
                f"Diagnostic file: `{Path(spatial_null['output']).name}`",
                "",
                (
                    "Within each data-codeword or RS-ECC region, the observed unmasked bits are "
                    "randomly reassigned to the same coordinates. The bit multiset and therefore "
                    "one-density are preserved exactly. Natural and synthetic members of one "
                    "pair use common random placements to avoid Monte-Carlo noise in the paired "
                    "contrast. A surviving residual is placement-specific 2D structure beyond "
                    "exact bit composition, not by itself semantic or emergent geometry."
                ),
                "",
            ]
        )
    if codeword is not None:
        lines.extend(
            [
                "## Data-codeword versus Reed–Solomon spatial diagnostic",
                "",
                f"Diagnostic file: `{Path(codeword['output']).name}`",
                "",
                (
                    "The configured QR mask is removed before geometry is computed separately "
                    "over data-codeword and RS-ECC module regions. The data-codeword region "
                    "includes QR framing/padding and must not be called pure payload."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            (
                "This stage measures properties of encoded stimulus matrices relative to declared "
                "structural nulls. It does not establish semantic representations, attention-head/"
                "expert carriers, spontaneous rotation, hysteresis, or cyclicity inside a neural "
                "model. Those require the later model/checkpoint causal assay."
            ),
            "",
            "## Secondary artifacts",
            "",
            f"- Descriptive/QC report: `{descriptive_report.name}`",
            "- Full paired diagnostics: `analysis.json`",
            "- Cluster-robust core inference: `cluster_analysis.json`",
            "- Structural-null QC: `null_qc.json`",
        ]
    )
    if bitstream is not None:
        lines.append("- Pre-spatial bitstream baseline: `bitstream_analysis.json`")
    if spatial_null is not None:
        lines.append("- Exact-bit placement null: `spatial_null_analysis.json`")
    if codeword is not None:
        lines.append("- Codeword-region diagnostic: `codeword_analysis.json`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
