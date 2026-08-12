from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.6g}"


def _did_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        (
            "| metric | mean DiD | Surface clusters | Onion clusters | SE | Holm p | "
            "simultaneous Bonferroni 95% CI | strict pass |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        low = row["bonferroni_simultaneous_ci_low"]
        high = row["bonferroni_simultaneous_ci_high"]
        interval = f"[{_fmt(low)}, {_fmt(high)}]" if low is not None and high is not None else "NA"
        lines.append(
            f"| {row['metric']} | {_fmt(row['mean_difference_in_differences'])} | "
            f"{row['surface_cluster_count']:,} | {row['onion_cluster_count']:,} | "
            f"{_fmt(row['independent_cluster_standard_error'])} | "
            f"{_fmt(row['p_holm_familywise'])} | {interval} | "
            f"{'PASS' if row['strict_confirmatory_pass'] else 'NO'} |"
        )
    return lines


def _familywise_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        (
            "| contrast | metric | clusters | effective clusters | mean Δ | Holm p | "
            "simultaneous Bonferroni 95% CI | claim-eligible |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        low = row["bonferroni_simultaneous_ci_low"]
        high = row["bonferroni_simultaneous_ci_high"]
        interval = f"[{_fmt(low)}, {_fmt(high)}]" if low is not None and high is not None else "NA"
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {row['cluster_count']:,} | "
            f"{row['effective_cluster_count']:.2f} | {_fmt(row['mean_difference'])} | "
            f"{_fmt(row['p_holm_familywise'])} | {interval} | "
            f"{'YES' if row['eligible_for_confirmatory_claim'] else 'NO'} |"
        )
    return lines


def _cluster_table(contrasts: dict[str, Any], level: str) -> list[str]:
    lines = [
        (
            "| contrast | metric | matches | clusters | effective clusters | max cluster | "
            "mean Δ | CR1 95% CI |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
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
                f"{row['max_cluster_size']:,} | {_fmt(row['mean_difference'])} | {interval} |"
            )
    return lines


def write_confirmatory_report(
    config: dict[str, Any],
    preparation: dict[str, Any],
    null_qc: dict[str, Any],
    clustered: dict[str, Any],
    familywise: dict[str, Any],
    did: dict[str, Any],
    *,
    descriptive_report: Path,
    codeword: dict[str, Any] | None = None,
    bitstream: dict[str, Any] | None = None,
    spatial_null: dict[str, Any] | None = None,
) -> Path:
    """Write the primary run-level report with DiD as the cross-corpus claim gate."""
    output_dir = Path(config["outputs"]["directory"])
    report_path = output_dir / "confirmatory_report.md"
    surface_granularity = str(config["sources"]["surface"].get("granularity", "url"))
    onion_granularity = str(config["sources"]["onion"].get("granularity", "url"))
    support = preparation.get("matching", {})
    selected_by_length = support.get("selected_by_length", {})
    support_text = (
        ", ".join(f"{length} B: {count:,}" for length, count in selected_by_length.items())
        or "not available"
    )

    lines = [
        "# QR URL Geometry Assay — confirmatory report",
        "",
        "The direct cross-corpus claim gate is the preregistered difference-in-differences (DiD).",
        "",
        "## Locked estimand",
        "",
        f"- Surface granularity: **{surface_granularity}**",
        f"- Onion granularity: **{onion_granularity}**",
        f"- Matched quadruples: **{preparation['matched_pairs']:,}**",
        f"- Length weighting: **{support.get('length_weighting', 'unknown')}**",
        f"- Selected shared byte-length support: **{support_text}**",
        f"- Synthetic null: **{preparation['synthetic_mode']}**",
        f"- QR masks per payload: **{len(config['qr']['masks'])}**, averaged before inference",
        "",
    ]

    if surface_granularity == onion_granularity == "origin":
        lines.extend(
            [
                (
                    "**Origin-arm interpretation.** Conditional hostname/cryptographic-identifier "
                    "control on shared exact byte-length support; not a Chomskyan grammar claim."
                ),
                "",
            ]
        )
    elif surface_granularity == onion_granularity == "path_query":
        lines.extend(
            [
                (
                    "**Grammar-arm interpretation.** Hosts and schemes are removed before encoding; "
                    "host/source hashes remain only as dependency/provenance clusters."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Structural-null QC",
            "",
            f"Hard invariants: **{'PASS' if null_qc['passed_hard_invariants'] else 'FAIL'}**",
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
            "## Primary cross-corpus test — difference in differences",
            "",
            "`mean(onion_natural - onion_null) - mean(surface_natural - surface_null)`",
            "",
            (
                "Exact Surface↔Onion matching is a balance device only. The DiD SE is pairing-"
                "invariant: Surface and Onion host-cluster CR1 variances are estimated separately "
                "and added. A strict pass additionally requires the fixed four-cell Holm family "
                "and its simultaneous Bonferroni interval to exclude zero."
            ),
            "",
            *_did_table(did["host_independent_cluster_difference"]),
            "",
            "## Within-corpus decomposition — fixed eight-cell family",
            "",
            (
                "These rows decompose the DiD. One corpus crossing a threshold while the other "
                "does not is not itself evidence that the corpora differ."
            ),
            "",
            *_familywise_table(familywise["host"]),
            "",
            "## Nominal host-cluster effect summaries",
            "",
            *_cluster_table(clustered["contrasts"], "host"),
            "",
            "## Source-file-cluster sensitivity",
            "",
            *_did_table(did["source_file_independent_cluster_difference"]),
            "",
            *_familywise_table(familywise["source"]),
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
            ]
        )
    if spatial_null is not None:
        lines.extend(
            [
                "## Exact-bit spatial-placement null",
                "",
                f"Diagnostic file: `{Path(spatial_null['output']).name}`",
                "",
            ]
        )
    if codeword is not None:
        lines.extend(
            [
                "## Data-codeword versus Reed–Solomon diagnostic",
                "",
                f"Diagnostic file: `{Path(codeword['output']).name}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            (
                "Stage 1 measures encoded stimulus structure relative to declared nulls. It does "
                "not establish semantics, neural carriers, spontaneous rotation, hysteresis, or "
                "cyclicity inside a model."
            ),
            "",
            "## Secondary artifacts",
            "",
            f"- Descriptive/QC report: `{descriptive_report.name}`",
            "- `analysis.json`",
            "- `cluster_analysis.json`",
            "- `familywise_analysis.json`",
            "- `did_analysis.json`",
            "- `null_qc.json`",
        ]
    )
    if bitstream is not None:
        lines.append("- `bitstream_analysis.json`")
    if spatial_null is not None:
        lines.append("- `spatial_null_analysis.json`")
    if codeword is not None:
        lines.append("- `codeword_analysis.json`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
