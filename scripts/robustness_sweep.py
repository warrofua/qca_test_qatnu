#!/usr/bin/env python3
"""
Run a controlled robustness matrix over deltaB, kappa, and bond_cutoff.
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running directly from scripts/ while importing repo modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase_analysis import PhaseAnalyzer
from scanners import ParameterScanner


def _parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _save_heatmap(df: pd.DataFrame, metric: str, chi: int, out_dir: Path) -> None:
    subset = df[df["bond_cutoff"] == chi]
    if subset.empty:
        return
    pivot = subset.pivot(index="kappa", columns="deltaB", values=metric).apply(
        pd.to_numeric, errors="coerce"
    )
    plt.figure(figsize=(6, 4.5))
    image = plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), [f"{v:.1f}" for v in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{v:.3f}" for v in pivot.index])
    plt.xlabel("deltaB")
    plt.ylabel("kappa")
    plt.title(f"{metric} (chi={chi})")
    cbar = plt.colorbar(image)
    cbar.set_label(metric)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"heatmap_{metric}_chi{chi}.png", dpi=220, bbox_inches="tight")
    plt.close()


def _format_value(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{value:.6f}"


def _write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    spans: dict[str, float],
    baseline_row: pd.Series | None,
) -> None:
    lines = []
    lines.append("# Robustness Sweep Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Global spans across all parameter combinations")
    lines.append("")
    for metric, span in spans.items():
        lines.append(f"- `{metric}` span: {span:.6f}")
    lines.append("")
    if baseline_row is not None:
        lines.append("## Baseline (deltaB=5.0, kappa=0.1, chi=4)")
        lines.append("")
        for key in ("lambda_c1", "lambda_revival", "lambda_c2", "mean_residual"):
            lines.append(f"- `{key}`: {_format_value(float(baseline_row[key]))}")
        lines.append("")
    lines.append("## Lowest-mean-residual configurations")
    lines.append("")
    top = summary.sort_values("mean_residual").head(5)
    for _, row in top.iterrows():
        lines.append(
            "- deltaB={d:.2f}, kappa={k:.3f}, chi={chi}: mean_residual={r:.6f}, "
            "lambda_c1={c1}, lambda_revival={rev}, lambda_c2={c2}".format(
                d=row["deltaB"],
                k=row["kappa"],
                chi=int(row["bond_cutoff"]),
                r=row["mean_residual"],
                c1=_format_value(row["lambda_c1"]),
                rev=_format_value(row["lambda_revival"]),
                c2=_format_value(row["lambda_c2"]),
            )
        )
    lines.append("")
    report_path = out_dir / "robustness_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scan_one(
    scanner: ParameterScanner,
    *,
    N: int,
    alpha: float,
    points: int,
    lambda_min: float,
    lambda_max: float,
    hotspot_multiplier: float,
    deltaB: float,
    kappa: float,
    k0: int,
    bond_cutoff: int,
    output_dir: Path,
    quiet_scanner: bool,
) -> pd.DataFrame:
    run_tag = f"dB{deltaB:.2f}_kap{kappa:.3f}_chi{bond_cutoff}"
    kwargs = dict(
        N=N,
        alpha=alpha,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        num_points=points,
        bond_cutoff=bond_cutoff,
        output_dir=str(output_dir),
        run_tag=run_tag,
        hotspot_multiplier=hotspot_multiplier,
        deltaB=deltaB,
        kappa=kappa,
        k0=k0,
    )
    if quiet_scanner:
        with open("/dev/null", "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return scanner.scan_lambda_parallel(**kwargs)
    return scanner.scan_lambda_parallel(**kwargs)


def run_sweep(
    *,
    N: int,
    alpha: float,
    points: int,
    lambda_min: float,
    lambda_max: float,
    hotspot_multiplier: float,
    deltaB_values: Iterable[float],
    kappa_values: Iterable[float],
    bond_cutoff_values: Iterable[int],
    k0: int,
    output_dir: Path,
    quiet_scanner: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scanner = ParameterScanner()
    combos = list(itertools.product(deltaB_values, kappa_values, bond_cutoff_values))
    rows = []

    total = len(combos)
    for idx, (deltaB, kappa, chi) in enumerate(combos, start=1):
        print(
            f"[{idx}/{total}] deltaB={deltaB:.2f}, kappa={kappa:.3f}, chi={chi}",
            flush=True,
        )
        df = _scan_one(
            scanner,
            N=N,
            alpha=alpha,
            points=points,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            hotspot_multiplier=hotspot_multiplier,
            deltaB=deltaB,
            kappa=kappa,
            k0=k0,
            bond_cutoff=chi,
            output_dir=output_dir,
            quiet_scanner=quiet_scanner,
        )
        crit = PhaseAnalyzer.analyze_critical_points(df)
        rows.append(
            {
                "deltaB": deltaB,
                "kappa": kappa,
                "bond_cutoff": chi,
                "lambda_c1": crit.get("lambda_c1"),
                "lambda_revival": crit.get("lambda_revival"),
                "lambda_c2": crit.get("lambda_c2"),
                "residual_min": crit.get("residual_min"),
                "lambda_revival_first": crit.get("lambda_revival_first"),
                "lambda_revival_global": crit.get("lambda_revival_global"),
                "revival_gap": crit.get("revival_gap"),
                "revival_reporting_rule": crit.get("revival_reporting_rule"),
                "mean_residual": float(df["residual"].mean()),
                "max_residual": float(df["residual"].max()),
                "min_residual": float(df["residual"].min()),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["deltaB", "kappa", "bond_cutoff"])
    summary_path = output_dir / "robustness_summary.csv"
    summary.to_csv(summary_path, index=False)

    baseline = summary[
        (summary["deltaB"] == 5.0)
        & (summary["kappa"] == 0.1)
        & (summary["bond_cutoff"] == 4)
    ]
    baseline_row = baseline.iloc[0] if not baseline.empty else None
    if baseline_row is not None:
        for metric in ("lambda_c1", "lambda_revival", "lambda_c2", "mean_residual"):
            summary[f"{metric}_delta_from_baseline"] = summary[metric] - float(baseline_row[metric])
    delta_path = output_dir / "robustness_summary_with_deltas.csv"
    summary.to_csv(delta_path, index=False)

    spans = {}
    for metric in ("lambda_c1", "lambda_revival", "lambda_c2", "mean_residual"):
        vals = pd.to_numeric(summary[metric], errors="coerce").dropna()
        spans[metric] = float(vals.max() - vals.min()) if not vals.empty else float("nan")
    spans_df = pd.DataFrame([{"metric": k, "span": v} for k, v in spans.items()])
    spans_path = output_dir / "robustness_spans.csv"
    spans_df.to_csv(spans_path, index=False)

    for metric in ("lambda_c1", "lambda_revival", "lambda_c2", "mean_residual"):
        for chi in sorted(set(summary["bond_cutoff"].tolist())):
            _save_heatmap(summary, metric, int(chi), output_dir)

    _write_report(output_dir, summary, spans, baseline_row)

    print("\nSweep completed.")
    print(summary.to_string(index=False))
    print("\nGlobal spans:")
    print(spans_df.to_string(index=False))
    print(f"\nOUTPUT_DIR={output_dir}")
    print(f"SUMMARY={summary_path}")
    print(f"DELTAS={delta_path}")
    print(f"SPANS={spans_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness sweep over deltaB, kappa, and bond_cutoff")
    parser.add_argument("--N", type=int, default=4, help="Number of matter sites")
    parser.add_argument("--alpha", type=float, default=0.8, help="Postulate alpha")
    parser.add_argument("--points", type=int, default=30, help="Lambda points per scan")
    parser.add_argument("--lambda-min", type=float, default=0.1, help="Minimum lambda")
    parser.add_argument("--lambda-max", type=float, default=1.5, help="Maximum lambda")
    parser.add_argument("--hotspot-multiplier", type=float, default=3.0, help="Hotspot multiplier")
    parser.add_argument("--k0", type=int, default=4, help="Target degree penalty center")
    parser.add_argument("--deltaB-values", type=str, default="3.0,5.0,7.0", help="Comma-separated deltaB values")
    parser.add_argument("--kappa-values", type=str, default="0.05,0.1,0.2", help="Comma-separated kappa values")
    parser.add_argument("--bond-cutoff-values", type=str, default="3,4,6", help="Comma-separated chi cutoffs")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output directory")
    parser.add_argument(
        "--quiet-scanner",
        action="store_true",
        help="Suppress scanner internals and only show per-combination progress",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs")
        / (
            f"robustness_sweep_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            f"_N{args.N}_alpha{args.alpha:.2f}_pts{args.points}"
        )
    )
    run_sweep(
        N=args.N,
        alpha=args.alpha,
        points=args.points,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        hotspot_multiplier=args.hotspot_multiplier,
        deltaB_values=_parse_float_list(args.deltaB_values),
        kappa_values=_parse_float_list(args.kappa_values),
        bond_cutoff_values=_parse_int_list(args.bond_cutoff_values),
        k0=args.k0,
        output_dir=out_dir,
        quiet_scanner=args.quiet_scanner,
    )


if __name__ == "__main__":
    main()
