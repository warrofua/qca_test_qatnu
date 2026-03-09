#!/usr/bin/env python3
"""
Calibrate model parameters to match PDF critical-point snapshot targets.
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


def _safe_abs_err(value: float | None, target: float, missing_penalty: float) -> float:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return missing_penalty
    return abs(float(value) - target)


def _score_row(
    *,
    c1: float | None,
    rev: float | None,
    c2: float | None,
    targets: dict[str, float],
    scales: dict[str, float],
    missing_penalty: float,
) -> tuple[float, float, float, float]:
    e_c1 = _safe_abs_err(c1, targets["lambda_c1"], missing_penalty)
    e_rev = _safe_abs_err(rev, targets["lambda_revival"], missing_penalty)
    e_c2 = _safe_abs_err(c2, targets["lambda_c2"], missing_penalty)
    score = e_c1 / scales["lambda_c1"] + e_rev / scales["lambda_revival"] + e_c2 / scales["lambda_c2"]
    return score, e_c1, e_rev, e_c2


def _run_scan(
    scanner: ParameterScanner,
    *,
    N: int,
    alpha: float,
    points: int,
    lambda_min: float,
    lambda_max: float,
    bond_cutoff: int,
    hotspot_multiplier: float,
    deltaB: float,
    kappa: float,
    k0: int,
    output_dir: Path,
    run_tag: str,
    quiet_scanner: bool,
) -> pd.DataFrame:
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


def _save_summary_plots(df: pd.DataFrame, targets: dict[str, float], out_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["combo"] = (
        "dB="
        + plot_df["deltaB"].map(lambda x: f"{x:.2f}")
        + ", k="
        + plot_df["kappa"].map(lambda x: f"{x:.3f}")
        + ", hs="
        + plot_df["hotspot_multiplier"].map(lambda x: f"{x:.2f}")
    )

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        plot_df["lambda_revival"],
        plot_df["lambda_c2"],
        c=plot_df["score"],
        cmap="viridis_r",
        s=90,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
    )
    plt.axvline(targets["lambda_revival"], color="red", linestyle="--", linewidth=1.5, label="target λ_rev")
    plt.axhline(targets["lambda_c2"], color="orange", linestyle="--", linewidth=1.5, label="target λ_c2")
    plt.xlabel("lambda_revival (official/global)")
    plt.ylabel("lambda_c2")
    plt.title("Calibration landscape: revival vs catastrophic onset")
    cbar = plt.colorbar(scatter)
    cbar.set_label("calibration score (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_revival_vs_c2.png", dpi=220, bbox_inches="tight")
    plt.close()

    top = plot_df.nsmallest(10, "score").copy()
    plt.figure(figsize=(10, 5))
    x = np.arange(len(top))
    width = 0.25
    plt.bar(x - width, top["err_c1"], width=width, label="|Δ λ_c1|")
    plt.bar(x, top["err_revival"], width=width, label="|Δ λ_rev|")
    plt.bar(x + width, top["err_c2"], width=width, label="|Δ λ_c2|")
    plt.xticks(x, [f"#{i+1}" for i in range(len(top))])
    plt.ylabel("absolute error")
    plt.title("Top-10 candidate errors vs PDF targets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_top10_errors.png", dpi=220, bbox_inches="tight")
    plt.close()


def _write_report(
    *,
    out_dir: Path,
    targets: dict[str, float],
    scales: dict[str, float],
    summary: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# PDF Calibration Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Targets (from PDF Appendix A)")
    lines.append("")
    lines.append(f"- lambda_c1: {targets['lambda_c1']}")
    lines.append(f"- lambda_revival: {targets['lambda_revival']}")
    lines.append(f"- lambda_c2: {targets['lambda_c2']}")
    lines.append("")
    lines.append("## Normalization scales")
    lines.append("")
    for k, v in scales.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Best candidates")
    lines.append("")
    top = summary.nsmallest(10, "score")
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"{i}. score={row['score']:.4f} | "
            f"deltaB={row['deltaB']:.2f}, kappa={row['kappa']:.3f}, hotspot={row['hotspot_multiplier']:.2f}, chi={int(row['bond_cutoff'])} | "
            f"lambda_c1={row['lambda_c1']:.6f}, lambda_revival={row['lambda_revival']:.6f}, lambda_c2={row['lambda_c2'] if np.isfinite(row['lambda_c2']) else 'NA'}"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- score = |Δc1|/s_c1 + |Δrev|/s_rev + |Δc2|/s_c2 (lower is better)")
    lines.append("- lambda_revival uses official global post-violation reporting rule")
    lines.append("- lambda_revival_first remains available for diagnostics in CSV")
    lines.append("")
    (out_dir / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_calibration(
    *,
    N: int,
    alpha: float,
    points: int,
    lambda_min: float,
    lambda_max: float,
    deltaB_values: Iterable[float],
    kappa_values: Iterable[float],
    hotspot_values: Iterable[float],
    bond_cutoff: int,
    k0: int,
    targets: dict[str, float],
    scales: dict[str, float],
    missing_penalty: float,
    output_dir: Path,
    quiet_scanner: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scanner = ParameterScanner()
    combos = list(itertools.product(deltaB_values, kappa_values, hotspot_values))
    rows = []
    total = len(combos)

    for idx, (deltaB, kappa, hotspot) in enumerate(combos, start=1):
        print(
            f"[{idx}/{total}] deltaB={deltaB:.2f}, kappa={kappa:.3f}, hotspot={hotspot:.2f}, chi={bond_cutoff}",
            flush=True,
        )
        run_tag = f"cal_dB{deltaB:.2f}_kap{kappa:.3f}_hs{hotspot:.2f}_chi{bond_cutoff}"
        df = _run_scan(
            scanner,
            N=N,
            alpha=alpha,
            points=points,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            bond_cutoff=bond_cutoff,
            hotspot_multiplier=hotspot,
            deltaB=deltaB,
            kappa=kappa,
            k0=k0,
            output_dir=output_dir,
            run_tag=run_tag,
            quiet_scanner=quiet_scanner,
        )
        crit = PhaseAnalyzer.analyze_critical_points(df)
        score, e_c1, e_rev, e_c2 = _score_row(
            c1=crit.get("lambda_c1"),
            rev=crit.get("lambda_revival"),
            c2=crit.get("lambda_c2"),
            targets=targets,
            scales=scales,
            missing_penalty=missing_penalty,
        )
        rows.append(
            {
                "deltaB": deltaB,
                "kappa": kappa,
                "hotspot_multiplier": hotspot,
                "bond_cutoff": bond_cutoff,
                "lambda_c1": crit.get("lambda_c1"),
                "lambda_revival": crit.get("lambda_revival"),
                "lambda_c2": crit.get("lambda_c2"),
                "residual_min": crit.get("residual_min"),
                "lambda_revival_first": crit.get("lambda_revival_first"),
                "lambda_revival_global": crit.get("lambda_revival_global"),
                "revival_gap": crit.get("revival_gap"),
                "revival_method": crit.get("revival_method"),
                "revival_reporting_rule": crit.get("revival_reporting_rule"),
                "err_c1": e_c1,
                "err_revival": e_rev,
                "err_c2": e_c2,
                "score": score,
                "mean_residual": float(df["residual"].mean()),
                "max_residual": float(df["residual"].max()),
            }
        )

    summary = pd.DataFrame(rows).sort_values("score")
    summary_path = output_dir / "calibration_summary.csv"
    summary.to_csv(summary_path, index=False)

    _save_summary_plots(summary, targets, output_dir)
    _write_report(out_dir=output_dir, targets=targets, scales=scales, summary=summary)

    print("\nCalibration completed.")
    print(summary.head(12).to_string(index=False))
    print(f"\nOUTPUT_DIR={output_dir}")
    print(f"SUMMARY={summary_path}")
    print(f"REPORT={output_dir / 'calibration_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate model parameters to PDF critical-point snapshot")
    parser.add_argument("--N", type=int, default=4, help="Number of matter sites")
    parser.add_argument("--alpha", type=float, default=0.8, help="Postulate alpha")
    parser.add_argument("--points", type=int, default=30, help="Lambda points per scan")
    parser.add_argument("--lambda-min", type=float, default=0.1, help="Minimum lambda")
    parser.add_argument("--lambda-max", type=float, default=1.5, help="Maximum lambda")
    parser.add_argument("--deltaB-values", type=str, default="5.0,5.5,6.0,6.5,7.0", help="Comma-separated deltaB grid")
    parser.add_argument("--kappa-values", type=str, default="0.08,0.10,0.12,0.15,0.20", help="Comma-separated kappa grid")
    parser.add_argument("--hotspot-values", type=str, default="1.5,2.0,2.5,3.0", help="Comma-separated hotspot multipliers")
    parser.add_argument("--bond-cutoff", type=int, default=4, help="Bond cutoff (chi)")
    parser.add_argument("--k0", type=int, default=4, help="Target degree parameter")
    parser.add_argument("--target-c1", type=float, default=0.203, help="Target lambda_c1 from PDF")
    parser.add_argument("--target-revival", type=float, default=1.058, help="Target lambda_revival from PDF")
    parser.add_argument("--target-c2", type=float, default=1.095, help="Target lambda_c2 from PDF")
    parser.add_argument("--scale-c1", type=float, default=0.1, help="Normalization scale for lambda_c1 error")
    parser.add_argument("--scale-revival", type=float, default=0.2, help="Normalization scale for lambda_revival error")
    parser.add_argument("--scale-c2", type=float, default=0.2, help="Normalization scale for lambda_c2 error")
    parser.add_argument("--missing-penalty", type=float, default=5.0, help="Absolute error used when metric is missing")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output directory")
    parser.add_argument(
        "--quiet-scanner",
        action="store_true",
        help="Suppress scanner internals and print only per-combination progress",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs")
        / (
            f"pdf_calibration_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            f"_N{args.N}_alpha{args.alpha:.2f}_pts{args.points}"
        )
    )
    targets = {
        "lambda_c1": args.target_c1,
        "lambda_revival": args.target_revival,
        "lambda_c2": args.target_c2,
    }
    scales = {
        "lambda_c1": args.scale_c1,
        "lambda_revival": args.scale_revival,
        "lambda_c2": args.scale_c2,
    }
    run_calibration(
        N=args.N,
        alpha=args.alpha,
        points=args.points,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        deltaB_values=_parse_float_list(args.deltaB_values),
        kappa_values=_parse_float_list(args.kappa_values),
        hotspot_values=_parse_float_list(args.hotspot_values),
        bond_cutoff=args.bond_cutoff,
        k0=args.k0,
        targets=targets,
        scales=scales,
        missing_penalty=args.missing_penalty,
        output_dir=out_dir,
        quiet_scanner=args.quiet_scanner,
    )


if __name__ == "__main__":
    main()
