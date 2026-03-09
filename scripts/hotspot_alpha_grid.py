#!/usr/bin/env python3
"""
Run hotspot-multiplier vs alpha sensitivity scans and summarize critical-point drift.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running this script directly from scripts/ while importing repo modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase_analysis import PhaseAnalyzer
from scanners import ParameterScanner


def _parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _save_metric_heatmap(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    pivot = df.pivot(index="alpha", columns="hotspot_multiplier", values=metric).apply(
        pd.to_numeric, errors="coerce"
    )
    plt.figure(figsize=(7, 4.5))
    image = plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), [f"x{m:.1f}" for m in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{a:.1f}" for a in pivot.index])
    plt.xlabel("Hotspot multiplier")
    plt.ylabel("alpha")
    plt.title(title)
    cbar = plt.colorbar(image)
    cbar.set_label(metric)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def _compute_spans(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha, part in summary.groupby("alpha"):
        rows.append(
            {
                "alpha": alpha,
                "lambda_c1_min": part["lambda_c1"].min(),
                "lambda_c1_max": part["lambda_c1"].max(),
                "lambda_c1_span": part["lambda_c1"].max() - part["lambda_c1"].min(),
                "lambda_revival_min": part["lambda_revival"].min(),
                "lambda_revival_max": part["lambda_revival"].max(),
                "lambda_revival_span": part["lambda_revival"].max() - part["lambda_revival"].min(),
                "lambda_revival_global_min": part["lambda_revival_global"].min(),
                "lambda_revival_global_max": part["lambda_revival_global"].max(),
                "lambda_revival_global_span": part["lambda_revival_global"].max()
                - part["lambda_revival_global"].min(),
                "lambda_c2_min": part["lambda_c2"].min(),
                "lambda_c2_max": part["lambda_c2"].max(),
                "lambda_c2_span": part["lambda_c2"].max() - part["lambda_c2"].min(),
                "mean_residual_min": part["mean_residual"].min(),
                "mean_residual_max": part["mean_residual"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("alpha")


def _run_scan(
    scanner: ParameterScanner,
    *,
    N: int,
    alpha: float,
    hotspot_multiplier: float,
    points: int,
    lambda_min: float,
    lambda_max: float,
    output_dir: Path,
    quiet_scanner: bool,
) -> pd.DataFrame:
    run_tag = f"a{alpha:.1f}_hotspotx{hotspot_multiplier:.1f}"
    kwargs = dict(
        N=N,
        alpha=alpha,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        num_points=points,
        output_dir=str(output_dir),
        run_tag=run_tag,
        hotspot_multiplier=hotspot_multiplier,
    )
    if quiet_scanner:
        with open("/dev/null", "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return scanner.scan_lambda_parallel(**kwargs)
    return scanner.scan_lambda_parallel(**kwargs)


def run_grid(
    *,
    N: int,
    alphas: Iterable[float],
    multipliers: Iterable[float],
    points: int,
    lambda_min: float,
    lambda_max: float,
    output_dir: Path,
    quiet_scanner: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scanner = ParameterScanner()
    rows = []

    for alpha in alphas:
        for multiplier in multipliers:
            print(f"Running alpha={alpha:.1f}, hotspot={multiplier:.1f}x ...", flush=True)
            df = _run_scan(
                scanner,
                N=N,
                alpha=alpha,
                hotspot_multiplier=multiplier,
                points=points,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                output_dir=output_dir,
                quiet_scanner=quiet_scanner,
            )
            crit = PhaseAnalyzer.analyze_critical_points(df)
            rows.append(
                {
                    "alpha": alpha,
                    "hotspot_multiplier": multiplier,
                    "lambda_c1": crit.get("lambda_c1"),
                    "lambda_revival": crit.get("lambda_revival"),
                    "residual_min": crit.get("residual_min"),
                    "lambda_revival_first": crit.get("lambda_revival_first"),
                    "residual_min_first": crit.get("residual_min_first"),
                    "lambda_revival_global": crit.get("lambda_revival_global"),
                    "residual_min_global": crit.get("residual_min_global"),
                    "first_violation_lambda": crit.get("first_violation_lambda"),
                    "revival_reporting_rule": crit.get("revival_reporting_rule"),
                    "revival_method": crit.get("revival_method"),
                    "revival_gap": crit.get("revival_gap"),
                    "lambda_c2": crit.get("lambda_c2"),
                    "max_residual": float(df["residual"].max()),
                    "mean_residual": float(df["residual"].mean()),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["alpha", "hotspot_multiplier"])
    summary_csv = output_dir / "critical_points_alpha_hotspot_grid.csv"
    summary.to_csv(summary_csv, index=False)

    spans = _compute_spans(summary)
    spans_csv = output_dir / "critical_point_spans_by_alpha.csv"
    spans.to_csv(spans_csv, index=False)

    _save_metric_heatmap(
        summary,
        "lambda_c1",
        output_dir / "heatmap_lambda_c1.png",
        f"Lambda c1 sensitivity (N={N}, points={points})",
    )
    _save_metric_heatmap(
        summary,
        "lambda_revival",
        output_dir / "heatmap_lambda_revival.png",
        f"Lambda revival (official/global) sensitivity (N={N}, points={points})",
    )
    _save_metric_heatmap(
        summary,
        "lambda_revival_global",
        output_dir / "heatmap_lambda_revival_global.png",
        f"Lambda revival (global) sensitivity (N={N}, points={points})",
    )
    _save_metric_heatmap(
        summary,
        "lambda_c2",
        output_dir / "heatmap_lambda_c2.png",
        f"Lambda c2 sensitivity (N={N}, points={points})",
    )

    plt.figure(figsize=(8, 5))
    for alpha in sorted(summary["alpha"].unique()):
        part = summary[summary["alpha"] == alpha].sort_values("hotspot_multiplier")
        plt.plot(
            part["hotspot_multiplier"],
            part["lambda_revival"],
            marker="o",
            linewidth=2,
            label=f"alpha={alpha:.1f}",
        )
    multipliers_sorted = sorted(summary["hotspot_multiplier"].unique())
    plt.xticks(multipliers_sorted, [f"x{m:.1f}" for m in multipliers_sorted])
    plt.xlabel("Hotspot multiplier")
    plt.ylabel("lambda_revival")
    plt.title(f"Revival-point drift vs hotspot multiplier (official/global, N={N}, points={points})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    revival_plot = output_dir / "lambda_revival_vs_hotspot_by_alpha.png"
    plt.savefig(revival_plot, dpi=220, bbox_inches="tight")
    plt.close()

    print("\nCompleted grid sweep.")
    print(summary.to_string(index=False))
    print("\nSpan summary:")
    print(spans.to_string(index=False))
    print(f"\nOUTPUT_DIR={output_dir}")
    print(f"SUMMARY_CSV={summary_csv}")
    print(f"SPANS_CSV={spans_csv}")
    print(f"REVIVAL_PLOT={revival_plot}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hotspot-multiplier vs alpha sensitivity runner")
    parser.add_argument("--N", type=int, default=4, help="Number of matter sites")
    parser.add_argument("--alphas", type=str, default="0.6,0.8,1.0,1.2", help="Comma-separated alpha values")
    parser.add_argument(
        "--hotspot-multipliers",
        type=str,
        default="2.0,3.0,4.0",
        help="Comma-separated hotspot multipliers",
    )
    parser.add_argument("--points", type=int, default=80, help="Number of lambda points per run")
    parser.add_argument("--lambda-min", type=float, default=0.1, help="Minimum lambda")
    parser.add_argument("--lambda-max", type=float, default=1.5, help="Maximum lambda")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. If omitted, a timestamped directory under outputs/ is created.",
    )
    parser.add_argument(
        "--quiet-scanner",
        action="store_true",
        help="Suppress scanner progress and worker logs; only show high-level progress.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs")
        / f"hotspot_alpha_multiplier_grid_{datetime.now().strftime('%Y%m%d-%H%M%S')}_N{args.N}_pts{args.points}"
    )
    run_grid(
        N=args.N,
        alphas=_parse_float_list(args.alphas),
        multipliers=_parse_float_list(args.hotspot_multipliers),
        points=args.points,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        output_dir=out_dir,
        quiet_scanner=args.quiet_scanner,
    )


if __name__ == "__main__":
    main()
