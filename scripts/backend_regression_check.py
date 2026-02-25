#!/usr/bin/env python3
"""
Dense-vs-iterative regression checker for critical-slowing scans.

Runs identical scan settings with both solver backends and fails when
the maximum absolute metric deviation exceeds the configured tolerance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = REPO_ROOT / "scripts" / "critical_slowing_scan.py"


def _build_scan_cmd(args: argparse.Namespace, backend: str, output_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(SCAN_SCRIPT),
        "--N",
        str(args.N),
        "--topologies",
        args.topologies,
        "--lambdas",
        args.lambdas,
        "--bond-cutoff",
        str(args.bond_cutoff),
        "--hotspot-multiplier",
        str(args.hotspot_multiplier),
        "--deltaB",
        str(args.deltaB),
        "--kappa",
        str(args.kappa),
        "--k0",
        str(args.k0),
        "--t-max",
        str(args.t_max),
        "--n-times",
        str(args.n_times),
        "--lambda-proxy",
        args.lambda_proxy,
        "--tail-frac",
        str(args.tail_frac),
        "--rel-tol",
        str(args.rel_tol),
        "--sustain-points",
        str(args.sustain_points),
        "--min-scale",
        str(args.min_scale),
        "--solver-backend",
        backend,
        "--iterative-tol",
        str(args.iterative_tol),
        "--iterative-maxiter",
        str(args.iterative_maxiter),
        "--output-dir",
        str(output_dir),
    ]


def _run_scan(cmd: List[str], env: Dict[str, str]) -> None:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Scan failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty summary file: {path}")
    return df


def _numeric_metric_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    skip = {"backend", "hamiltonian_mode", "topology"}
    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def _compare(dense: pd.DataFrame, iterative: pd.DataFrame) -> pd.DataFrame:
    keys = ["topology", "lambda"]
    left = dense.sort_values(keys).reset_index(drop=True)
    right = iterative.sort_values(keys).reset_index(drop=True)

    if left.shape[0] != right.shape[0]:
        raise ValueError(f"Row count mismatch: dense={left.shape[0]}, iterative={right.shape[0]}")

    if not np.array_equal(left[keys].to_numpy(), right[keys].to_numpy()):
        raise ValueError("Key mismatch between dense and iterative summaries (topology/lambda).")

    metrics = sorted(set(_numeric_metric_columns(left)).intersection(_numeric_metric_columns(right)))
    rows: List[Dict[str, object]] = []
    for idx in range(left.shape[0]):
        topo = str(left.at[idx, "topology"])
        lam = float(left.at[idx, "lambda"])
        for metric in metrics:
            dv = float(left.at[idx, metric])
            iv = float(right.at[idx, metric])
            if np.isnan(dv) and np.isnan(iv):
                diff = 0.0
            else:
                diff = abs(dv - iv)
            rows.append(
                {
                    "topology": topo,
                    "lambda": lam,
                    "metric": metric,
                    "dense_value": dv,
                    "iterative_value": iv,
                    "abs_diff": diff,
                }
            )
    return pd.DataFrame(rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense-vs-iterative regression checker")
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--topologies", type=str, default="path,star")
    p.add_argument("--lambdas", type=str, default="path:0.9,1.0;star:1.5,1.6")
    p.add_argument("--bond-cutoff", type=int, default=4)
    p.add_argument("--hotspot-multiplier", type=float, default=3.0)
    p.add_argument("--deltaB", type=float, default=5.0)
    p.add_argument("--kappa", type=float, default=0.1)
    p.add_argument("--k0", type=float, default=4.0)
    p.add_argument("--t-max", type=float, default=70.0)
    p.add_argument("--n-times", type=int, default=140)
    p.add_argument("--lambda-proxy", type=str, default="log", choices=["log", "linear"])
    p.add_argument("--tail-frac", type=float, default=0.2)
    p.add_argument("--rel-tol", type=float, default=0.08)
    p.add_argument("--sustain-points", type=int, default=8)
    p.add_argument("--min-scale", type=float, default=1e-8)
    p.add_argument("--iterative-tol", type=float, default=1e-9)
    p.add_argument("--iterative-maxiter", type=int, default=0)
    p.add_argument("--max-abs-diff", type=float, default=1e-8)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else REPO_ROOT / "outputs" / f"backend_regression_check_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    dense_dir = out_dir / "dense"
    iter_dir = out_dir / "iterative"
    dense_dir.mkdir(parents=True, exist_ok=True)
    iter_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["PYTHONHASHSEED"] = "0"

    dense_cmd = _build_scan_cmd(args, backend="dense", output_dir=dense_dir)
    iter_cmd = _build_scan_cmd(args, backend="iterative", output_dir=iter_dir)

    print(f"[run] dense: {' '.join(dense_cmd)}", flush=True)
    _run_scan(dense_cmd, env=env)
    print(f"[run] iterative: {' '.join(iter_cmd)}", flush=True)
    _run_scan(iter_cmd, env=env)

    dense_summary = _load_summary(dense_dir / "summary.csv")
    iter_summary = _load_summary(iter_dir / "summary.csv")
    diffs = _compare(dense_summary, iter_summary)
    diffs_path = out_dir / "metric_diffs.csv"
    diffs.to_csv(diffs_path, index=False)

    max_abs_diff = float(diffs["abs_diff"].max()) if not diffs.empty else 0.0
    worst = diffs.iloc[0].to_dict() if not diffs.empty else {}
    status = "PASS" if max_abs_diff <= float(args.max_abs_diff) else "FAIL"

    summary = {
        "status": status,
        "max_abs_diff": max_abs_diff,
        "threshold": float(args.max_abs_diff),
        "worst_metric": worst,
        "dense_summary": str(dense_dir / "summary.csv"),
        "iterative_summary": str(iter_dir / "summary.csv"),
        "metric_diffs": str(diffs_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Backend Regression Check",
        "",
        f"- status: **{status}**",
        f"- threshold: `{args.max_abs_diff:.3e}`",
        f"- max_abs_diff: `{max_abs_diff:.3e}`",
        f"- dense summary: `{dense_dir / 'summary.csv'}`",
        f"- iterative summary: `{iter_dir / 'summary.csv'}`",
        f"- metric diffs: `{diffs_path}`",
        "",
        "## Worst metric",
        "```json",
        json.dumps(worst, indent=2, default=float),
        "```",
        "",
        "## Top 20 diffs",
        "```csv",
        diffs.head(20).to_csv(index=False).strip(),
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"OUTPUT_DIR={out_dir}")
    print(f"MAX_ABS_DIFF={max_abs_diff:.3e}")
    print(f"THRESHOLD={float(args.max_abs_diff):.3e}")
    print(f"STATUS={status}")

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
