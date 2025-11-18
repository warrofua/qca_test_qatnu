#!/usr/bin/env python3
"""Plot critical points vs alpha for the N=4 path sweep."""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SweepPoint:
    alpha: float
    lambda_c1: float
    lambda_rev: float
    lambda_c2: float
    residual_min: float


SUMMARY_RE = re.compile(r"run_\d+-[0-9]+_N(?P<N>\d+)_path_alpha(?P<alpha>[0-9.]+)/summary")


def parse_summary(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("lambda_c1:"):
            metrics["lambda_c1"] = float(line.split(":", 1)[1])
        elif line.startswith("lambda_revival:"):
            metrics["lambda_rev"] = float(line.split(":", 1)[1])
        elif line.startswith("lambda_c2:"):
            metrics["lambda_c2"] = float(line.split(":", 1)[1])
        elif line.startswith("residual_min:"):
            metrics["residual_min"] = float(line.split(":", 1)[1])
    missing = {k for k in ("lambda_c1", "lambda_rev", "lambda_c2", "residual_min") if k not in metrics}
    if missing:
        raise ValueError(f"Missing {missing} in {path}")
    return metrics


def collect_points() -> List[SweepPoint]:
    points: List[SweepPoint] = []
    for summary in sorted(glob.glob("outputs/run_20251117-22*_N4_path_alpha*/summary_*")) + sorted(
        glob.glob("outputs/run_20251117-23*_N4_path_alpha*/summary_*")
    ):
        match = SUMMARY_RE.search(summary)
        if not match:
            continue
        alpha = float(match.group("alpha"))
        data = parse_summary(Path(summary))
        points.append(
            SweepPoint(
                alpha=alpha,
                lambda_c1=data["lambda_c1"],
                lambda_rev=data["lambda_rev"],
                lambda_c2=data["lambda_c2"],
                residual_min=data["residual_min"],
            )
        )
    points.sort(key=lambda p: p.alpha)
    if not points:
        raise SystemExit("No alpha sweep summaries found.")
    return points


def plot(points: List[SweepPoint], save_path: Path) -> None:
    alphas = [p.alpha for p in points]
    c1 = [p.lambda_c1 for p in points]
    rev = [p.lambda_rev for p in points]
    c2 = [p.lambda_c2 for p in points]
    resid = [p.residual_min for p in points]

    regimes = [
        (0.1, 0.4, "Legacy plateau"),
        (0.5, 0.9, "Low-lambda plateau"),
        (1.0, 1.5, "Return / collapse"),
    ]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    for start, end, label in regimes:
        for ax in (ax_top, ax_bottom):
            ax.axvspan(start, end, color="#f1f1f1", alpha=0.6, zorder=0)
        ax_top.text((start + end) / 2, max(max(c1), max(rev), max(c2)) * 1.02, label, ha="center", va="bottom", fontsize=10)

    ax_top.plot(alphas, c1, marker="o", label=r"$\lambda_{c1}$")
    ax_top.plot(alphas, rev, marker="s", label=r"$\lambda_{\mathrm{rev}}$")
    ax_top.plot(alphas, c2, marker="^", label=r"$\lambda_{c2}$")
    ax_top.set_ylabel("Critical lambda")
    ax_top.legend(loc="best")

    ax_bottom.bar(alphas, resid, width=0.08, color="#4a90e2")
    ax_bottom.set_ylabel("Residual$_{min}$")
    ax_bottom.set_xlabel("alpha")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("N=4 path: alpha sweep summary", fontsize=14)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    pts = collect_points()
    output = Path("figures/alpha_sweep_N4_path.png")
    plot(pts, output)
    print(f"Saved alpha sweep figure to {output}")
