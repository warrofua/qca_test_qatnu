#!/usr/bin/env python3
"""
Run a topology sweep for the correlator-based spin-2 proxy.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running directly from scripts/ while importing repo modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry import EmergentGeometryAnalyzer
from topologies import get_topology


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    N: int
    topology: str
    alpha: float


DEFAULT_SCENARIOS: List[Scenario] = [
    Scenario("N4_path_alpha0.8", 4, "path", 0.8),
    Scenario("N4_cycle_alpha0.8", 4, "cycle", 0.8),
    Scenario("N4_star_alpha0.8", 4, "star", 0.8),
]


def _parse_scenarios(raw: str) -> List[Scenario]:
    """
    Parse comma-separated scenario descriptors:
      N4:path:0.8
      N5:cycle:1.0
    """
    if not raw.strip():
        return DEFAULT_SCENARIOS

    rows: List[Scenario] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid scenario token '{token}'. Expected format N<sites>:<topology>:<alpha>."
            )
        n_text, topology, alpha_text = parts
        if not n_text.lower().startswith("n"):
            raise ValueError(f"Invalid scenario token '{token}': first part must start with 'N'.")
        N = int(n_text[1:])
        alpha = float(alpha_text)
        scenario_id = f"N{N}_{topology}_alpha{alpha:.1f}"
        rows.append(Scenario(scenario_id=scenario_id, N=N, topology=topology, alpha=alpha))

    if not rows:
        raise ValueError("No scenarios parsed.")
    return rows


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("No float values parsed.")
    return vals


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("No integer values parsed.")
    return vals


def _compute_spin2_point(payload: Dict[str, float | int | str]) -> Dict[str, float | int | str]:
    """
    Compute one scenario/lambda point for correlator-based spin-2 diagnostics.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from core_qca import ExactQCA

    scenario_id = str(payload["scenario_id"])
    N = int(payload["N"])
    topology = str(payload["topology"])
    alpha = float(payload["alpha"])
    lambda_val = float(payload["lambda"])
    hotspot_multiplier = float(payload["hotspot_multiplier"])
    frustration_time = float(payload["frustration_time"])
    bond_cutoff = int(payload["bond_cutoff"])
    deltaB = float(payload["deltaB"])
    kappa = float(payload["kappa"])
    k0 = int(payload["k0"])
    omega = float(payload["omega"])
    J0 = float(payload["J0"])
    virtual_size = int(payload["virtual_size"])

    topology_spec = get_topology(topology, N)
    edges = topology_spec.edges
    probe_out, probe_in = topology_spec.probes

    config = {
        "omega": omega,
        "deltaB": deltaB,
        "lambda": lambda_val,
        "kappa": kappa,
        "k0": k0,
        "bondCutoff": bond_cutoff,
        "J0": J0,
        "gamma": 0.0,
        "alpha": alpha,
        "probeOut": probe_out,
        "probeIn": probe_in,
        "edges": edges,
    }
    qca = ExactQCA(N, config, bond_cutoff=bond_cutoff, edges=edges)
    hotspot_config = dict(config)
    hotspot_config["lambda"] = lambda_val * hotspot_multiplier
    qca_hot = ExactQCA(N, hotspot_config, bond_cutoff=bond_cutoff, edges=edges)

    ground = qca.get_ground_state()
    frustrated = qca_hot.evolve_state(ground, frustration_time)

    geom = EmergentGeometryAnalyzer(rng_seed=0)
    ks, psd = geom.spin2_from_bond_correlators(qca, frustrated, virtual_size=virtual_size)
    method = "bond_correlator"

    if ks.size < 3 or psd.size < 3:
        chi_profile = [qca.get_bond_dimension(edge_idx, frustrated) for edge_idx in range(qca.edge_count)]
        ks, psd = geom.spin2_from_chi_profile(chi_profile)
        method = "chi_fallback"

    fit = geom.analyze_spin2_scaling(ks, psd, expected_power=2.0)

    power = float(fit["measured_power"]) if np.isfinite(fit["measured_power"]) else float("nan")
    slope = -power if np.isfinite(power) else float("nan")
    target_error = abs(power - 2.0) if np.isfinite(power) else float("nan")

    lambda_out = float(qca.count_circuit_depth(probe_out, frustrated))
    lambda_in = float(qca.count_circuit_depth(probe_in, frustrated))

    return {
        "scenario_id": scenario_id,
        "N": N,
        "topology": topology,
        "alpha": alpha,
        "lambda": lambda_val,
        "bond_cutoff": bond_cutoff,
        "spin2_power": power,
        "spin2_slope": slope,
        "spin2_target_error": target_error,
        "fit_quality": bool(fit["fit_quality"]),
        "num_modes": int(ks.size),
        "method": method,
        "lambda_out": lambda_out,
        "lambda_in": lambda_in,
        "edge_count": int(qca.edge_count),
    }


def _plot_slope_vs_lambda(points: pd.DataFrame, out_dir: Path) -> Path:
    plt.figure(figsize=(9, 6))
    groups = points.groupby(["scenario_id", "bond_cutoff"])
    for (scenario_id, cutoff), subset in sorted(groups, key=lambda item: (item[0][0], item[0][1])):
        subset = subset.sort_values("lambda")
        label = f"{scenario_id} (chi={int(cutoff)})"
        plt.plot(subset["lambda"], subset["spin2_slope"], marker="o", linewidth=1.8, label=label)
    plt.axhline(-2.0, color="black", linestyle="--", linewidth=1.5, label="target slope -2")
    plt.xlabel("lambda")
    plt.ylabel("spin2 slope")
    plt.title("Correlator Spin-2 Slope vs Lambda")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "spin2_slope_vs_lambda.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    return out_path


def _build_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for (scenario_id, cutoff), subset in points.groupby(["scenario_id", "bond_cutoff"]):
        subset = subset.sort_values("lambda")
        valid = subset[np.isfinite(subset["spin2_power"])]
        if valid.empty:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "bond_cutoff": int(cutoff),
                    "points": int(len(subset)),
                    "valid_points": 0,
                    "best_lambda": None,
                    "best_slope": None,
                    "best_power": None,
                    "best_target_error": None,
                    "mean_target_error": None,
                    "median_target_error": None,
                }
            )
            continue

        best_idx = valid["spin2_target_error"].idxmin()
        best = valid.loc[best_idx]
        rows.append(
            {
                "scenario_id": scenario_id,
                "bond_cutoff": int(cutoff),
                "points": int(len(subset)),
                "valid_points": int(len(valid)),
                "best_lambda": float(best["lambda"]),
                "best_slope": float(best["spin2_slope"]),
                "best_power": float(best["spin2_power"]),
                "best_target_error": float(best["spin2_target_error"]),
                "mean_target_error": float(valid["spin2_target_error"].mean()),
                "median_target_error": float(valid["spin2_target_error"].median()),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["best_target_error", "mean_target_error", "scenario_id", "bond_cutoff"],
        na_position="last",
    )
    return out


def _write_report(
    out_dir: Path,
    points: pd.DataFrame,
    summary: pd.DataFrame,
    plot_path: Path,
    config: Dict[str, float | int | str],
) -> Path:
    lines: List[str] = []
    lines.append("# Spin-2 Correlator Topology Sweep")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Sweep configuration")
    lines.append("")
    lines.append(f"- hotspot_multiplier: {config['hotspot_multiplier']}")
    lines.append(f"- frustration_time: {config['frustration_time']}")
    lines.append(f"- bond_cutoffs: {config['bond_cutoffs']}")
    lines.append(f"- deltaB: {config['deltaB']}")
    lines.append(f"- kappa: {config['kappa']}")
    lines.append(f"- k0: {config['k0']}")
    lines.append(f"- virtual_size: {config['virtual_size']}")
    lines.append("")
    lines.append("## Ranked scenarios (closest to slope -2)")
    lines.append("")
    for _, row in summary.iterrows():
        lines.append(
            "- {sid} (chi={chi}): best λ={lam}, best slope={slope}, |power-2|={err:.3f}, mean |power-2|={mean_err:.3f}".format(
                sid=row["scenario_id"],
                chi=int(row["bond_cutoff"]),
                lam=f"{row['best_lambda']:.3f}" if pd.notna(row["best_lambda"]) else "NA",
                slope=f"{row['best_slope']:.3f}" if pd.notna(row["best_slope"]) else "NA",
                err=row["best_target_error"] if pd.notna(row["best_target_error"]) else float("nan"),
                mean_err=row["mean_target_error"] if pd.notna(row["mean_target_error"]) else float("nan"),
            )
        )
    lines.append("")
    lines.append("## Artifact index")
    lines.append("")
    lines.append(f"- points CSV: `{out_dir / 'spin2_points.csv'}`")
    lines.append(f"- summary CSV: `{out_dir / 'spin2_summary.csv'}`")
    lines.append(f"- slope plot: `{plot_path}`")
    lines.append("")
    report_path = out_dir / "spin2_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_sweep(
    *,
    scenarios: Iterable[Scenario],
    lambdas: Iterable[float],
    hotspot_multiplier: float,
    frustration_time: float,
    bond_cutoffs: Iterable[int],
    deltaB: float,
    kappa: float,
    k0: int,
    omega: float,
    J0: float,
    virtual_size: int,
    workers: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff_values = sorted({int(c) for c in bond_cutoffs})
    if not cutoff_values:
        raise ValueError("bond_cutoffs must include at least one value.")
    tasks: List[Dict[str, float | int | str]] = []
    for scenario in scenarios:
        for cutoff in cutoff_values:
            for lam in lambdas:
                tasks.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "N": scenario.N,
                        "topology": scenario.topology,
                        "alpha": scenario.alpha,
                        "lambda": float(lam),
                        "hotspot_multiplier": float(hotspot_multiplier),
                        "frustration_time": float(frustration_time),
                        "bond_cutoff": int(cutoff),
                        "deltaB": float(deltaB),
                        "kappa": float(kappa),
                        "k0": int(k0),
                        "omega": float(omega),
                        "J0": float(J0),
                        "virtual_size": int(virtual_size),
                    }
                )

    rows: List[Dict[str, float | int | str]] = []
    print(f"Running {len(tasks)} spin-2 points with {workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_compute_spin2_point, payload) for payload in tasks]
        total = len(futures)
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{idx}/{total}] {row['scenario_id']} λ={float(row['lambda']):.3f} "
                f"chi={int(row['bond_cutoff'])} slope={float(row['spin2_slope']):.3f} method={row['method']}",
                flush=True,
            )

    points = pd.DataFrame(rows).sort_values(["scenario_id", "bond_cutoff", "lambda"])
    points_path = output_dir / "spin2_points.csv"
    points.to_csv(points_path, index=False)

    summary = _build_summary(points)
    summary_path = output_dir / "spin2_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_path = _plot_slope_vs_lambda(points, output_dir)

    report_path = _write_report(
        out_dir=output_dir,
        points=points,
        summary=summary,
        plot_path=plot_path,
        config={
            "hotspot_multiplier": hotspot_multiplier,
            "frustration_time": frustration_time,
            "bond_cutoffs": ",".join(str(c) for c in cutoff_values),
            "deltaB": deltaB,
            "kappa": kappa,
            "k0": k0,
            "virtual_size": virtual_size,
        },
    )

    print("\nSweep completed.")
    print(summary.to_string(index=False))
    print(f"\nOUTPUT_DIR={output_dir}")
    print(f"POINTS={points_path}")
    print(f"SUMMARY={summary_path}")
    print(f"PLOT={plot_path}")
    print(f"REPORT={report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlator spin-2 topology sweep")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated descriptors (e.g. N4:path:0.8,N4:cycle:0.8). Default: N4 path/cycle/star.",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.4",
        help="Comma-separated lambda values.",
    )
    parser.add_argument("--hotspot-multiplier", type=float, default=1.5, help="Hotspot multiplier")
    parser.add_argument("--frustration-time", type=float, default=1.0, help="Frustration evolution time")
    parser.add_argument("--bond-cutoff", type=int, default=4, help="Bond cutoff (chi)")
    parser.add_argument(
        "--bond-cutoffs",
        type=str,
        default="",
        help="Comma-separated bond cutoffs. Overrides --bond-cutoff when provided.",
    )
    parser.add_argument("--deltaB", type=float, default=6.5, help="Bond ladder spacing")
    parser.add_argument("--kappa", type=float, default=0.2, help="Degree penalty")
    parser.add_argument("--k0", type=int, default=4, help="Target degree")
    parser.add_argument("--omega", type=float, default=1.0, help="Bare frequency")
    parser.add_argument("--J0", type=float, default=0.01, help="Ising coupling")
    parser.add_argument("--virtual-size", type=int, default=256, help="Virtual lattice size for PSD")
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)), help="Worker count")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = _parse_scenarios(args.scenarios)
    lambdas = _parse_float_list(args.lambdas)
    if args.bond_cutoffs.strip():
        bond_cutoffs = _parse_int_list(args.bond_cutoffs)
    else:
        bond_cutoffs = [int(args.bond_cutoff)]

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path("outputs") / f"spin2_correlator_sweep_{ts}"

    run_sweep(
        scenarios=scenarios,
        lambdas=lambdas,
        hotspot_multiplier=float(args.hotspot_multiplier),
        frustration_time=float(args.frustration_time),
        bond_cutoffs=bond_cutoffs,
        deltaB=float(args.deltaB),
        kappa=float(args.kappa),
        k0=int(args.k0),
        omega=float(args.omega),
        J0=float(args.J0),
        virtual_size=int(args.virtual_size),
        workers=max(1, int(args.workers)),
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
