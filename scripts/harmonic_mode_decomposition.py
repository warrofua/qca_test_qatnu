#!/usr/bin/env python3
"""
Decompose bond covariance into line-graph Laplacian modes.

This is used to test whether a harmonic-background TT signal is just the
leftover highest-frequency line-graph mode rather than a robust tensor sector.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_qca import ExactQCA
from geometry import EmergentGeometryAnalyzer
from topologies import get_topology
from v3over9000.tensor_spin2 import TensorSpin2Analyzer


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    N: int
    topology: str
    alpha: float


def _parse_scenarios(raw: str) -> List[Scenario]:
    rows: List[Scenario] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid scenario token '{token}'. Expected N<sites>:<topology>:<alpha>.")
        n_text, topology, alpha_text = parts
        if not n_text.lower().startswith("n"):
            raise ValueError(f"Invalid scenario token '{token}': first part must start with 'N'.")
        n_sites = int(n_text[1:])
        alpha = float(alpha_text)
        rows.append(Scenario(f"N{n_sites}_{topology}_alpha{alpha:.1f}", n_sites, topology, alpha))
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


def _backend_settings(total_dim: int, backend: str, auto_dense_threshold: int) -> tuple[str, str, str]:
    mode = (backend or "auto").lower()
    if mode not in {"auto", "dense", "iterative"}:
        raise ValueError(f"Unsupported backend '{backend}'.")
    if mode == "auto":
        mode = "dense" if int(total_dim) <= int(auto_dense_threshold) else "iterative"
    if mode == "dense":
        return "dense", "dense", "dense"
    return "sparse", "iterative", "krylov"


def _estimate_total_dim(n_sites: int, edge_count: int, bond_cutoff: int) -> int:
    return int((2 ** int(n_sites)) * (int(bond_cutoff) ** max(int(edge_count), 0)))


def _choose_source_site(lambda_profile: np.ndarray) -> int:
    return int(np.argmax(lambda_profile)) if lambda_profile.size else 0


def _analyze_point(
    *,
    scenario: Scenario,
    lambda_val: float,
    bond_cutoff: int,
    hotspot_multiplier: float,
    frustration_time: float,
    delta_b: float,
    kappa: float,
    k0: int,
    omega: float,
    j0: float,
    backend: str,
    auto_dense_threshold: int,
    n_modes: int,
    n_angles: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    topology = get_topology(scenario.topology, scenario.N)
    total_dim = _estimate_total_dim(scenario.N, len(topology.edges), bond_cutoff)
    hamiltonian_mode, ground_method, evolve_method = _backend_settings(
        total_dim=total_dim,
        backend=backend,
        auto_dense_threshold=auto_dense_threshold,
    )

    config = {
        "omega": omega,
        "deltaB": delta_b,
        "lambda": lambda_val,
        "kappa": kappa,
        "k0": k0,
        "bondCutoff": bond_cutoff,
        "J0": j0,
        "gamma": 0.0,
        "alpha": scenario.alpha,
        "probeOut": topology.probes[0],
        "probeIn": topology.probes[1],
        "edges": topology.edges,
    }
    qca = ExactQCA(
        scenario.N,
        config,
        bond_cutoff=bond_cutoff,
        edges=topology.edges,
        hamiltonian_mode=hamiltonian_mode,
    )
    hotspot_config = dict(config)
    hotspot_config["lambda"] = lambda_val * hotspot_multiplier
    qca_hot = ExactQCA(
        scenario.N,
        hotspot_config,
        bond_cutoff=bond_cutoff,
        edges=topology.edges,
        hamiltonian_mode=hamiltonian_mode,
    )

    geom = EmergentGeometryAnalyzer(rng_seed=0)
    tensor = TensorSpin2Analyzer()
    ground = qca.get_ground_state(method=ground_method)
    frustrated = qca_hot.evolve_state(ground, frustration_time, method=evolve_method)

    lambda_profile = geom.site_lambda_profile(qca, frustrated)
    source_site = _choose_source_site(lambda_profile)
    vertex_shells = geom.vertex_graph_distances(scenario.N, topology.edges, source_site)
    shell_ids, _ = geom.shell_average(lambda_profile, vertex_shells)
    harmonic_rank = max(1, int(shell_ids.size))

    _, covariance = geom.bond_occupation_moments(qca, frustrated)
    adjacency = geom.line_graph_adjacency(topology.edges)
    degree = np.diag(np.sum(adjacency, axis=1))
    lap = degree - adjacency
    evals, evecs = np.linalg.eigh(lap)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    keep = min(harmonic_rank, evecs.shape[1])
    low_basis = evecs[:, :keep]
    high_basis = evecs[:, keep:]

    covariance_hat = evecs.T @ covariance @ evecs
    low_block = covariance_hat[:keep, :keep]
    high_block = covariance_hat[keep:, keep:]
    cross_block = covariance_hat[:keep, keep:]
    low_projector = low_basis @ low_basis.T if low_basis.size else np.zeros_like(covariance)
    eye = np.eye(covariance.shape[0], dtype=float)
    harmonic_residual = (eye - low_projector) @ covariance @ (eye - low_projector)
    harmonic_residual = 0.5 * (harmonic_residual + harmonic_residual.T)
    harm_spec = tensor.compute_spectrum_from_covariance(
        covariance=harmonic_residual,
        n=scenario.N,
        edges=topology.edges,
        topology_name=scenario.topology,
        n_modes=n_modes,
        n_angles=n_angles,
    )
    harm_fit = tensor.fit_power_law(harm_spec.k, harm_spec.power, expected_power=2.0)

    summary = {
        "scenario_id": scenario.scenario_id,
        "N": scenario.N,
        "topology": scenario.topology,
        "alpha": scenario.alpha,
        "lambda": lambda_val,
        "bond_cutoff": bond_cutoff,
        "backend": backend,
        "effective_backend": hamiltonian_mode,
        "source_site": source_site,
        "shell_count": int(shell_ids.size),
        "harmonic_rank": int(keep),
        "edge_count": int(len(topology.edges)),
        "high_mode_dim": int(max(0, evecs.shape[1] - keep)),
        "harmbg_power": float(harm_fit["measured_power"]),
        "harmbg_residual": float(harm_fit["residual"]),
        "harmonic_residual_l2": float(np.linalg.norm(harmonic_residual)),
        "harmonic_residual_rank_numeric": int(np.linalg.matrix_rank(harmonic_residual, tol=1e-10)),
        "high_block_l2": float(np.linalg.norm(high_block)),
        "cross_block_l2": float(np.linalg.norm(cross_block)),
        "top_high_mode_eigenvalue": float(evals[-1]) if evals.size else float("nan"),
        "top_high_mode_diag": float(high_block[-1, -1]) if high_block.size else float("nan"),
        "top_high_mode_weight_fraction": (
            float((high_block[-1, -1] ** 2) / np.sum(high_block**2)) if high_block.size and np.sum(high_block**2) > 0 else float("nan")
        ),
        "laplacian_eigenvalues_json": json.dumps([float(v) for v in evals], sort_keys=False),
    }

    mode_rows: List[Dict[str, Any]] = []
    total_high_sq = float(np.sum(high_block**2))
    for mode_idx, eig in enumerate(evals):
        is_low = mode_idx < keep
        diag_coeff = float(covariance_hat[mode_idx, mode_idx])
        if is_low or high_block.size == 0:
            high_weight_fraction = 0.0
        else:
            local_idx = mode_idx - keep
            row = high_block[local_idx, :]
            col = high_block[:, local_idx]
            high_weight_fraction = float((np.sum(row**2) + np.sum(col**2) - high_block[local_idx, local_idx] ** 2) / total_high_sq) if total_high_sq > 0 else 0.0
        mode_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "N": scenario.N,
                "topology": scenario.topology,
                "lambda": lambda_val,
                "bond_cutoff": bond_cutoff,
                "mode_index": int(mode_idx),
                "laplacian_eigenvalue": float(eig),
                "is_low_mode": bool(is_low),
                "diag_covariance_coeff": diag_coeff,
                "high_weight_fraction": high_weight_fraction,
            }
        )

    return summary, mode_rows


def _write_report(out_dir: Path, summary: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Harmonic Mode Decomposition")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    for _, row in summary.iterrows():
        lines.append(f"## {row['scenario_id']} lambda={row['lambda']:.3f} chi={int(row['bond_cutoff'])}")
        lines.append("")
        lines.append(f"- shell_count / harmonic_rank: {int(row['shell_count'])}")
        lines.append(f"- edge_count: {int(row['edge_count'])}")
        lines.append(f"- high_mode_dim: {int(row['high_mode_dim'])}")
        lines.append(f"- harmonic residual numeric rank: {int(row['harmonic_residual_rank_numeric'])}")
        lines.append(f"- harmonic TT power: {row['harmbg_power']:.6f}")
        lines.append(f"- harmonic TT residual to target 2: {row['harmbg_residual']:.6f}")
        lines.append(f"- high-block L2: {row['high_block_l2']:.6e}")
        lines.append(f"- cross-block L2: {row['cross_block_l2']:.6e}")
        lines.append(f"- top high-mode eigenvalue: {row['top_high_mode_eigenvalue']:.6f}")
        lines.append(f"- top high-mode weight fraction: {row['top_high_mode_weight_fraction']:.6f}")
        lines.append(f"- Laplacian eigenvalues: {row['laplacian_eigenvalues_json']}")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose harmonic-background covariance into line-graph modes")
    parser.add_argument("--scenarios", type=str, required=True, help="Comma-separated N<sites>:<topology>:<alpha>")
    parser.add_argument("--lambdas", type=str, required=True, help="Comma-separated lambda values")
    parser.add_argument("--bond-cutoffs", type=str, default="4", help="Comma-separated chi values")
    parser.add_argument("--hotspot-multiplier", type=float, default=1.5)
    parser.add_argument("--frustration-time", type=float, default=1.0)
    parser.add_argument("--deltaB", type=float, default=6.5)
    parser.add_argument("--kappa", type=float, default=0.2)
    parser.add_argument("--k0", type=int, default=4)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--J0", type=float, default=1.0)
    parser.add_argument("--backend", type=str, default="auto")
    parser.add_argument("--auto-dense-threshold", type=int, default=8000)
    parser.add_argument("--n-modes", type=int, default=16)
    parser.add_argument("--n-angles", type=int, default=24)
    parser.add_argument("--output-dir", type=str, default="", help="Optional explicit output dir")
    args = parser.parse_args()

    scenarios = _parse_scenarios(args.scenarios)
    lambdas = _parse_float_list(args.lambdas)
    bond_cutoffs = _parse_int_list(args.bond_cutoffs)
    if args.output_dir.strip():
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path("outputs") / f"harmonic_mode_decomposition_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    mode_rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for lam in lambdas:
            for cutoff in bond_cutoffs:
                summary, rows = _analyze_point(
                    scenario=scenario,
                    lambda_val=float(lam),
                    bond_cutoff=int(cutoff),
                    hotspot_multiplier=float(args.hotspot_multiplier),
                    frustration_time=float(args.frustration_time),
                    delta_b=float(args.deltaB),
                    kappa=float(args.kappa),
                    k0=int(args.k0),
                    omega=float(args.omega),
                    j0=float(args.J0),
                    backend=str(args.backend),
                    auto_dense_threshold=int(args.auto_dense_threshold),
                    n_modes=int(args.n_modes),
                    n_angles=int(args.n_angles),
                )
                summary_rows.append(summary)
                mode_rows.extend(rows)
                print(
                    f"{scenario.scenario_id} lambda={lam:.3f} chi={cutoff}: "
                    f"harmbg_power={summary['harmbg_power']:.6f}, high_dim={summary['high_mode_dim']}, "
                    f"rank={summary['harmonic_residual_rank_numeric']}",
                    flush=True,
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["scenario_id", "bond_cutoff", "lambda"])
    modes_df = pd.DataFrame(mode_rows).sort_values(["scenario_id", "bond_cutoff", "lambda", "mode_index"])
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    modes_df.to_csv(out_dir / "modes.csv", index=False)
    _write_report(out_dir, summary_df)
    print(f"OUTPUT_DIR={out_dir}")
    print(f"SUMMARY={out_dir / 'summary.csv'}")
    print(f"MODES={out_dir / 'modes.csv'}")
    print(f"REPORT={out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
