#!/usr/bin/env python3
"""
Run a background-subtracted TT matrix around a scalar Lambda background.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry import EmergentGeometryAnalyzer
from topologies import get_topology
from v3over9000.tensor_spin2 import TensorSpin2Analyzer


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    N: int
    topology: str
    alpha: float


DEFAULT_SCENARIOS: List[Scenario] = [
    Scenario("N4_star_alpha0.8", 4, "star", 0.8),
    Scenario("N4_path_alpha0.8", 4, "path", 0.8),
    Scenario("N4_cycle_alpha0.8", 4, "cycle", 0.8),
]


def _parse_scenarios(raw: str) -> List[Scenario]:
    if not raw.strip():
        return DEFAULT_SCENARIOS

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
        rows.append(
            Scenario(
                scenario_id=f"N{n_sites}_{topology}_alpha{alpha:.1f}",
                N=n_sites,
                topology=topology,
                alpha=alpha,
            )
        )
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


def _parse_str_list(raw: str) -> List[str]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("No string values parsed.")
    return vals


def _estimate_total_dim(n_sites: int, edge_count: int, bond_cutoff: int) -> int:
    return int((2 ** int(n_sites)) * (int(bond_cutoff) ** max(int(edge_count), 0)))


def _backend_settings(total_dim: int, backend: str, auto_dense_threshold: int) -> Tuple[str, str, str]:
    mode = (backend or "auto").lower()
    if mode not in {"auto", "dense", "iterative"}:
        raise ValueError(f"Unsupported backend '{backend}'.")
    if mode == "auto":
        mode = "dense" if int(total_dim) <= int(auto_dense_threshold) else "iterative"
    if mode == "dense":
        return "dense", "dense", "dense"
    return "sparse", "iterative", "krylov"


def _choose_source_site(
    topology: str,
    n_sites: int,
    lambda_profile: np.ndarray,
    source_mode: str,
) -> int:
    mode = (source_mode or "lambda_max").lower()
    if mode == "lambda_max":
        return int(np.argmax(lambda_profile)) if lambda_profile.size else 0
    if mode == "hub":
        return 0
    if mode == "center":
        return int(n_sites // 2)
    if mode == "probe_out":
        return int(get_topology(topology, n_sites).probes[0])
    raise ValueError(f"Unsupported source mode '{source_mode}'.")


def _edge_delta_lambda(edges: List[Tuple[int, int]], delta_site_lambda: np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return np.zeros(0, dtype=float)
    return np.asarray(
        [0.5 * (float(delta_site_lambda[u]) + float(delta_site_lambda[v])) for (u, v) in edges],
        dtype=float,
    )


def _fro_norm(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr))


def _compute_point(payload: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from core_qca import ExactQCA

    scenario_id = str(payload["scenario_id"])
    n_sites = int(payload["N"])
    topology_name = str(payload["topology"])
    alpha = float(payload["alpha"])
    lambda_val = float(payload["lambda"])
    hotspot_multiplier = float(payload["hotspot_multiplier"])
    frustration_time = float(payload["frustration_time"])
    bond_cutoff = int(payload["bond_cutoff"])
    delta_b = float(payload["deltaB"])
    kappa = float(payload["kappa"])
    k0 = int(payload["k0"])
    omega = float(payload["omega"])
    j0 = float(payload["J0"])
    backend = str(payload["backend"])
    auto_dense_threshold = int(payload["auto_dense_threshold"])
    n_modes = int(payload["n_modes"])
    n_angles = int(payload["n_angles"])
    source_mode = str(payload["source_mode"])

    topology = get_topology(topology_name, n_sites)
    total_dim = _estimate_total_dim(n_sites, len(topology.edges), bond_cutoff)
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
        "alpha": alpha,
        "probeOut": topology.probes[0],
        "probeIn": topology.probes[1],
        "edges": topology.edges,
    }
    qca = ExactQCA(
        n_sites,
        config,
        bond_cutoff=bond_cutoff,
        edges=topology.edges,
        hamiltonian_mode=hamiltonian_mode,
    )
    hotspot_config = dict(config)
    hotspot_config["lambda"] = lambda_val * hotspot_multiplier
    qca_hot = ExactQCA(
        n_sites,
        hotspot_config,
        bond_cutoff=bond_cutoff,
        edges=topology.edges,
        hamiltonian_mode=hamiltonian_mode,
    )

    ground = qca.get_ground_state(method=ground_method)
    frustrated = qca_hot.evolve_state(ground, frustration_time, method=evolve_method)

    geom = EmergentGeometryAnalyzer(rng_seed=0)
    tensor = TensorSpin2Analyzer()

    raw_spec = tensor.compute_spectrum(
        qca,
        frustrated,
        topology_name=topology_name,
        n_modes=n_modes,
        n_angles=n_angles,
    )
    raw_fit = tensor.fit_power_law(raw_spec.k, raw_spec.power, expected_power=2.0)

    site_lambda = geom.site_lambda_profile(qca, frustrated)
    source_site = _choose_source_site(
        topology=topology_name,
        n_sites=n_sites,
        lambda_profile=site_lambda,
        source_mode=source_mode,
    )
    vertex_shells = geom.vertex_graph_distances(n_sites, topology.edges, source_site)
    shell_ids, shell_means = geom.shell_average(site_lambda, vertex_shells)
    shell_map = {int(shell): float(mean) for shell, mean in zip(shell_ids, shell_means)}
    lambda_bg = np.asarray([shell_map[int(shell)] for shell in vertex_shells], dtype=float)
    delta_lambda = site_lambda - lambda_bg
    edge_delta = _edge_delta_lambda(topology.edges, delta_lambda)

    bg_spec = tensor.compute_spectrum_from_edge_field(
        edge_field=edge_delta,
        n=n_sites,
        edges=topology.edges,
        topology_name=topology_name,
        n_modes=n_modes,
        n_angles=n_angles,
    )
    bg_fit = tensor.fit_power_law(bg_spec.k, bg_spec.power, expected_power=2.0)
    edge_shells = geom.edge_shells_from_vertex_shells(topology.edges, vertex_shells)

    mean_occ, covariance = geom.bond_occupation_moments(qca, frustrated)
    cov_background = geom.shell_pair_average_covariance(covariance, edge_shells)
    cov_delta = covariance - cov_background
    low_mode_projector = geom.line_graph_low_mode_projector(topology.edges, max(1, int(shell_ids.size)))
    eye = np.eye(low_mode_projector.shape[0], dtype=float) if low_mode_projector.size else np.zeros((0, 0), dtype=float)
    harmonic_residual = (eye - low_mode_projector) @ covariance @ (eye - low_mode_projector)
    harmonic_residual = 0.5 * (harmonic_residual + harmonic_residual.T)

    covbg_spec = tensor.compute_spectrum_from_covariance(
        covariance=cov_delta,
        n=n_sites,
        edges=topology.edges,
        topology_name=topology_name,
        n_modes=n_modes,
        n_angles=n_angles,
    )
    covbg_fit = tensor.fit_power_law(covbg_spec.k, covbg_spec.power, expected_power=2.0)
    harmbg_spec = tensor.compute_spectrum_from_covariance(
        covariance=harmonic_residual,
        n=n_sites,
        edges=topology.edges,
        topology_name=topology_name,
        n_modes=n_modes,
        n_angles=n_angles,
    )
    harmbg_fit = tensor.fit_power_law(harmbg_spec.k, harmbg_spec.power, expected_power=2.0)

    shell_profile_rows: List[Dict[str, Any]] = []
    for site_idx in range(n_sites):
        shell_profile_rows.append(
            {
                "scenario_id": scenario_id,
                "N": n_sites,
                "topology": topology_name,
                "alpha": alpha,
                "lambda": lambda_val,
                "bond_cutoff": bond_cutoff,
                "hotspot_multiplier": hotspot_multiplier,
                "deltaB": delta_b,
                "kappa": kappa,
                "backend": backend,
                "source_site": source_site,
                "site": site_idx,
                "shell": int(vertex_shells[site_idx]),
                "lambda_site": float(site_lambda[site_idx]),
                "lambda_bg": float(lambda_bg[site_idx]),
                "delta_lambda": float(delta_lambda[site_idx]),
            }
        )

    edge_profile_rows: List[Dict[str, Any]] = []
    for edge_idx, (u, v) in enumerate(topology.edges):
        edge_profile_rows.append(
            {
                "scenario_id": scenario_id,
                "N": n_sites,
                "topology": topology_name,
                "alpha": alpha,
                "lambda": lambda_val,
                "bond_cutoff": bond_cutoff,
                "hotspot_multiplier": hotspot_multiplier,
                "deltaB": delta_b,
                "kappa": kappa,
                "backend": backend,
                "source_site": source_site,
                "edge": edge_idx,
                "u": int(u),
                "v": int(v),
                "edge_shell": int(edge_shells[edge_idx]) if edge_idx < edge_shells.size else -1,
                "mean_occ": float(mean_occ[edge_idx]) if edge_idx < mean_occ.size else float("nan"),
                "delta_lambda_edge": float(edge_delta[edge_idx]) if edge_idx < edge_delta.size else float("nan"),
            }
        )

    return {
        "scenario_id": scenario_id,
        "N": n_sites,
        "topology": topology_name,
        "alpha": alpha,
        "lambda": lambda_val,
        "bond_cutoff": bond_cutoff,
        "hotspot_multiplier": hotspot_multiplier,
        "frustration_time": frustration_time,
        "deltaB": delta_b,
        "kappa": kappa,
        "k0": k0,
        "backend": backend,
        "effective_backend": hamiltonian_mode,
        "total_dim": total_dim,
        "source_site": source_site,
        "raw_power": float(raw_fit["measured_power"]),
        "raw_residual": float(raw_fit["residual"]),
        "raw_r2": float(raw_fit["r2"]),
        "bg_power": float(bg_fit["measured_power"]),
        "bg_residual": float(bg_fit["residual"]),
        "bg_r2": float(bg_fit["r2"]),
        "covbg_power": float(covbg_fit["measured_power"]),
        "covbg_residual": float(covbg_fit["residual"]),
        "covbg_r2": float(covbg_fit["r2"]),
        "harmbg_power": float(harmbg_fit["measured_power"]),
        "harmbg_residual": float(harmbg_fit["residual"]),
        "harmbg_r2": float(harmbg_fit["r2"]),
        "delta_lambda_edge_l2": float(np.linalg.norm(edge_delta)),
        "delta_lambda_edge_maxabs": float(np.max(np.abs(edge_delta))) if edge_delta.size else 0.0,
        "covariance_l2": _fro_norm(covariance),
        "covariance_bg_l2": _fro_norm(cov_background),
        "covariance_delta_l2": _fro_norm(cov_delta),
        "covariance_delta_maxabs": float(np.max(np.abs(cov_delta))) if cov_delta.size else 0.0,
        "harmonic_rank": int(shell_ids.size),
        "harmonic_residual_l2": _fro_norm(harmonic_residual),
        "harmonic_residual_maxabs": float(np.max(np.abs(harmonic_residual))) if harmonic_residual.size else 0.0,
        "shell_count": int(shell_ids.size),
        "lambda_shell_json": json.dumps(shell_map, sort_keys=True),
        "shell_profile_rows": shell_profile_rows,
        "edge_profile_rows": edge_profile_rows,
    }


def _build_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if points.empty:
        return pd.DataFrame()

    group_cols = [
        "scenario_id",
        "N",
        "topology",
        "alpha",
        "bond_cutoff",
        "hotspot_multiplier",
        "deltaB",
        "kappa",
        "backend",
    ]
    for keys, subset in points.groupby(group_cols):
        subset = subset.sort_values("lambda")
        best_raw = subset.loc[subset["raw_residual"].idxmin()]
        best_bg = subset.loc[subset["bg_residual"].idxmin()]
        best_covbg = subset.loc[subset["covbg_residual"].idxmin()]
        best_harmbg = subset.loc[subset["harmbg_residual"].idxmin()]
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "points": int(len(subset)),
                "best_raw_lambda": float(best_raw["lambda"]),
                "best_raw_power": float(best_raw["raw_power"]),
                "best_raw_residual": float(best_raw["raw_residual"]),
                "best_bg_lambda": float(best_bg["lambda"]),
                "best_bg_power": float(best_bg["bg_power"]),
                "best_bg_residual": float(best_bg["bg_residual"]),
                "best_covbg_lambda": float(best_covbg["lambda"]),
                "best_covbg_power": float(best_covbg["covbg_power"]),
                "best_covbg_residual": float(best_covbg["covbg_residual"]),
                "best_harmbg_lambda": float(best_harmbg["lambda"]),
                "best_harmbg_power": float(best_harmbg["harmbg_power"]),
                "best_harmbg_residual": float(best_harmbg["harmbg_residual"]),
                "bg_minus_raw_residual": float(best_bg["bg_residual"] - best_raw["raw_residual"]),
                "covbg_minus_raw_residual": float(best_covbg["covbg_residual"] - best_raw["raw_residual"]),
                "harmbg_minus_raw_residual": float(best_harmbg["harmbg_residual"] - best_raw["raw_residual"]),
                "source_site_at_best_bg": int(best_bg["source_site"]),
                "source_site_at_best_covbg": int(best_covbg["source_site"]),
                "source_site_at_best_harmbg": int(best_harmbg["source_site"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# TT Background-Subtracted Matrix")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for key, value in config.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Best Background-Subtracted Rows")
    lines.append("")
    if summary.empty:
        lines.append("- No points computed.")
    else:
        for _, row in summary.iterrows():
            lines.append(
                f"- `{row['scenario_id']}` chi={int(row['bond_cutoff'])} hotspot={row['hotspot_multiplier']:.2f} "
                f"dB={row['deltaB']:.2f} kappa={row['kappa']:.3f} backend={row['backend']}: "
                f"best_bg_lambda={row['best_bg_lambda']:.3f}, best_bg_power={row['best_bg_power']:.3f}, "
                f"best_bg_residual={row['best_bg_residual']:.3f}; "
                f"best_covbg_lambda={row['best_covbg_lambda']:.3f}, best_covbg_power={row['best_covbg_power']:.3f}, "
                f"best_covbg_residual={row['best_covbg_residual']:.3f}; "
                f"best_harmbg_lambda={row['best_harmbg_lambda']:.3f}, best_harmbg_power={row['best_harmbg_power']:.3f}, "
                f"best_harmbg_residual={row['best_harmbg_residual']:.3f}; "
                f"raw_residual={row['best_raw_residual']:.3f}"
            )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary.empty:
        return None
    plt.figure(figsize=(9, 6))
    for scenario_id, subset in summary.groupby("scenario_id"):
        subset = subset.sort_values("bond_cutoff")
        plt.plot(
            subset["bond_cutoff"],
            subset["best_harmbg_power"],
            marker="o",
            linewidth=1.8,
            label=f"{scenario_id} harmonic-bg-subtracted",
        )
    plt.axhline(2.0, color="black", linestyle="--", linewidth=1.4, label="target power 2")
    plt.xlabel("bond cutoff")
    plt.ylabel("best harmonic-background power")
    plt.title("Harmonic-Background TT Power by Cutoff")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "harmbg_tt_power_by_cutoff.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    return out_path


def run_matrix(
    *,
    scenarios: List[Scenario],
    lambdas: List[float],
    bond_cutoffs: List[int],
    hotspot_multipliers: List[float],
    delta_bs: List[float],
    kappas: List[float],
    backends: List[str],
    frustration_time: float,
    k0: int,
    omega: float,
    j0: float,
    source_mode: str,
    n_modes: int,
    n_angles: int,
    auto_dense_threshold: int,
    workers: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for lam in lambdas:
            for cutoff in bond_cutoffs:
                for hotspot in hotspot_multipliers:
                    for delta_b in delta_bs:
                        for kappa in kappas:
                            for backend in backends:
                                payloads.append(
                                    {
                                        "scenario_id": scenario.scenario_id,
                                        "N": scenario.N,
                                        "topology": scenario.topology,
                                        "alpha": scenario.alpha,
                                        "lambda": float(lam),
                                        "bond_cutoff": int(cutoff),
                                        "hotspot_multiplier": float(hotspot),
                                        "frustration_time": float(frustration_time),
                                        "deltaB": float(delta_b),
                                        "kappa": float(kappa),
                                        "k0": int(k0),
                                        "omega": float(omega),
                                        "J0": float(j0),
                                        "backend": str(backend),
                                        "auto_dense_threshold": int(auto_dense_threshold),
                                        "n_modes": int(n_modes),
                                        "n_angles": int(n_angles),
                                        "source_mode": source_mode,
                                    }
                                )

    rows: List[Dict[str, Any]] = []
    shell_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max(int(workers), 1)) as pool:
        futures = [pool.submit(_compute_point, payload) for payload in payloads]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            shell_rows.extend(result.pop("shell_profile_rows"))
            edge_rows.extend(result.pop("edge_profile_rows"))
            rows.append(result)
            print(f"[{idx}/{len(futures)}] {result['scenario_id']} lambda={result['lambda']:.3f} complete", flush=True)

    points = pd.DataFrame(rows).sort_values(
        ["scenario_id", "bond_cutoff", "hotspot_multiplier", "deltaB", "kappa", "backend", "lambda"]
    )
    shell_df = pd.DataFrame(shell_rows).sort_values(["scenario_id", "lambda", "site"])
    edge_df = pd.DataFrame(edge_rows).sort_values(["scenario_id", "lambda", "edge"])
    summary = _build_summary(points)

    points.to_csv(output_dir / "points.csv", index=False)
    shell_df.to_csv(output_dir / "shell_profiles.csv", index=False)
    edge_df.to_csv(output_dir / "edge_profiles.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    config = {
        "scenarios": [s.scenario_id for s in scenarios],
        "lambdas": lambdas,
        "bond_cutoffs": bond_cutoffs,
        "hotspot_multipliers": hotspot_multipliers,
        "deltaBs": delta_bs,
        "kappas": kappas,
        "backends": backends,
        "frustration_time": frustration_time,
        "source_mode": source_mode,
        "n_modes": n_modes,
        "n_angles": n_angles,
        "auto_dense_threshold": auto_dense_threshold,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output_dir, summary, config)
    _plot_summary(summary, output_dir)

    print("\nMatrix complete.")
    print(points.to_string(index=False))
    print(f"\nOUTPUT_DIR={output_dir}")
    print(f"POINTS={output_dir / 'points.csv'}")
    print(f"SUMMARY={output_dir / 'summary.csv'}")
    print(f"SHELLS={output_dir / 'shell_profiles.csv'}")
    print(f"EDGES={output_dir / 'edge_profiles.csv'}")
    print(f"REPORT={output_dir / 'report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background-subtracted TT matrix")
    parser.add_argument("--scenarios", type=str, default="", help="Comma-separated scenarios N<sites>:<topology>:<alpha>")
    parser.add_argument("--lambdas", type=str, default="0.6,1.0,1.3", help="Comma-separated lambda list")
    parser.add_argument("--bond-cutoffs", type=str, default="4", help="Comma-separated chi values")
    parser.add_argument("--hotspot-multipliers", type=str, default="1.5", help="Comma-separated hotspot multipliers")
    parser.add_argument("--deltaBs", type=str, default="6.5", help="Comma-separated deltaB values")
    parser.add_argument("--kappas", type=str, default="0.2", help="Comma-separated kappa values")
    parser.add_argument("--backends", type=str, default="auto", help="Comma-separated backends: auto,dense,iterative")
    parser.add_argument("--frustration-time", type=float, default=1.0)
    parser.add_argument("--k0", type=int, default=4)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--J0", type=float, default=0.01)
    parser.add_argument("--source-mode", type=str, default="lambda_max", help="lambda_max|hub|center|probe_out")
    parser.add_argument("--n-modes", type=int, default=16)
    parser.add_argument("--n-angles", type=int, default=24)
    parser.add_argument("--auto-dense-threshold", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="", help="Optional explicit output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = _parse_scenarios(args.scenarios)
    lambdas = _parse_float_list(args.lambdas)
    bond_cutoffs = _parse_int_list(args.bond_cutoffs)
    hotspot_multipliers = _parse_float_list(args.hotspot_multipliers)
    delta_bs = _parse_float_list(args.deltaBs)
    kappas = _parse_float_list(args.kappas)
    backends = _parse_str_list(args.backends)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("outputs") / f"tt_background_subtracted_matrix_{ts}"

    run_matrix(
        scenarios=scenarios,
        lambdas=lambdas,
        bond_cutoffs=bond_cutoffs,
        hotspot_multipliers=hotspot_multipliers,
        delta_bs=delta_bs,
        kappas=kappas,
        backends=backends,
        frustration_time=float(args.frustration_time),
        k0=int(args.k0),
        omega=float(args.omega),
        j0=float(args.J0),
        source_mode=str(args.source_mode),
        n_modes=int(args.n_modes),
        n_angles=int(args.n_angles),
        auto_dense_threshold=int(args.auto_dense_threshold),
        workers=int(args.workers),
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
