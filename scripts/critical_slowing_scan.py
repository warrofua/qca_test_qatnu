#!/usr/bin/env python3
"""
Deep-time bond-sector relaxation / critical-slowing scan.

Protocol:
- Prepare ground state of nominal Hamiltonian H(lambda).
- Quench to hotspot Hamiltonian H(lambda * hotspot_multiplier).
- Evolve for t in [0, t_max], measure bond-derived Lambda(t).
- Extract equilibration times from sustained closeness to tail average.

This targets the Appendix-D "critical slowing near revival" hypothesis.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_qca import ExactQCA
from topologies import get_topology


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one float.")
    return vals


def _parse_lambdas_by_topology(raw: str, topologies: Sequence[str]) -> Dict[str, List[float]]:
    """
    Format:
      path:0.6,1.0,1.2;cycle:0.3,0.5,0.7;star:1.0,1.3,1.5
    """
    if ":" not in raw:
        vals = _parse_float_list(raw)
        return {t: vals for t in topologies}
    out: Dict[str, List[float]] = {}
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    for p in parts:
        name, value = p.split(":", 1)
        out[name.strip().lower()] = _parse_float_list(value)
    missing = [t for t in topologies if t not in out]
    if missing:
        raise ValueError(f"Missing lambda list for topologies: {missing}")
    return out


def _parse_topologies(raw: str) -> List[str]:
    vals = [v.strip().lower() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one topology")
    return vals


def _incident_edges(n: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    out: List[List[int]] = [[] for _ in range(n)]
    for edge_idx, (u, v) in enumerate(edges):
        out[u].append(edge_idx)
        out[v].append(edge_idx)
    return out


def _decode_configs(bond_cutoff: int, edge_count: int) -> np.ndarray:
    if edge_count == 0:
        return np.zeros((1, 0), dtype=np.int16)
    shape = (bond_cutoff,) * edge_count
    return np.array(np.unravel_index(np.arange(bond_cutoff**edge_count), shape)).T.astype(np.int16)


def _equilibration_time(
    times: np.ndarray,
    values: np.ndarray,
    tail_frac: float,
    rel_tol: float,
    sustain: int,
) -> Tuple[float, float, float, float]:
    """
    Return:
      tau_eq, tail_mean, tail_std, threshold
    tau_eq is NaN if criterion is never met.
    """
    n = len(values)
    if n < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")

    tail_n = max(4, int(math.ceil(float(tail_frac) * n)))
    tail = values[-tail_n:]
    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail))

    dev = np.abs(values - tail_mean)
    scale = max(float(np.max(dev)), 1e-15)
    threshold = float(rel_tol) * scale

    w = int(max(1, sustain))
    if w > n:
        w = n
    for i in range(0, n - w + 1):
        if float(np.max(dev[i : i + w])) <= threshold:
            return float(times[i]), tail_mean, tail_std, threshold
    return float("nan"), tail_mean, tail_std, threshold


def _dephasing_time(
    times: np.ndarray,
    values: np.ndarray,
    tail_mean: float,
    rel_tol: float,
    sustain: int,
) -> float:
    """
    Running-average convergence time.
    Useful in closed finite systems where pointwise equilibration may not occur
    but time-averaged observables dephase.
    """
    n = len(values)
    if n < 4 or not np.isfinite(tail_mean):
        return float("nan")
    cumsum = np.cumsum(values, dtype=float)
    running_mean = cumsum / (np.arange(n, dtype=float) + 1.0)
    dev = np.abs(running_mean - float(tail_mean))
    scale = max(float(np.max(np.abs(values - tail_mean))), 1e-15)
    threshold = float(rel_tol) * scale
    w = int(max(1, sustain))
    if w > n:
        w = n
    for i in range(0, n - w + 1):
        if float(np.max(dev[i : i + w])) <= threshold:
            return float(times[i])
    return float("nan")


def _build_qca(
    n: int,
    edges: Sequence[Tuple[int, int]],
    bond_cutoff: int,
    lambda_value: float,
    delta_b: float,
    kappa: float,
    k0: float,
) -> ExactQCA:
    cfg = {
        "omega": 1.0,
        "deltaB": float(delta_b),
        "lambda": float(lambda_value),
        "kappa": float(kappa),
        "k0": float(k0),
        "bondCutoff": int(bond_cutoff),
        "J0": 0.01,
        "gamma": 0.0,
        "edges": list(edges),
        "probeOut": 0,
        "probeIn": 1 if n > 1 else 0,
    }
    return ExactQCA(N=n, config=cfg, bond_cutoff=int(bond_cutoff), edges=list(edges))


def _lambda_timeseries(
    qca_hot: ExactQCA,
    psi0: np.ndarray,
    times: np.ndarray,
    proxy: str,
) -> np.ndarray:
    """
    Return Lambda_i(t) as array shape [N, T].
    """
    eig = qca_hot.diagonalize()
    v = eig["eigenvectors"]
    e = eig["eigenvalues"]

    coeff0 = v.T @ psi0
    phases = np.exp(-1j * np.outer(e, times))
    states = v @ (coeff0[:, None] * phases)  # [dim, T]
    probs = np.abs(states) ** 2

    t_count = len(times)
    probs3 = probs.reshape(qca_hot.matter_dim, qca_hot.bond_dim, t_count)
    weights_bond = np.sum(probs3, axis=0)  # [bond_dim, T]

    configs = _decode_configs(qca_hot.bond_cutoff, qca_hot.edge_count).astype(float)  # [bond_dim, E]
    if configs.shape[1] == 0:
        return np.zeros((qca_hot.N, t_count), dtype=float)

    mean_n_edge = configs.T @ weights_bond  # [E, T]
    incident = _incident_edges(qca_hot.N, qca_hot.edges)

    lam_site = np.zeros((qca_hot.N, t_count), dtype=float)
    for site, inc in enumerate(incident):
        if not inc:
            continue
        vals = mean_n_edge[np.asarray(inc, dtype=int), :]  # [deg, T]
        if proxy == "log":
            lam_site[site, :] = np.sum(np.log1p(vals), axis=0)
        elif proxy == "linear":
            lam_site[site, :] = np.sum(vals, axis=0)
        else:
            raise ValueError(f"Unsupported proxy '{proxy}'")
    return lam_site


def _plot_tau(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(9, 5.4))
    for topo, subset in summary.groupby("topology"):
        subset = subset.sort_values("lambda")
        plt.plot(
            subset["lambda"],
            subset["tau_eq_probe"],
            marker="o",
            linewidth=1.8,
            label=f"{topo} probe-gap",
        )
    plt.xlabel("lambda")
    plt.ylabel("tau_eq (probe-gap)")
    plt.title("Deep-time equilibration time vs lambda")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tau_eq_probe_vs_lambda.png", dpi=220, bbox_inches="tight")
    plt.close()


def _plot_examples(example_rows: List[Dict[str, object]], out_dir: Path) -> None:
    if not example_rows:
        return
    for row in example_rows:
        times = np.asarray(row["times"], dtype=float)
        probe_gap = np.asarray(row["probe_gap"], dtype=float)
        global_lam = np.asarray(row["global_lambda"], dtype=float)
        topo = str(row["topology"])
        lam = float(row["lambda"])
        tau = float(row["tau_eq_probe"])
        tail_mean = float(row["tail_mean_probe"])
        thr = float(row["threshold_probe"])

        plt.figure(figsize=(8.8, 4.8))
        plt.plot(times, probe_gap, linewidth=1.8, label="probe gap: Lambda_in - Lambda_out")
        plt.plot(times, global_lam, linewidth=1.3, alpha=0.7, label="global mean Lambda")
        plt.axhline(tail_mean, color="black", linestyle="--", alpha=0.7, label="tail mean")
        plt.axhline(tail_mean + thr, color="gray", linestyle=":", alpha=0.6)
        plt.axhline(tail_mean - thr, color="gray", linestyle=":", alpha=0.6, label="eq threshold band")
        if np.isfinite(tau):
            plt.axvline(tau, color="purple", linestyle="--", linewidth=1.5, label=f"tau_eq={tau:.2f}")
        plt.xlabel("t")
        plt.ylabel("observable")
        plt.title(f"{topo} lambda={lam:.3f}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        fname = out_dir / f"timeseries_{topo}_lam{lam:.3f}.png"
        plt.savefig(fname, dpi=220, bbox_inches="tight")
        plt.close()


def _write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Critical Slowing Scan")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- N: {args.N}")
    lines.append(f"- topologies: {args.topologies}")
    lines.append(f"- lambdas: {args.lambdas}")
    lines.append(f"- bond_cutoff: {args.bond_cutoff}")
    lines.append(f"- hotspot_multiplier: {args.hotspot_multiplier}")
    lines.append(f"- t_max: {args.t_max}")
    lines.append(f"- n_times: {args.n_times}")
    lines.append(f"- lambda_proxy: {args.lambda_proxy}")
    lines.append(f"- tail_frac: {args.tail_frac}")
    lines.append(f"- rel_tol: {args.rel_tol}")
    lines.append(f"- sustain_points: {args.sustain_points}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if summary.empty:
        lines.append("(no rows)")
    else:
        lines.append("```csv")
        lines.append(summary.to_csv(index=False).strip())
        lines.append("```")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `{out_dir / 'summary.csv'}`")
    lines.append(f"- `{out_dir / 'tau_eq_probe_vs_lambda.png'}`")
    lines.append(f"- `{out_dir / 'timeseries_samples.json'}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    topologies = _parse_topologies(args.topologies)
    lambdas_by_topo = _parse_lambdas_by_topology(args.lambdas, topologies)
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        REPO_ROOT / "outputs" / f"critical_slowing_scan_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    times = np.linspace(0.0, float(args.t_max), int(args.n_times))
    rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []

    for topo_name in topologies:
        topo = get_topology(topo_name, int(args.N))
        lam_list = lambdas_by_topo[topo_name]
        print(f"Topology={topo_name} lambdas={lam_list}", flush=True)
        for lam in lam_list:
            qca_nom = _build_qca(
                n=int(args.N),
                edges=topo.edges,
                bond_cutoff=int(args.bond_cutoff),
                lambda_value=float(lam),
                delta_b=float(args.deltaB),
                kappa=float(args.kappa),
                k0=float(args.k0),
            )
            qca_hot = _build_qca(
                n=int(args.N),
                edges=topo.edges,
                bond_cutoff=int(args.bond_cutoff),
                lambda_value=float(lam) * float(args.hotspot_multiplier),
                delta_b=float(args.deltaB),
                kappa=float(args.kappa),
                k0=float(args.k0),
            )

            psi0 = qca_nom.get_ground_state()
            lam_site_t = _lambda_timeseries(
                qca_hot=qca_hot,
                psi0=psi0,
                times=times,
                proxy=args.lambda_proxy,
            )
            global_lam = np.mean(lam_site_t, axis=0)
            probe_out, probe_in = topo.probes
            probe_gap = lam_site_t[probe_in, :] - lam_site_t[probe_out, :]

            tau_g, tail_g, std_g, thr_g = _equilibration_time(
                times=times,
                values=global_lam,
                tail_frac=float(args.tail_frac),
                rel_tol=float(args.rel_tol),
                sustain=int(args.sustain_points),
            )
            tau_p, tail_p, std_p, thr_p = _equilibration_time(
                times=times,
                values=probe_gap,
                tail_frac=float(args.tail_frac),
                rel_tol=float(args.rel_tol),
                sustain=int(args.sustain_points),
            )
            tau_dephase_g = _dephasing_time(
                times=times,
                values=global_lam,
                tail_mean=tail_g,
                rel_tol=float(args.rel_tol),
                sustain=int(args.sustain_points),
            )
            tau_dephase_p = _dephasing_time(
                times=times,
                values=probe_gap,
                tail_mean=tail_p,
                rel_tol=float(args.rel_tol),
                sustain=int(args.sustain_points),
            )
            rows.append(
                {
                    "topology": topo_name,
                    "lambda": float(lam),
                    "tau_eq_global": float(tau_g),
                    "tau_eq_probe": float(tau_p),
                    "tau_dephase_global": float(tau_dephase_g),
                    "tau_dephase_probe": float(tau_dephase_p),
                    "tail_mean_global": float(tail_g),
                    "tail_std_global": float(std_g),
                    "threshold_global": float(thr_g),
                    "tail_mean_probe": float(tail_p),
                    "tail_std_probe": float(std_p),
                    "threshold_probe": float(thr_p),
                    "final_global_lambda": float(global_lam[-1]),
                    "final_probe_gap": float(probe_gap[-1]),
                    "max_abs_probe_gap": float(np.max(np.abs(probe_gap))),
                    "dim_nominal": int(qca_nom.total_dim),
                    "dim_hotspot": int(qca_hot.total_dim),
                }
            )
            print(
                f"  lambda={lam:.3f} tau_eq_probe={tau_p:.3f} tau_eq_global={tau_g:.3f} "
                f"tau_dephase_probe={tau_dephase_p:.3f} "
                f"tail_std_probe={std_p:.3e}",
                flush=True,
            )

            if len(sample_rows) < int(args.max_sample_plots):
                sample_rows.append(
                    {
                        "topology": topo_name,
                        "lambda": float(lam),
                        "times": times.tolist(),
                        "probe_gap": probe_gap.tolist(),
                        "global_lambda": global_lam.tolist(),
                        "tau_eq_probe": float(tau_p),
                        "tail_mean_probe": float(tail_p),
                        "threshold_probe": float(thr_p),
                    }
                )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["topology", "lambda"]).reset_index(drop=True)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    _plot_tau(summary, out_dir)
    _plot_examples(sample_rows, out_dir)
    (out_dir / "timeseries_samples.json").write_text(
        json.dumps(sample_rows, indent=2),
        encoding="utf-8",
    )
    _write_report(out_dir=out_dir, args=args, summary=summary)

    print(f"\nOUTPUT_DIR={out_dir}")
    print(f"SUMMARY={summary_path}")
    print(f"REPORT={out_dir / 'report.md'}")
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep-time critical-slowing scan")
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--topologies", type=str, default="path,cycle,star")
    p.add_argument(
        "--lambdas",
        type=str,
        default="path:0.6,1.0,1.2;cycle:0.3,0.5,0.7;star:1.0,1.3,1.5",
        help="Either shared comma list or per-topology map",
    )
    p.add_argument("--bond-cutoff", type=int, default=4)
    p.add_argument("--hotspot-multiplier", type=float, default=1.5)
    p.add_argument("--deltaB", type=float, default=5.0)
    p.add_argument("--kappa", type=float, default=0.1)
    p.add_argument("--k0", type=float, default=4.0)
    p.add_argument("--t-max", type=float, default=40.0)
    p.add_argument("--n-times", type=int, default=120)
    p.add_argument("--lambda-proxy", type=str, default="log", choices=["log", "linear"])
    p.add_argument("--tail-frac", type=float, default=0.2)
    p.add_argument("--rel-tol", type=float, default=0.08)
    p.add_argument("--sustain-points", type=int, default=8)
    p.add_argument("--max-sample-plots", type=int, default=6)
    p.add_argument("--output-dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
