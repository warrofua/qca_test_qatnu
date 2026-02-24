#!/usr/bin/env python3
"""
Frozen-matter static screening scan.

Implements the Appendix-D-style check described in docs:
- freeze matter spins z_i in {+1, -1}
- fix frustration tiles F_ij = (1 - z_i z_j)/2
- solve the bond-sector ground state only
- measure Lambda_i from bond occupations
- fit graph-Poisson (massless) vs screened-Laplacian (Yukawa-like) response

Model fits (per topology, per lambda):
  L Lambda = -kappa * rho
  L Lambda = -kappa * rho + mu^2 * Lambda
where L is the graph Laplacian and rho is a vertex source derived from F_ij.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topologies import get_topology


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one float.")
    return vals


def _parse_topologies(raw: str) -> List[str]:
    vals = [v.strip().lower() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one topology.")
    return vals


def _graph_laplacian(n: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    for u, v in edges:
        a[u, v] = 1.0
        a[v, u] = 1.0
    deg = np.diag(np.sum(a, axis=1))
    return deg - a


def _incident_edges(n: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    out: List[List[int]] = [[] for _ in range(n)]
    for edge_idx, (u, v) in enumerate(edges):
        out[u].append(edge_idx)
        out[v].append(edge_idx)
    return out


@dataclass(frozen=True)
class FrozenPattern:
    pattern_id: str
    z: Tuple[int, ...]
    frustration_by_edge: Tuple[int, ...]


def _enumerate_patterns(
    n: int,
    edges: Sequence[Tuple[int, int]],
    include_uniform: bool,
) -> List[FrozenPattern]:
    """
    Enumerate unique frustration patterns realizable from static z_i assignments.
    Global spin-flip gauge is removed by fixing z_0 = +1.
    """
    seen: set[Tuple[int, ...]] = set()
    patterns: List[FrozenPattern] = []
    for mask in range(1 << max(n - 1, 0)):
        z = [1]
        for i in range(1, n):
            bit = (mask >> (i - 1)) & 1
            z.append(1 if bit else -1)
        fr = tuple(int((1 - z[u] * z[v]) // 2) for (u, v) in edges)
        if not include_uniform and all(v == 0 for v in fr):
            continue
        if fr in seen:
            continue
        seen.add(fr)
        patterns.append(
            FrozenPattern(
                pattern_id=f"p{len(patterns):02d}",
                z=tuple(z),
                frustration_by_edge=fr,
            )
        )
    return patterns


@dataclass(frozen=True)
class BasisCache:
    configs: np.ndarray
    powers: np.ndarray
    step_up_indices: Tuple[np.ndarray, ...]
    step_dn_indices: Tuple[np.ndarray, ...]
    incident: List[List[int]]
    diag_base: np.ndarray


def _build_basis_cache(
    n: int,
    edges: Sequence[Tuple[int, int]],
    bond_cutoff: int,
    delta_b: float,
    kappa: float,
    k0: float,
) -> BasisCache:
    e_count = len(edges)
    if e_count == 0:
        configs = np.zeros((1, 0), dtype=np.int16)
        powers = np.zeros(0, dtype=np.int64)
    else:
        shape = (bond_cutoff,) * e_count
        configs = np.array(np.unravel_index(np.arange(bond_cutoff**e_count), shape)).T.astype(np.int16)
        powers = (bond_cutoff ** np.arange(e_count - 1, -1, -1)).astype(np.int64)

    incident = _incident_edges(n, edges)
    dim = int(configs.shape[0])
    diag = np.zeros(dim, dtype=float)
    if e_count > 0:
        diag += float(delta_b) * np.sum(configs, axis=1)

    if n > 0:
        degrees = np.zeros((dim, n), dtype=np.int16)
        for site, inc in enumerate(incident):
            if inc:
                degrees[:, site] = np.sum(configs[:, inc] > 0, axis=1)
        penalty = np.zeros(dim, dtype=float)
        for u, v in edges:
            penalty += (degrees[:, u] - float(k0)) ** 2 + (degrees[:, v] - float(k0)) ** 2
        diag += float(kappa) * penalty

    step_up: List[np.ndarray] = []
    step_dn: List[np.ndarray] = []
    for edge_idx in range(e_count):
        occ = configs[:, edge_idx]
        up = np.nonzero(occ < (bond_cutoff - 1))[0].astype(np.int64)
        dn = np.nonzero(occ > 0)[0].astype(np.int64)
        step_up.append(up)
        step_dn.append(dn)

    return BasisCache(
        configs=configs,
        powers=powers,
        step_up_indices=tuple(step_up),
        step_dn_indices=tuple(step_dn),
        incident=incident,
        diag_base=diag,
    )


def _build_hamiltonian(
    basis: BasisCache,
    edges: Sequence[Tuple[int, int]],
    bond_cutoff: int,
    lambda_value: float,
    frustration_by_edge: Sequence[int],
) -> np.ndarray:
    dim = int(basis.configs.shape[0])
    h = np.zeros((dim, dim), dtype=float)
    np.fill_diagonal(h, basis.diag_base)
    if not edges:
        return h

    fr = np.asarray(frustration_by_edge, dtype=float)
    lam = float(lambda_value)
    for edge_idx in range(len(edges)):
        amp = lam * fr[edge_idx]
        if abs(amp) <= 1e-15:
            continue
        step = int(basis.powers[edge_idx])
        up = basis.step_up_indices[edge_idx]
        dn = basis.step_dn_indices[edge_idx]
        if up.size > 0:
            h[up, up + step] += amp
        if dn.size > 0:
            h[dn, dn - step] += amp

    h = (h + h.T) * 0.5
    return h


def _lambda_from_mean_n(
    mean_n_by_edge: np.ndarray,
    incident: Sequence[Sequence[int]],
    proxy: str,
) -> np.ndarray:
    out = np.zeros(len(incident), dtype=float)
    for site, inc in enumerate(incident):
        if not inc:
            out[site] = 0.0
            continue
        vals = mean_n_by_edge[np.asarray(inc, dtype=int)]
        if proxy == "linear":
            out[site] = float(np.sum(vals))
        elif proxy == "log":
            out[site] = float(np.sum(np.log1p(vals)))
        else:
            raise ValueError(f"Unsupported proxy '{proxy}'")
    return out


def _source_from_frustration(
    frustration_by_edge: Sequence[int],
    incident: Sequence[Sequence[int]],
    center_source: bool,
) -> np.ndarray:
    fr = np.asarray(frustration_by_edge, dtype=float)
    rho = np.zeros(len(incident), dtype=float)
    for site, inc in enumerate(incident):
        if inc:
            rho[site] = float(np.sum(fr[np.asarray(inc, dtype=int)]))
    if center_source:
        rho = rho - float(np.mean(rho))
    return rho


def _fit_massless_and_screened(
    laplacian: np.ndarray,
    lambda_vectors: Sequence[np.ndarray],
    source_vectors: Sequence[np.ndarray],
    constrain_mu_nonnegative: bool,
    min_improvement: float,
) -> Dict[str, float | str]:
    eps = 1e-18
    ys: List[np.ndarray] = []
    xs_rho: List[np.ndarray] = []
    xs_lam: List[np.ndarray] = []
    for lam_vec, rho_vec in zip(lambda_vectors, source_vectors):
        y = laplacian @ lam_vec
        ys.append(y)
        xs_rho.append(-rho_vec)
        xs_lam.append(lam_vec)

    y_stack = np.concatenate(ys)
    rho_stack = np.concatenate(xs_rho)
    lam_stack = np.concatenate(xs_lam)

    denom = float(np.dot(rho_stack, rho_stack))
    if denom <= eps:
        return {
            "kappa_massless": float("nan"),
            "resid_massless": float("nan"),
            "kappa_screened": float("nan"),
            "mu2_screened": float("nan"),
            "resid_screened": float("nan"),
            "rss_massless": float("nan"),
            "rss_screened": float("nan"),
            "improvement_frac": float("nan"),
            "best_model": "undetermined",
        }

    kappa_mass = float(np.dot(rho_stack, y_stack) / denom)
    pred_mass = kappa_mass * rho_stack
    rss_mass = float(np.sum((y_stack - pred_mass) ** 2))
    norm_y = float(np.sum(y_stack**2))
    resid_mass = float(math.sqrt(rss_mass / (norm_y + eps)))

    a = np.column_stack([rho_stack, lam_stack])
    theta, *_ = np.linalg.lstsq(a, y_stack, rcond=None)
    kappa_scr = float(theta[0])
    mu2_scr = float(theta[1])
    pred_scr = a @ np.array([kappa_scr, mu2_scr], dtype=float)
    rss_scr = float(np.sum((y_stack - pred_scr) ** 2))

    if constrain_mu_nonnegative and mu2_scr < 0.0:
        mu2_scr = 0.0
        kappa_scr = kappa_mass
        rss_scr = rss_mass
        pred_scr = pred_mass

    resid_scr = float(math.sqrt(rss_scr / (norm_y + eps)))
    improvement = float((rss_mass - rss_scr) / (rss_mass + eps))
    best = "screened" if (mu2_scr > 0.0 and improvement >= min_improvement) else "massless"
    return {
        "kappa_massless": kappa_mass,
        "resid_massless": resid_mass,
        "kappa_screened": kappa_scr,
        "mu2_screened": mu2_scr,
        "resid_screened": resid_scr,
        "rss_massless": rss_mass,
        "rss_screened": rss_scr,
        "improvement_frac": improvement,
        "best_model": best,
    }


def _plot_mu(fits: pd.DataFrame, out_dir: Path) -> None:
    if fits.empty:
        return
    plt.figure(figsize=(8.5, 5.2))
    for topo, subset in fits.groupby("topology"):
        subset = subset.sort_values("lambda")
        plt.plot(
            subset["lambda"],
            subset["mu2_screened"],
            marker="o",
            linewidth=1.8,
            label=topo,
        )
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel("lambda")
    plt.ylabel("mu^2 (screened fit)")
    plt.title("Frozen-matter screening scan")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mu2_vs_lambda.png", dpi=220, bbox_inches="tight")
    plt.close()


def _plot_residuals(fits: pd.DataFrame, out_dir: Path) -> None:
    if fits.empty:
        return
    plt.figure(figsize=(8.5, 5.2))
    for topo, subset in fits.groupby("topology"):
        subset = subset.sort_values("lambda")
        plt.plot(
            subset["lambda"],
            subset["resid_massless"],
            marker="o",
            linewidth=1.5,
            linestyle="--",
            label=f"{topo} massless",
        )
        plt.plot(
            subset["lambda"],
            subset["resid_screened"],
            marker="s",
            linewidth=1.5,
            linestyle="-",
            label=f"{topo} screened",
        )
    plt.xlabel("lambda")
    plt.ylabel("relative residual")
    plt.title("Massless vs screened operator fit")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fit_residuals_vs_lambda.png", dpi=220, bbox_inches="tight")
    plt.close()


def _write_report(
    out_dir: Path,
    args: argparse.Namespace,
    fit_df: pd.DataFrame,
    case_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Frozen-Matter Screening Scan")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- N: {args.N}")
    lines.append(f"- topologies: {args.topologies}")
    lines.append(f"- lambdas: {args.lambdas}")
    lines.append(f"- bond_cutoff: {args.bond_cutoff}")
    lines.append(f"- deltaB: {args.deltaB}")
    lines.append(f"- kappa: {args.kappa}")
    lines.append(f"- k0: {args.k0}")
    lines.append(f"- lambda_proxy: {args.lambda_proxy}")
    lines.append(f"- center_source: {bool(args.center_source)}")
    lines.append(f"- constrain_mu_nonnegative: {bool(args.constrain_mu_nonnegative)}")
    lines.append(f"- min_improvement: {args.min_improvement}")
    lines.append("")
    lines.append("## Aggregate Fits")
    lines.append("")
    if fit_df.empty:
        lines.append("(no fit rows)")
    else:
        lines.append("```csv")
        lines.append(fit_df.to_csv(index=False).strip())
        lines.append("```")
    lines.append("")
    lines.append("## Case Count")
    lines.append("")
    lines.append(f"- total cases: {len(case_df)}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `{out_dir / 'frozen_cases.csv'}`")
    lines.append(f"- `{out_dir / 'frozen_fits.csv'}`")
    lines.append(f"- `{out_dir / 'mu2_vs_lambda.png'}`")
    lines.append(f"- `{out_dir / 'fit_residuals_vs_lambda.png'}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    topologies = _parse_topologies(args.topologies)
    lambdas = _parse_float_list(args.lambdas)
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        REPO_ROOT / "outputs" / f"frozen_screening_scan_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_case_rows: List[Dict[str, object]] = []
    all_fit_rows: List[Dict[str, object]] = []

    for topo_name in topologies:
        topo = get_topology(topo_name, int(args.N))
        edges = topo.edges
        n_edges = len(edges)
        if n_edges == 0:
            print(f"Skipping {topo_name}: no edges.")
            continue

        patterns = _enumerate_patterns(
            n=int(args.N),
            edges=edges,
            include_uniform=bool(args.include_uniform),
        )
        if not patterns:
            print(f"Skipping {topo_name}: no usable patterns.")
            continue

        lap = _graph_laplacian(int(args.N), edges)
        basis = _build_basis_cache(
            n=int(args.N),
            edges=edges,
            bond_cutoff=int(args.bond_cutoff),
            delta_b=float(args.deltaB),
            kappa=float(args.kappa),
            k0=float(args.k0),
        )

        print(
            f"Topology={topo_name} edges={n_edges} dim={basis.configs.shape[0]} patterns={len(patterns)}",
            flush=True,
        )
        for lam in lambdas:
            lambda_vectors: List[np.ndarray] = []
            source_vectors: List[np.ndarray] = []
            for pat in patterns:
                h = _build_hamiltonian(
                    basis=basis,
                    edges=edges,
                    bond_cutoff=int(args.bond_cutoff),
                    lambda_value=float(lam),
                    frustration_by_edge=pat.frustration_by_edge,
                )
                eigvals, eigvecs = np.linalg.eigh(h)
                gs = eigvecs[:, 0]
                prob = np.abs(gs) ** 2
                mean_n = prob @ basis.configs
                lam_vec = _lambda_from_mean_n(
                    mean_n_by_edge=mean_n,
                    incident=basis.incident,
                    proxy=args.lambda_proxy,
                )
                rho_vec = _source_from_frustration(
                    frustration_by_edge=pat.frustration_by_edge,
                    incident=basis.incident,
                    center_source=bool(args.center_source),
                )

                lambda_vectors.append(lam_vec)
                source_vectors.append(rho_vec)

                all_case_rows.append(
                    {
                        "topology": topo_name,
                        "lambda": float(lam),
                        "pattern_id": pat.pattern_id,
                        "z_pattern": json.dumps(list(pat.z)),
                        "frustration_by_edge": json.dumps(list(pat.frustration_by_edge)),
                        "rho_vector": json.dumps([float(x) for x in rho_vec]),
                        "lambda_vector": json.dumps([float(x) for x in lam_vec]),
                        "lambda_mean": float(np.mean(lam_vec)),
                        "lambda_max": float(np.max(lam_vec)),
                        "lambda_min": float(np.min(lam_vec)),
                        "ground_energy": float(eigvals[0]),
                    }
                )

            fit = _fit_massless_and_screened(
                laplacian=lap,
                lambda_vectors=lambda_vectors,
                source_vectors=source_vectors,
                constrain_mu_nonnegative=bool(args.constrain_mu_nonnegative),
                min_improvement=float(args.min_improvement),
            )
            fit_row: Dict[str, object] = {
                "topology": topo_name,
                "lambda": float(lam),
                "n_patterns": int(len(patterns)),
                "dim_bond": int(basis.configs.shape[0]),
            }
            fit_row.update(fit)
            all_fit_rows.append(fit_row)
            print(
                f"  lambda={lam:.3f} best={fit_row['best_model']} "
                f"mu2={fit_row['mu2_screened']:.6g} "
                f"res_mass={fit_row['resid_massless']:.4g} res_scr={fit_row['resid_screened']:.4g}",
                flush=True,
            )

    case_df = pd.DataFrame(all_case_rows)
    fit_df = pd.DataFrame(all_fit_rows)
    if not fit_df.empty:
        fit_df = fit_df.sort_values(["topology", "lambda"]).reset_index(drop=True)
    if not case_df.empty:
        case_df = case_df.sort_values(["topology", "lambda", "pattern_id"]).reset_index(drop=True)

    cases_path = out_dir / "frozen_cases.csv"
    fits_path = out_dir / "frozen_fits.csv"
    case_df.to_csv(cases_path, index=False)
    fit_df.to_csv(fits_path, index=False)

    _plot_mu(fit_df, out_dir)
    _plot_residuals(fit_df, out_dir)
    _write_report(out_dir=out_dir, args=args, fit_df=fit_df, case_df=case_df)

    print(f"\nOUTPUT_DIR={out_dir}")
    print(f"CASES={cases_path}")
    print(f"FITS={fits_path}")
    print(f"REPORT={out_dir / 'report.md'}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-matter static screening scan")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--topologies", type=str, default="path,cycle,star")
    parser.add_argument("--lambdas", type=str, default="0.1,0.2,0.4,0.6,0.8,1.0")

    parser.add_argument("--bond-cutoff", type=int, default=4)
    parser.add_argument("--deltaB", type=float, default=5.0)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--k0", type=float, default=4.0)

    parser.add_argument(
        "--lambda-proxy",
        type=str,
        default="log",
        choices=["log", "linear"],
        help="How to map mean bond occupations to Lambda_i.",
    )
    parser.add_argument(
        "--include-uniform",
        action="store_true",
        help="Include the all-unfrustrated source pattern.",
    )
    parser.add_argument(
        "--center-source",
        action="store_true",
        default=True,
        help="Center source rho_i -> rho_i - mean(rho) before fitting.",
    )
    parser.add_argument(
        "--no-center-source",
        action="store_false",
        dest="center_source",
    )
    parser.add_argument(
        "--constrain-mu-nonnegative",
        action="store_true",
        default=True,
        help="Clamp screened fit to mu^2 >= 0.",
    )
    parser.add_argument(
        "--allow-negative-mu2",
        action="store_false",
        dest="constrain_mu_nonnegative",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="Minimum fractional RSS gain required to call screened model better.",
    )
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
