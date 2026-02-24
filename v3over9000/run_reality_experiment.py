#!/usr/bin/env python3
"""
v3over9000 reality check:
- TT-projected spin-2 spectrum
- self-energy alpha extraction
- baseline vs correlated-promotion Hamiltonian
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_qca import ExactQCA
from topologies import get_topology
from v3over9000.alpha_self_energy import estimate_alpha_from_self_energy
from v3over9000.correlated_qca import CorrelatedPromotionQCA
from v3over9000.tensor_spin2 import TensorSpin2Analyzer


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one numeric value.")
    return vals


def _make_config(args: argparse.Namespace, lam: float) -> Dict[str, float]:
    return {
        "omega": float(args.omega),
        "deltaB": float(args.deltaB),
        "lambda": float(lam),
        "kappa": float(args.kappa),
        "k0": float(args.k0),
        "J0": float(args.J0),
        "alpha": float(args.alpha),
        "gamma_corr": float(args.gamma_corr),
        "gamma_corr_diag": float(args.gamma_corr_diag),
    }


def _build_qca(
    model: str,
    args: argparse.Namespace,
    edges: List[Tuple[int, int]],
    lam: float,
):
    config = _make_config(args, lam=lam)
    if model == "baseline":
        return ExactQCA(
            N=int(args.N),
            config=config,
            bond_cutoff=int(args.bond_cutoff),
            edges=edges,
        )
    if model == "correlated":
        return CorrelatedPromotionQCA(
            N=int(args.N),
            config=config,
            bond_cutoff=int(args.bond_cutoff),
            edges=edges,
        )
    raise ValueError(f"Unknown model '{model}'")


def _plot_spectra(spectra_rows: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    for model, subset in spectra_rows.groupby("model"):
        subset = subset.sort_values("k")
        plt.loglog(subset["k"], subset["power"], marker="o", linewidth=1.8, label=model)

    if not spectra_rows.empty:
        k_ref = np.sort(np.unique(spectra_rows["k"].to_numpy()))
        k_ref = k_ref[k_ref > 0]
        if k_ref.size > 0:
            scale = float(np.median(spectra_rows["power"]))
            ref = scale * (k_ref[0] / k_ref) ** 2
            plt.loglog(k_ref, ref, "--", color="black", linewidth=1.5, label="1/k^2 ref")

    plt.xlabel("k")
    plt.ylabel("TT power")
    plt.title("v3over9000 TT Spin-2 Spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def run(args: argparse.Namespace) -> Path:
    topology = get_topology(args.topology, args.N)
    edges = topology.edges
    probe_site = int(args.probe_site if args.probe_site >= 0 else topology.probes[0])

    out_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        REPO_ROOT / "outputs" / f"v3over9000_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor = TensorSpin2Analyzer()
    lambda_scan = _parse_float_list(args.alpha_scan_lambdas)
    summary_rows: List[Dict[str, float | str | int]] = []
    spectra_rows: List[Dict[str, float | str]] = []
    alpha_json: Dict[str, Dict[str, object]] = {}

    for model in ("baseline", "correlated"):
        qca_nominal = _build_qca(model=model, args=args, edges=edges, lam=float(args.lambda_value))
        qca_hot = _build_qca(
            model=model,
            args=args,
            edges=edges,
            lam=float(args.lambda_value * args.hotspot_multiplier),
        )

        ground = qca_nominal.get_ground_state()
        state = qca_hot.evolve_state(ground, float(args.frustration_time))

        spec = tensor.compute_spectrum(
            qca=qca_nominal,
            state=state,
            topology_name=topology.name,
            n_modes=int(args.k_modes),
            n_angles=int(args.k_angles),
        )
        fit = tensor.fit_power_law(k=spec.k, power=spec.power, expected_power=2.0)

        for k_val, p_val in zip(spec.k, spec.power):
            spectra_rows.append({"model": model, "k": float(k_val), "power": float(p_val)})

        alpha_result = estimate_alpha_from_self_energy(
            qca_builder=lambda lam: _build_qca(model=model, args=args, edges=edges, lam=float(lam)),
            lambdas=lambda_scan,
            probe_site=probe_site,
            t_max=float(args.ramsey_tmax),
            n_points=int(args.ramsey_points),
            state_builder=lambda qca_obj, lam: _build_qca(
                model=model,
                args=args,
                edges=edges,
                lam=float(lam * args.hotspot_multiplier),
            ).evolve_state(qca_obj.get_ground_state(), float(args.frustration_time)),
        )
        alpha_json[model] = alpha_result

        summary_rows.append(
            {
                "model": model,
                "N": int(args.N),
                "topology": topology.name,
                "bond_cutoff": int(args.bond_cutoff),
                "lambda_value": float(args.lambda_value),
                "hotspot_multiplier": float(args.hotspot_multiplier),
                "gamma_corr": float(args.gamma_corr) if model == "correlated" else 0.0,
                "gamma_corr_diag": float(args.gamma_corr_diag) if model == "correlated" else 0.0,
                "spin2_measured_power": float(fit["measured_power"]),
                "spin2_measured_slope": float(fit["measured_slope"]),
                "spin2_residual_power": float(fit["residual"]),
                "spin2_r2": float(fit["r2"]),
                "spin2_fit_quality": bool(fit["fit_quality"]),
                "alpha_self_energy": float(alpha_result["alpha_self_energy"]),
                "alpha_fit_r2": float(alpha_result["r2"]),
                "alpha_fit_points": int(alpha_result.get("fit_points", 0)),
            }
        )

    spectra_df = pd.DataFrame(spectra_rows)
    summary_df = pd.DataFrame(summary_rows)
    spectra_csv = out_dir / "tt_spin2_spectra.csv"
    summary_csv = out_dir / "v3over9000_summary.csv"
    alpha_json_path = out_dir / "alpha_self_energy.json"
    plot_path = out_dir / "tt_spin2_spectrum.png"
    report_path = out_dir / "report.md"

    spectra_df.to_csv(spectra_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    alpha_json_path.write_text(json.dumps(alpha_json, indent=2), encoding="utf-8")
    _plot_spectra(spectra_df, plot_path)

    summary_table = summary_df.to_csv(index=False).strip()

    report_lines = [
        "# v3over9000 Reality Check",
        "",
        "## Configuration",
        f"- N: {args.N}",
        f"- topology: {topology.name}",
        f"- alpha(input): {args.alpha}",
        f"- lambda_value: {args.lambda_value}",
        f"- bond_cutoff: {args.bond_cutoff}",
        f"- hotspot_multiplier: {args.hotspot_multiplier}",
        f"- frustration_time: {args.frustration_time}",
        f"- gamma_corr: {args.gamma_corr}",
        f"- gamma_corr_diag: {args.gamma_corr_diag}",
        "",
        "## Summary (baseline vs correlated)",
        "",
        "```csv",
        summary_table,
        "```",
        "",
        "## Artifacts",
        f"- `{spectra_csv}`",
        f"- `{summary_csv}`",
        f"- `{alpha_json_path}`",
        f"- `{plot_path}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"OUTPUT_DIR={out_dir}")
    print(f"SUMMARY={summary_csv}")
    print(f"SPECTRA={spectra_csv}")
    print(f"ALPHA_JSON={alpha_json_path}")
    print(f"PLOT={plot_path}")
    print(f"REPORT={report_path}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3over9000 TT + alpha experiment")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--topology", type=str, default="path")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--lambda-value", type=float, default=0.6)
    parser.add_argument("--bond-cutoff", type=int, default=4)
    parser.add_argument("--hotspot-multiplier", type=float, default=1.5)
    parser.add_argument("--frustration-time", type=float, default=1.0)
    parser.add_argument("--probe-site", type=int, default=-1, help="Default: topology probe_out")

    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--J0", type=float, default=0.01)
    parser.add_argument("--deltaB", type=float, default=5.0)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--k0", type=float, default=4.0)
    parser.add_argument("--gamma-corr", type=float, default=0.05)
    parser.add_argument("--gamma-corr-diag", type=float, default=0.0)

    parser.add_argument("--k-modes", type=int, default=16)
    parser.add_argument("--k-angles", type=int, default=24)

    parser.add_argument("--alpha-scan-lambdas", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--ramsey-tmax", type=float, default=16.0)
    parser.add_argument("--ramsey-points", type=int, default=96)

    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
