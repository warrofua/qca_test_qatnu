#!/usr/bin/env python3
"""
Sweep topology x gamma_corr x lambda for v3over9000 TT spin-2 diagnostics.

Outputs:
- per-point CSV with TT and Postulate-1 residual metrics
- summary CSV ranked by spin-2 residual while tracking clock-law degradation
- markdown report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_qca import ExactQCA
from topologies import get_topology
from v3over9000.correlated_qca import CorrelatedPromotionQCA
from v3over9000.tensor_spin2 import TensorSpin2Analyzer


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def _parse_topologies(raw: str) -> List[str]:
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one topology.")
    return vals


def _float_key(value: float) -> str:
    return f"{float(value):.12g}"


def _payload_point_key(payload: Dict[str, object]) -> Tuple[str, str, str]:
    return (
        str(payload["topology"]).lower(),
        _float_key(float(payload["gamma_corr"])),
        _float_key(float(payload["lambda"])),
    )


def _row_point_key(row: Dict[str, object]) -> Tuple[str, str, str]:
    return (
        str(row["topology"]).lower(),
        _float_key(float(row["gamma_corr"])),
        _float_key(float(row["lambda"])),
    )


def _write_checkpoints(
    points_path: Path,
    rows: List[Dict[str, object]],
    error_path: Path,
    err_rows: List[Dict[str, object]],
) -> None:
    pd.DataFrame(rows).to_csv(points_path, index=False)
    if err_rows:
        pd.DataFrame(err_rows).to_csv(error_path, index=False)
    elif error_path.exists():
        error_path.unlink()


def _compute_dedupe_key(payload: Dict[str, object]) -> Tuple[str, ...]:
    gamma_eff = _effective_gamma(
        gamma_base=float(payload["gamma_corr"]),
        lambda_value=float(payload["lambda"]),
        mode=str(payload["gamma_mode"]),
        lambda_low=float(payload["gamma_lambda_low"]),
        lambda_high=float(payload["gamma_lambda_high"]),
        taper_power=float(payload["gamma_taper_power"]),
    )
    return (
        str(payload["N"]),
        str(payload["topology"]).lower(),
        _float_key(float(payload["alpha"])),
        _float_key(float(payload["lambda"])),
        _float_key(float(gamma_eff)),
        _float_key(float(payload["bond_cutoff"])),
        _float_key(float(payload["omega"])),
        _float_key(float(payload["J0"])),
        _float_key(float(payload["deltaB"])),
        _float_key(float(payload["kappa"])),
        _float_key(float(payload["k0"])),
        _float_key(float(payload["hotspot_multiplier"])),
        _float_key(float(payload["frustration_time"])),
        _float_key(float(payload["k_modes"])),
        _float_key(float(payload["k_angles"])),
        _float_key(float(payload["ramsey_tmax"])),
        _float_key(float(payload["ramsey_points"])),
    )


def _effective_gamma(
    gamma_base: float,
    lambda_value: float,
    mode: str,
    lambda_low: float,
    lambda_high: float,
    taper_power: float,
) -> float:
    mode = (mode or "constant").lower()
    if mode == "constant":
        return float(gamma_base)

    if mode == "taper_high":
        if lambda_value <= lambda_low:
            return float(gamma_base)
        if lambda_high <= lambda_low:
            return 0.0
        if lambda_value >= lambda_high:
            return 0.0
        frac = (lambda_high - lambda_value) / (lambda_high - lambda_low)
        frac = max(0.0, min(1.0, float(frac)))
        power = max(float(taper_power), 1e-6)
        return float(gamma_base) * (frac**power)

    raise ValueError(f"Unsupported gamma mode '{mode}'")


def _fit_frequency_fft(times: np.ndarray, signal: np.ndarray) -> float:
    if len(times) < 10:
        return 0.0
    dt = float(times[1] - times[0])
    if dt <= 0.0:
        return 0.0
    centered = signal - np.mean(signal)
    fft = np.fft.fft(centered)
    freqs = np.fft.fftfreq(len(times), d=dt)
    mask = freqs > 0
    if not np.any(mask):
        return 0.0

    amps = np.abs(fft[mask])
    pos_freqs = freqs[mask]
    idx = int(np.argmax(amps))
    base_freq = float(pos_freqs[idx])
    if 0 < idx < len(amps) - 1:
        a = float(amps[idx - 1])
        b = float(amps[idx])
        c = float(amps[idx + 1])
        denom = (a - 2.0 * b + c)
        if abs(denom) > 1e-14:
            delta = 0.5 * (a - c) / denom
            step = float(pos_freqs[1] - pos_freqs[0]) if len(pos_freqs) > 1 else 0.0
            base_freq = base_freq + delta * step

    return float(2.0 * np.pi * base_freq)


def _measure_probe_frequency(qca, base_state: np.ndarray, site: int, t_max: float, n_points: int) -> float:
    times = np.linspace(0.0, float(t_max), int(max(n_points, 24)))
    psi = qca.apply_pi2_pulse(base_state, int(site))
    eig = qca.diagonalize()
    eigenvectors = eig["eigenvectors"]
    eigenvalues = eig["eigenvalues"]

    coeffs0 = eigenvectors.T @ psi
    phases = np.exp(-1j * np.outer(eigenvalues, times))
    evolved = eigenvectors @ (coeffs0[:, None] * phases)

    if hasattr(qca, "Z_lookup") and hasattr(qca, "bond_dim"):
        z_full = np.repeat(qca.Z_lookup[:, int(site)], int(qca.bond_dim)).astype(np.float64)
        z_vals = (z_full @ (np.abs(evolved) ** 2)).astype(np.float64)
        return _fit_frequency_fft(times, z_vals)

    # Fallback for unexpected QCA implementations.
    z_vals = np.zeros_like(times, dtype=float)
    for i in range(len(times)):
        z_vals[i] = float(qca.measure_Z(evolved[:, i], int(site)))
    return _fit_frequency_fft(times, z_vals)


def _build_qca(
    N: int,
    edges: List[Tuple[int, int]],
    bond_cutoff: int,
    omega: float,
    J0: float,
    alpha: float,
    deltaB: float,
    kappa: float,
    k0: float,
    lambda_value: float,
    gamma_corr: float,
):
    config = {
        "omega": float(omega),
        "deltaB": float(deltaB),
        "lambda": float(lambda_value),
        "kappa": float(kappa),
        "k0": float(k0),
        "J0": float(J0),
        "alpha": float(alpha),
        "gamma_corr": float(gamma_corr),
        "gamma_corr_diag": 0.0,
    }
    if abs(float(gamma_corr)) <= 1e-15:
        return ExactQCA(N=N, config=config, bond_cutoff=int(bond_cutoff), edges=edges)
    return CorrelatedPromotionQCA(N=N, config=config, bond_cutoff=int(bond_cutoff), edges=edges)


def _compute_one(payload: Dict[str, object]) -> Dict[str, object]:
    N = int(payload["N"])
    topology = str(payload["topology"])
    alpha = float(payload["alpha"])
    lam = float(payload["lambda"])
    gamma_corr = float(payload["gamma_corr"])
    gamma_mode = str(payload["gamma_mode"])
    gamma_lambda_low = float(payload["gamma_lambda_low"])
    gamma_lambda_high = float(payload["gamma_lambda_high"])
    gamma_taper_power = float(payload["gamma_taper_power"])
    bond_cutoff = int(payload["bond_cutoff"])
    omega = float(payload["omega"])
    J0 = float(payload["J0"])
    deltaB = float(payload["deltaB"])
    kappa = float(payload["kappa"])
    k0 = float(payload["k0"])
    hotspot_multiplier = float(payload["hotspot_multiplier"])
    frustration_time = float(payload["frustration_time"])
    k_modes = int(payload["k_modes"])
    k_angles = int(payload["k_angles"])
    ramsey_tmax = float(payload["ramsey_tmax"])
    ramsey_points = int(payload["ramsey_points"])

    topo = get_topology(topology, N)
    edges = topo.edges
    probe_out, probe_in = topo.probes

    gamma_eff = _effective_gamma(
        gamma_base=gamma_corr,
        lambda_value=lam,
        mode=gamma_mode,
        lambda_low=gamma_lambda_low,
        lambda_high=gamma_lambda_high,
        taper_power=gamma_taper_power,
    )

    qca_nominal = _build_qca(
        N=N,
        edges=edges,
        bond_cutoff=bond_cutoff,
        omega=omega,
        J0=J0,
        alpha=alpha,
        deltaB=deltaB,
        kappa=kappa,
        k0=k0,
        lambda_value=lam,
        gamma_corr=gamma_eff,
    )
    qca_hot = _build_qca(
        N=N,
        edges=edges,
        bond_cutoff=bond_cutoff,
        omega=omega,
        J0=J0,
        alpha=alpha,
        deltaB=deltaB,
        kappa=kappa,
        k0=k0,
        lambda_value=lam * hotspot_multiplier,
        gamma_corr=gamma_eff,
    )

    ground = qca_nominal.get_ground_state()
    frustrated = qca_hot.evolve_state(ground, frustration_time)

    tensor = TensorSpin2Analyzer()
    spectrum = tensor.compute_spectrum(
        qca=qca_nominal,
        state=frustrated,
        topology_name=topology,
        n_modes=k_modes,
        n_angles=k_angles,
    )
    spin2_fit = tensor.fit_power_law(k=spectrum.k, power=spectrum.power, expected_power=2.0)

    lambda_out = float(qca_nominal.count_circuit_depth(probe_out, frustrated))
    lambda_in = float(qca_nominal.count_circuit_depth(probe_in, frustrated))
    pred_out = float(omega / (1.0 + alpha * lambda_out))
    pred_in = float(omega / (1.0 + alpha * lambda_in))
    omega_out = _measure_probe_frequency(
        qca=qca_nominal,
        base_state=frustrated,
        site=probe_out,
        t_max=ramsey_tmax,
        n_points=ramsey_points,
    )
    omega_in = _measure_probe_frequency(
        qca=qca_nominal,
        base_state=frustrated,
        site=probe_in,
        t_max=ramsey_tmax,
        n_points=ramsey_points,
    )
    if abs(omega_out) > 1e-12 and abs(pred_out) > 1e-12:
        measured_ratio = omega_in / omega_out
        predicted_ratio = pred_in / pred_out
        postulate_residual = float(abs(measured_ratio - predicted_ratio))
    else:
        postulate_residual = float("inf")

    return {
        "N": N,
        "topology": topology,
        "alpha": alpha,
        "lambda": lam,
        "gamma_corr": gamma_corr,
        "gamma_corr_effective": float(gamma_eff),
        "gamma_mode": gamma_mode,
        "bond_cutoff": bond_cutoff,
        "hotspot_multiplier": hotspot_multiplier,
        "spin2_measured_power": float(spin2_fit["measured_power"]),
        "spin2_measured_slope": float(spin2_fit["measured_slope"]),
        "spin2_residual_power": float(spin2_fit["residual"]),
        "spin2_r2": float(spin2_fit["r2"]),
        "spin2_fit_quality": bool(spin2_fit["fit_quality"]),
        "postulate_residual": float(postulate_residual),
        "lambda_out": float(lambda_out),
        "lambda_in": float(lambda_in),
        "omega_out": float(omega_out),
        "omega_in": float(omega_in),
        "pred_out": float(pred_out),
        "pred_in": float(pred_in),
        "num_k_modes": int(spectrum.n_modes),
        "num_k_angles": int(spectrum.n_angles),
    }


def _plot_best_spin2(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(9, 6))
    for topo, subset in summary.groupby("topology"):
        subset = subset.sort_values("gamma_corr")
        plt.plot(
            subset["gamma_corr"],
            subset["best_spin2_residual"],
            marker="o",
            linewidth=1.8,
            label=topo,
        )
    plt.xlabel("gamma_corr")
    plt.ylabel("best |power-2|")
    plt.title("v3over9000: Spin-2 residual vs correlated promotion strength")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def run(args: argparse.Namespace) -> Path:
    topologies = _parse_topologies(args.topologies)
    lambdas = _parse_float_list(args.lambdas)
    gammas = _parse_float_list(args.gammas)
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        REPO_ROOT / "outputs" / f"v3over9000_gamma_sweep_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    points_path = out_dir / "points.csv"
    summary_path = out_dir / "summary.csv"
    error_path = out_dir / "errors.csv"
    plot_path = out_dir / "spin2_residual_vs_gamma.png"
    report_path = out_dir / "report.md"

    payloads: List[Dict[str, object]] = []
    for topology in topologies:
        # Validate once up front.
        get_topology(topology, int(args.N))
        for gamma in gammas:
            for lam in lambdas:
                payloads.append(
                    {
                        "N": int(args.N),
                        "topology": topology,
                        "alpha": float(args.alpha),
                        "lambda": float(lam),
                        "gamma_corr": float(gamma),
                        "gamma_mode": str(args.gamma_mode),
                        "gamma_lambda_low": float(args.gamma_lambda_low),
                        "gamma_lambda_high": float(args.gamma_lambda_high),
                        "gamma_taper_power": float(args.gamma_taper_power),
                        "bond_cutoff": int(args.bond_cutoff),
                        "omega": float(args.omega),
                        "J0": float(args.J0),
                        "deltaB": float(args.deltaB),
                        "kappa": float(args.kappa),
                        "k0": float(args.k0),
                        "hotspot_multiplier": float(args.hotspot_multiplier),
                        "frustration_time": float(args.frustration_time),
                        "k_modes": int(args.k_modes),
                        "k_angles": int(args.k_angles),
                        "ramsey_tmax": float(args.ramsey_tmax),
                        "ramsey_points": int(args.ramsey_points),
                    }
                )

    rows: List[Dict[str, object]] = []
    err_rows: List[Dict[str, object]] = []
    completed_keys: Set[Tuple[str, str, str]] = set()
    if bool(args.resume):
        if points_path.exists():
            existing_points = pd.read_csv(points_path)
            if not existing_points.empty:
                rows = existing_points.to_dict(orient="records")
                completed_keys = {_row_point_key(row) for row in rows}
        if error_path.exists():
            existing_errors = pd.read_csv(error_path)
            if not existing_errors.empty:
                err_rows = existing_errors.to_dict(orient="records")

    n_payloads_total = len(payloads)
    if completed_keys:
        payloads = [payload for payload in payloads if _payload_point_key(payload) not in completed_keys]
        skipped = n_payloads_total - len(payloads)
        if skipped > 0:
            print(f"Resume: skipping {skipped} completed points from existing {points_path}.", flush=True)

    workers = max(1, int(args.workers))
    checkpoint_every = max(1, int(args.checkpoint_every))
    interrupted = False
    pending_logical = len(payloads)
    dedupe_groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    for payload in payloads:
        dedupe_key = _compute_dedupe_key(payload)
        dedupe_groups.setdefault(dedupe_key, []).append(payload)
    pending_unique = len(dedupe_groups)
    dedupe_saved = pending_logical - pending_unique
    if payloads:
        if dedupe_saved > 0:
            print(
                f"Deduplicated {dedupe_saved} logical points by effective gamma "
                f"({pending_logical} -> {pending_unique} unique computations).",
                flush=True,
            )
        print(
            f"Running {pending_logical} pending logical points as {pending_unique} "
            f"unique computations with {workers} workers...",
            flush=True,
        )
        executor = ProcessPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(_compute_one, group[0]): dedupe_key
            for dedupe_key, group in dedupe_groups.items()
        }
        total = len(futures)
        completed_since_start = 0
        try:
            for idx, future in enumerate(as_completed(futures), start=1):
                dedupe_key = futures[future]
                payload_group = dedupe_groups[dedupe_key]
                payload_rep = payload_group[0]
                try:
                    row = future.result()
                    for payload_alias in payload_group:
                        alias_row = dict(row)
                        alias_row["gamma_corr"] = float(payload_alias["gamma_corr"])
                        alias_row["lambda"] = float(payload_alias["lambda"])
                        alias_row["gamma_mode"] = str(payload_alias["gamma_mode"])
                        alias_row["gamma_corr_effective"] = _effective_gamma(
                            gamma_base=float(payload_alias["gamma_corr"]),
                            lambda_value=float(payload_alias["lambda"]),
                            mode=str(payload_alias["gamma_mode"]),
                            lambda_low=float(payload_alias["gamma_lambda_low"]),
                            lambda_high=float(payload_alias["gamma_lambda_high"]),
                            taper_power=float(payload_alias["gamma_taper_power"]),
                        )
                        rows.append(alias_row)
                    print(
                        f"[{idx}/{total}] {row['topology']} gamma={row['gamma_corr']:.3f} "
                        f"(eff={row['gamma_corr_effective']:.3f}) "
                        f"lambda={row['lambda']:.3f} spin2_res={row['spin2_residual_power']:.3f} "
                        f"postulate_res={row['postulate_residual']:.3f} "
                        f"(expanded={len(payload_group)})",
                        flush=True,
                    )
                except Exception as exc:
                    for payload_alias in payload_group:
                        err = dict(payload_alias)
                        err["error"] = str(exc)
                        err_rows.append(err)
                    print(
                        f"[{idx}/{total}] ERROR {payload_rep}: {exc} "
                        f"(expanded={len(payload_group)})",
                        flush=True,
                    )

                completed_since_start += 1
                if completed_since_start % checkpoint_every == 0:
                    _write_checkpoints(points_path=points_path, rows=rows, error_path=error_path, err_rows=err_rows)
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted. Writing checkpoints and cancelling outstanding workers...", flush=True)
            for fut in futures:
                fut.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            _write_checkpoints(points_path=points_path, rows=rows, error_path=error_path, err_rows=err_rows)
        finally:
            if not interrupted:
                executor.shutdown(wait=True)
    else:
        print("No pending points to run (all requested points already completed).", flush=True)

    points = pd.DataFrame(rows)
    points.to_csv(points_path, index=False)

    summary_rows: List[Dict[str, object]] = []
    for (topology, gamma), subset in points.groupby(["topology", "gamma_corr"]):
        subset = subset.sort_values("lambda")
        valid = subset[np.isfinite(subset["spin2_residual_power"])]
        if valid.empty:
            continue
        best_idx = valid["spin2_residual_power"].idxmin()
        best = valid.loc[best_idx]
        summary_rows.append(
            {
                "topology": topology,
                "gamma_corr": float(gamma),
                "n_points": int(len(subset)),
                "best_lambda": float(best["lambda"]),
                "best_spin2_power": float(best["spin2_measured_power"]),
                "best_spin2_slope": float(best["spin2_measured_slope"]),
                "best_spin2_residual": float(best["spin2_residual_power"]),
                "best_spin2_r2": float(best["spin2_r2"]),
                "best_postulate_residual": float(best["postulate_residual"]),
                "mean_postulate_residual": float(valid["postulate_residual"].mean()),
                "max_postulate_residual": float(valid["postulate_residual"].max()),
                "p90_postulate_residual": float(np.quantile(valid["postulate_residual"], 0.9)),
                "frac_postulate_gt_0_2": float(np.mean(valid["postulate_residual"] > 0.2)),
                "mean_gamma_corr_effective": float(valid["gamma_corr_effective"].mean()),
                "min_gamma_corr_effective": float(valid["gamma_corr_effective"].min()),
                "max_gamma_corr_effective": float(valid["gamma_corr_effective"].max()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["topology", "best_spin2_residual", "gamma_corr"],
            ascending=[True, True, True],
        )
    summary.to_csv(summary_path, index=False)
    if err_rows:
        pd.DataFrame(err_rows).to_csv(error_path, index=False)
    elif error_path.exists():
        error_path.unlink()

    _plot_best_spin2(summary, plot_path)

    report_lines = [
        "# v3over9000 gamma sweep",
        "",
        "## Configuration",
        f"- N: {args.N}",
        f"- alpha: {args.alpha}",
        f"- topologies: {','.join(topologies)}",
        f"- gammas: {','.join(str(g) for g in gammas)}",
        f"- lambdas: {','.join(str(l) for l in lambdas)}",
        f"- gamma_mode: {args.gamma_mode}",
        f"- gamma_lambda_low: {args.gamma_lambda_low}",
        f"- gamma_lambda_high: {args.gamma_lambda_high}",
        f"- gamma_taper_power: {args.gamma_taper_power}",
        f"- bond_cutoff: {args.bond_cutoff}",
        f"- hotspot_multiplier: {args.hotspot_multiplier}",
        f"- frustration_time: {args.frustration_time}",
        f"- workers: {workers}",
        f"- resume: {bool(args.resume)}",
        f"- checkpoint_every: {checkpoint_every}",
        f"- pending_logical_points: {pending_logical}",
        f"- pending_unique_computations: {pending_unique}",
        f"- dedupe_saved: {dedupe_saved}",
        f"- interrupted: {interrupted}",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        report_lines.extend(["(no valid points)", ""])
    else:
        report_lines.extend(["```csv", summary.to_csv(index=False).strip(), "```", ""])
    report_lines.extend(
        [
            "## Artifacts",
            f"- `{points_path}`",
            f"- `{summary_path}`",
            f"- `{plot_path}`",
        ]
    )
    if err_rows:
        report_lines.append(f"- `{error_path}`")

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"\nOUTPUT_DIR={out_dir}")
    print(f"POINTS={points_path}")
    print(f"SUMMARY={summary_path}")
    print(f"PLOT={plot_path}")
    if err_rows:
        print(f"ERRORS={error_path}")
    print(f"REPORT={report_path}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep topology x gamma_corr x lambda (v3over9000)")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--topologies", type=str, default="path,cycle,star")
    parser.add_argument("--gammas", type=str, default="0.0,0.02,0.05,0.1")
    parser.add_argument("--lambdas", type=str, default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument(
        "--gamma-mode",
        type=str,
        default="constant",
        choices=["constant", "taper_high"],
        help="How gamma_corr depends on lambda.",
    )
    parser.add_argument(
        "--gamma-lambda-low",
        type=float,
        default=0.4,
        help="For taper_high: lambda where full gamma is still applied.",
    )
    parser.add_argument(
        "--gamma-lambda-high",
        type=float,
        default=0.8,
        help="For taper_high: lambda where gamma reaches 0.",
    )
    parser.add_argument(
        "--gamma-taper-power",
        type=float,
        default=1.0,
        help="For taper_high: taper curve exponent.",
    )

    parser.add_argument("--bond-cutoff", type=int, default=4)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--J0", type=float, default=0.01)
    parser.add_argument("--deltaB", type=float, default=5.0)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--k0", type=float, default=4.0)
    parser.add_argument("--hotspot-multiplier", type=float, default=1.5)
    parser.add_argument("--frustration-time", type=float, default=1.0)

    parser.add_argument("--k-modes", type=int, default=16)
    parser.add_argument("--k-angles", type=int, default=24)
    parser.add_argument("--ramsey-tmax", type=float, default=14.0)
    parser.add_argument("--ramsey-points", type=int, default=80)

    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing points.csv in --output-dir by skipping completed topology/gamma/lambda points.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write checkpoint files every N completed points.",
    )
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
