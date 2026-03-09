from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from core_qca import ExactQCA
from geometry import EmergentGeometryAnalyzer
from runtime_config import configure_runtime, detect_accelerate
from scanners import ParameterScanner
from srqid import SRQIDValidators
from tester import QCATester, plot_ramsey_overlay
from topologies import get_topology


def _timestamped_dirs(
    N: int,
    alpha: float,
    graph: str,
    bond_cutoff: int = 4,
    lambda_min: float = 0.1,
    lambda_max: float = 1.5,
    num_points: int = 100,
    **extra_kwargs
) -> Dict[str, str]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Build base run tag
    run_tag = f"run_{ts}_N{N}_{graph}_alpha{alpha:.2f}"
    
    # Add non-default parameters to tag for tracking
    extras = []
    if bond_cutoff != 4:
        extras.append(f"chi{bond_cutoff}")
    if abs(lambda_min - 0.1) > 1e-6 or abs(lambda_max - 1.5) > 1e-6:
        extras.append(f"lam{lambda_min:.2f}-{lambda_max:.2f}")
    if num_points != 100:
        extras.append(f"pts{num_points}")
    
    # Add any extra kwargs that were passed
    for key, value in extra_kwargs.items():
        if value is not None and key not in ('edges', 'probes', 'run_phase_space'):
            # Clean up key name (remove underscores, limit length)
            clean_key = key.replace('_', '')[:4]
            if isinstance(value, float):
                extras.append(f"{clean_key}{value:.2f}")
            elif isinstance(value, int):
                extras.append(f"{clean_key}{value}")
            elif isinstance(value, str) and len(value) <= 4:
                extras.append(f"{clean_key}{value}")
    
    if extras:
        run_tag += "_" + "_".join(extras)
    
    output_dir = os.path.join("outputs", run_tag)
    fig_dir = os.path.join("figures", run_tag)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    return {"tag": run_tag, "outputs": output_dir, "figures": fig_dir, "timestamp": ts}


def _choose_lambda_focus(crit: Dict, default: float) -> float:
    for key in ("lambda_revival", "lambda_c1", "lambda_c2"):
        if crit.get(key):
            return float(crit[key])
    return default


def _parse_optional_json_list(raw: Optional[str], flag_name: str) -> Optional[List[float]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{flag_name} must decode to a JSON list.")
    return [float(item) for item in value]


def _parse_optional_hotspot_stages(raw: Optional[str], flag_name: str) -> Optional[List[Dict]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{flag_name} must decode to a JSON list of stage objects.")
    stages: List[Dict] = []
    for idx, stage in enumerate(value):
        if not isinstance(stage, dict):
            raise ValueError(f"{flag_name} stage {idx} must be an object.")
        clean = dict(stage)
        if "multiplier" in clean:
            clean["multiplier"] = float(clean["multiplier"])
        if "time" in clean:
            clean["time"] = float(clean["time"])
        if "edge_weights" in clean and clean["edge_weights"] is not None:
            if not isinstance(clean["edge_weights"], list):
                raise ValueError(f"{flag_name} stage {idx} edge_weights must be a list.")
            clean["edge_weights"] = [float(item) for item in clean["edge_weights"]]
        stages.append(clean)
    return stages


def _write_summary(summary_path: str, context: Dict) -> None:
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("QATNU/SRQID Production Run Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {context['timestamp']}\n")
        f.write(f"Run tag: {context['tag']}\n")
        f.write(
            "N={N}, graph={graph}, α={alpha}, points={points}, hotspot_multiplier={mult:.2f}x, hotspot_time={ht:.2f}\n\n".format(
                N=context["N"],
                graph=context.get("graph", "path"),
                alpha=context["alpha"],
                points=context["num_points"],
                mult=context.get("hotspot_multiplier", 3.0),
                ht=context.get("hotspot_time", 1.0),
            )
        )
        if abs(context.get("gamma_corr", 0.0)) > 1e-12 or abs(context.get("gamma_corr_diag", 0.0)) > 1e-12:
            f.write(
                "Correlated promotion: gamma_corr={gc:.4f}, gamma_corr_diag={gcd:.4f}\n\n".format(
                    gc=context.get("gamma_corr", 0.0),
                    gcd=context.get("gamma_corr_diag", 0.0),
                )
            )
        if (
            abs(context.get("readout_gamma_corr", context.get("gamma_corr", 0.0)) - context.get("gamma_corr", 0.0)) > 1e-12
            or abs(
                context.get("readout_gamma_corr_diag", context.get("gamma_corr_diag", 0.0))
                - context.get("gamma_corr_diag", 0.0)
            )
            > 1e-12
        ):
            f.write(
                "Readout correlated promotion: gamma_corr={gc:.4f}, gamma_corr_diag={gcd:.4f}\n\n".format(
                    gc=context.get("readout_gamma_corr", context.get("gamma_corr", 0.0)),
                    gcd=context.get("readout_gamma_corr_diag", context.get("gamma_corr_diag", 0.0)),
                )
            )
        if context.get("hotspot_edge_weights") is not None:
            f.write(f"Hotspot edge weights: {context['hotspot_edge_weights']}\n\n")
        if context.get("hotspot_stages") is not None:
            f.write(f"Hotspot stages: {context['hotspot_stages']}\n\n")

        f.write("Critical Points:\n")
        for name in ("lambda_c1", "lambda_revival", "lambda_c2"):
            f.write(f"  {name}: {context['critical_points'].get(name)}\n")
        f.write(f"  residual_min: {context['critical_points'].get('residual_min')}\n\n")
        f.write("Revival Diagnostics:\n")
        f.write(f"  first_violation_lambda: {context['critical_points'].get('first_violation_lambda')}\n")
        f.write(f"  revival_reporting_rule: {context['critical_points'].get('revival_reporting_rule')}\n")
        f.write(f"  revival_method: {context['critical_points'].get('revival_method')}\n")
        f.write(f"  lambda_revival_first: {context['critical_points'].get('lambda_revival_first')}\n")
        f.write(f"  residual_min_first: {context['critical_points'].get('residual_min_first')}\n")
        f.write(f"  lambda_revival_global: {context['critical_points'].get('lambda_revival_global')}\n")
        f.write(f"  residual_min_global: {context['critical_points'].get('residual_min_global')}\n")
        f.write(f"  revival_gap: {context['critical_points'].get('revival_gap')}\n\n")

        f.write("SRQID Validations:\n")
        f.write(f"  v_LR: {context['v_lr']}\n")
        f.write(f"  No-signalling: {context['ns_violation']:.3e}\n")
        f.write(f"  Energy drift: {context['energy_err']:.3e}\n\n")

        mf = context["validation"]
        f.write("Mean-field Comparison:\n")
        f.write(f"  λ_focus: {context['lambda_focus']:.4f}\n")
        f.write(f"  Residual: {mf['residual']:.4e}\n")
        f.write(f"  Measured ω_out / ω_in: {mf['measuredFreqOut']:.4f} / {mf['measuredFreqIn']:.4f}\n")
        f.write(f"  Predicted ω_out / ω_in: {mf['predictedFreqOut']:.4f} / {mf['predictedFreqIn']:.4f}\n\n")

        spin2 = context["spin2"]
        f.write("Spin-2 PSD:\n")
        f.write(f"  Measured power: {spin2['measured_power']:.3f}\n")
        f.write(f"  Expected power: {spin2['expected_power']:.3f}\n")
        f.write(f"  Residual: {spin2['residual']:.3e}\n\n")

        f.write("Artifacts:\n")
        for key, path in context["artifacts"].items():
            f.write(f"  {key}: {path}\n")


def production_run(
    N: int = 4,
    alpha: float = 0.8,
    num_points: int = 100,
    run_phase_space: bool = False,
    edges: Optional[List[Tuple[int, int]]] = None,
    probes: Tuple[int, int] = (0, 1),
    graph_name: str = "path",
    lambda_min: float = 0.1,
    lambda_max: float = 1.5,
    bond_cutoff: int = 4,
    hotspot_multiplier: float = 3.0,
    hotspot_time: float = 1.0,
    hotspot_edge_weights: Optional[List[float]] = None,
    hotspot_stages: Optional[List[Dict]] = None,
    gamma_corr: float = 0.0,
    gamma_corr_diag: float = 0.0,
    readout_gamma_corr: Optional[float] = None,
    readout_gamma_corr_diag: Optional[float] = None,
) -> Dict:
    if edges is None:
        edges = [(i, i + 1) for i in range(max(N - 1, 0))]
    dirs = _timestamped_dirs(
        N, alpha, graph_name,
        bond_cutoff=bond_cutoff,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        num_points=num_points,
        hotspot_multiplier=hotspot_multiplier if abs(hotspot_multiplier - 3.0) > 1e-9 else None,
        hotspot_time=hotspot_time if abs(hotspot_time - 1.0) > 1e-9 else None,
        gamma_corr=gamma_corr if abs(gamma_corr) > 1e-12 else None,
        gamma_corr_diag=gamma_corr_diag if abs(gamma_corr_diag) > 1e-12 else None,
    )
    timestamp = dirs["timestamp"]
    run_tag = dirs["tag"]

    print("=" * 60)
    print("QATNU/SRQID PRODUCTION RUN")
    print("=" * 60)
    print(f"Parameters: N={N}, α={alpha}, points={num_points}")
    print(f"Graph topology: {graph_name}, probes={probes}")
    print(f"Hotspot multiplier: {hotspot_multiplier:.2f}×")
    print(f"Hotspot time: {hotspot_time:.2f}")
    if hotspot_edge_weights is not None:
        print(f"Hotspot edge weights: {hotspot_edge_weights}")
    if hotspot_stages is not None:
        print(f"Hotspot stages: {hotspot_stages}")
    if gamma_corr != 0.0 or gamma_corr_diag != 0.0:
        print(f"Correlated params: gamma_corr={gamma_corr:.3f}, gamma_corr_diag={gamma_corr_diag:.3f}")
    effective_readout_gamma_corr = gamma_corr if readout_gamma_corr is None else readout_gamma_corr
    effective_readout_gamma_corr_diag = (
        gamma_corr_diag if readout_gamma_corr_diag is None else readout_gamma_corr_diag
    )
    if (
        abs(effective_readout_gamma_corr - gamma_corr) > 1e-12
        or abs(effective_readout_gamma_corr_diag - gamma_corr_diag) > 1e-12
    ):
        print(
            "Readout correlated params: "
            f"gamma_corr={effective_readout_gamma_corr:.3f}, "
            f"gamma_corr_diag={effective_readout_gamma_corr_diag:.3f}"
        )
    print(f"System: {platform.system()} ({platform.machine()})")
    print(f"Run tag: {run_tag}")
    print("=" * 60)

    scanner = ParameterScanner()
    df = scanner.scan_lambda_parallel(
        N=N,
        alpha=alpha,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        num_points=num_points,
        bond_cutoff=bond_cutoff,
        output_dir=dirs["outputs"],
        run_tag=run_tag,
        edges=edges,
        probes=probes,
        hotspot_multiplier=hotspot_multiplier,
        hotspot_time=hotspot_time,
        hotspot_edge_weights=hotspot_edge_weights,
        hotspot_stages=hotspot_stages,
        gamma_corr=gamma_corr,
        gamma_corr_diag=gamma_corr_diag,
        readout_gamma_corr=effective_readout_gamma_corr,
        readout_gamma_corr_diag=effective_readout_gamma_corr_diag,
    )

    from phase_analysis import PhaseAnalyzer
    
    analyzer = PhaseAnalyzer()
    phase_path = os.path.join(dirs["figures"], f"phase_diagram_{run_tag}.png")
    fig, crit = analyzer.plot_phase_diagram(df, N=N, alpha=alpha, save_path=phase_path)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("CRITICAL POINT ANALYSIS")
    print("=" * 60)
    for name, value in crit.items():
        print(f"{name}: {value}")

    print("\nRunning SRQID structural validation...")
    test_config = {
        "N": N,
        "omega": 1.0,
        "deltaB": 5.0,
        "lambda": 0.5,
        "kappa": 0.1,
        "k0": 4,
        "bondCutoff": 4,
        "J0": 0.01,
        "gamma": 0.0,
        "gamma_corr": gamma_corr,
        "gamma_corr_diag": gamma_corr_diag,
        "probeOut": probes[0],
        "probeIn": probes[1],
        "edges": edges,
    }
    qca_val = ExactQCA(N, test_config, edges=edges)
    v_lr, _ = SRQIDValidators.extract_lr_velocity(qca_val)
    ns_violation = SRQIDValidators.no_signalling_quench(qca_val)
    energy_err = SRQIDValidators.energy_drift(qca_val)

    tester = QCATester(
        N=N,
        alpha=alpha,
        bond_cutoff=bond_cutoff,
        hotspot_multiplier=hotspot_multiplier,
        hotspot_time=hotspot_time,
        hotspot_edge_weights=hotspot_edge_weights,
        hotspot_stages=hotspot_stages,
        gamma_corr=gamma_corr,
        gamma_corr_diag=gamma_corr_diag,
        readout_gamma_corr=effective_readout_gamma_corr,
        readout_gamma_corr_diag=effective_readout_gamma_corr_diag,
        edges=edges,
        probes=probes,
    )
    lambda_focus = _choose_lambda_focus(crit, default=df["lambda"].iloc[len(df) // 2])
    tester_config = tester.parameters.copy()
    tester_config["lambda"] = lambda_focus
    exact, mean_field, validation = tester.run_full_validation(config=tester_config)
    overlay_path = None
    if mean_field is not None:
        overlay_path = os.path.join(dirs["figures"], f"ramsey_overlay_{run_tag}.png")
        plot_ramsey_overlay(exact, mean_field, overlay_path, title_prefix="Ramsey")

    geom = EmergentGeometryAnalyzer(rng_seed=int(datetime.now().timestamp()))
    spin2_title = "Spin-2 Tail (bond correlator)"
    try:
        ks, psd = geom.spin2_from_bond_correlators(exact["qca"], exact["frustrated_state"])
        if ks.size == 0 or psd.size == 0:
            raise ValueError("insufficient bond correlator modes")
    except Exception:
        chi_profile = [d["chi"] for d in exact["bondDims"]]
        ks, psd = geom.spin2_from_chi_profile(chi_profile)
        spin2_title = "Spin-2 Tail (χ-informed fallback)"
    spin2_fit = geom.analyze_spin2_scaling(ks, psd)
    spin2_path = os.path.join(dirs["figures"], f"spin2_psd_{run_tag}.png")
    geom.plot_spin2_tail(ks, psd, title=spin2_title, save_path=spin2_path)

    artifacts = {
        "scan": os.path.join(dirs["outputs"], f"scan_{run_tag}.csv"),
        "phase_diagram": phase_path,
        "spin2_psd": spin2_path,
    }
    if overlay_path:
        artifacts["ramsey_overlay"] = overlay_path

    summary_path = os.path.join(dirs["outputs"], f"summary_{run_tag}.txt")
    context = {
        "timestamp": timestamp,
        "tag": run_tag,
        "N": N,
        "alpha": alpha,
        "num_points": num_points,
        "graph": graph_name,
        "hotspot_multiplier": hotspot_multiplier,
        "hotspot_time": hotspot_time,
        "hotspot_edge_weights": hotspot_edge_weights,
        "hotspot_stages": hotspot_stages,
        "gamma_corr": gamma_corr,
        "gamma_corr_diag": gamma_corr_diag,
        "readout_gamma_corr": effective_readout_gamma_corr,
        "readout_gamma_corr_diag": effective_readout_gamma_corr_diag,
        "critical_points": crit,
        "v_lr": v_lr,
        "ns_violation": ns_violation,
        "energy_err": energy_err,
        "validation": validation,
        "lambda_focus": lambda_focus,
        "spin2": spin2_fit,
        "artifacts": artifacts,
    }
    _write_summary(summary_path, context)
    artifacts["summary"] = summary_path
    print(f"\n💾 Summary saved to {summary_path}")

    phase_space_artifact = None
    if run_phase_space:
        df_2d = scanner.scan_2d_phase_space(
            N=N,
            bond_cutoff=bond_cutoff,
            output_dir=dirs["outputs"],
            run_tag=run_tag,
            edges=edges,
            probes=probes,
            hotspot_multiplier=hotspot_multiplier,
            hotspot_time=hotspot_time,
            hotspot_edge_weights=hotspot_edge_weights,
            hotspot_stages=hotspot_stages,
            gamma_corr=gamma_corr,
            gamma_corr_diag=gamma_corr_diag,
            readout_gamma_corr=effective_readout_gamma_corr,
            readout_gamma_corr_diag=effective_readout_gamma_corr_diag,
        )
        heatmap_path = os.path.join(dirs["figures"], f"phase_space_{run_tag}.png")
        _plot_phase_space_heatmap(df_2d, heatmap_path)
        print(f"🖼️  2D heatmap saved to {heatmap_path}")
        phase_space_artifact = heatmap_path

    return {
        "dataframe": df,
        "critical_points": crit,
        "artifacts": artifacts,
        "phase_space": phase_space_artifact,
    }


def _plot_phase_space_heatmap(df: pd.DataFrame, save_path: str) -> None:
    import seaborn as sns

    pivot = df.pivot(index="lambda", columns="alpha", values="residual") * 100
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        cmap="RdYlGn_r",
        center=10,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Residual (%)"},
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title("2D Phase Diagram: Postulate 1 Residual", fontsize=16, fontweight="bold")
    plt.xlabel("α (postulate coefficient)", fontsize=12, fontweight="bold")
    plt.ylabel("λ (promotion strength)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_legacy_visuals() -> None:
    """Execute the Dataverse proof-of-concept visualization scripts."""
    legacy_script = Path("dataverse_files/qatnu_poc.py")
    if not legacy_script.exists():
        print("⚠️  Legacy visualization script not found; skipping.")
        return
    print("\n" + "=" * 60)
    print("Running legacy visualization (qatnu_poc.py)")
    print("=" * 60)
    try:
        subprocess.run(
            [sys.executable, str(legacy_script)],
            check=True,
        )
        print("✅ Legacy visualization completed.")
    except subprocess.CalledProcessError as exc:
        print(f"⚠️  Legacy visualization failed with return code {exc.returncode}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QATNU/SRQID unified simulation driver")
    parser.add_argument("--N", type=int, default=5, help="Number of sites")
    parser.add_argument("--alpha", type=float, default=0.8, help="Postulate coefficient")
    parser.add_argument("--points", type=int, default=100, help="Number of λ scan points")
    parser.add_argument("--phase-space", action="store_true", help="Run 2D phase space scan")
    parser.add_argument(
        "--legacy-viz",
        action="store_true",
        help="Also run the Dataverse legacy visualization scripts (qatnu_poc).",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="path",
        help="Graph topology (path, cycle, diamond, bowtie, pyramid/star).",
    )
    parser.add_argument(
        "--probes",
        type=int,
        nargs=2,
        metavar=("OUTER", "INNER"),
        help="Override probe vertex indices (outer, inner).",
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=0.1,
        help="Minimum λ for the 1D scan",
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=1.5,
        help="Maximum λ for the 1D scan",
    )
    parser.add_argument(
        "--bond-cutoff",
        type=int,
        default=4,
        help="Maximum bond dimension (χ_max). Default 4. Increase for high-λ catastrophe physics, decrease for larger N.",
    )
    parser.add_argument(
        "--hotspot-multiplier",
        type=float,
        default=3.0,
        help="Multiplier applied to λ when preparing the frustrated hotspot background (default: 3.0).",
    )
    parser.add_argument(
        "--hotspot-time",
        type=float,
        default=1.0,
        help="Evolution time used to prepare the frustrated hotspot background (default: 1.0).",
    )
    parser.add_argument(
        "--hotspot-edge-weights-json",
        type=str,
        default="",
        help="Optional JSON list of per-edge weights used only during hotspot preparation.",
    )
    parser.add_argument(
        "--hotspot-stages-json",
        type=str,
        default="",
        help="Optional JSON list of sequential hotspot stage objects.",
    )
    parser.add_argument(
        "--gamma-corr",
        type=float,
        default=0.0,
        help="Correlated edge-pair promotion amplitude (default: 0.0).",
    )
    parser.add_argument(
        "--gamma-corr-diag",
        type=float,
        default=0.0,
        help="Diagonal correlated bond-pair penalty/reward (default: 0.0).",
    )
    parser.add_argument(
        "--readout-gamma-corr",
        type=float,
        default=None,
        help="Optional correlated edge-pair promotion amplitude used only during readout evolution.",
    )
    parser.add_argument(
        "--readout-gamma-corr-diag",
        type=float,
        default=None,
        help="Optional correlated diagonal amplitude used only during readout evolution.",
    )
    return parser.parse_args()


def main():
    configure_runtime()
    accel, status = detect_accelerate()
    print("=== BLAS Backend Check ===")
    print(status)
    if not accel:
        print("   (If you see this but performance is good, it's a false positive)")

    if platform.system() == "Darwin":
        import multiprocessing

        multiprocessing.set_start_method("forkserver", force=True)
        print("🍎 macOS detected: using 'forkserver' start method")

    args = parse_args()
    hotspot_edge_weights = _parse_optional_json_list(
        args.hotspot_edge_weights_json,
        "--hotspot-edge-weights-json",
    )
    hotspot_stages = _parse_optional_hotspot_stages(
        args.hotspot_stages_json,
        "--hotspot-stages-json",
    )
    topology = get_topology(args.graph, args.N)
    probes = tuple(args.probes) if args.probes else topology.probes
    production_run(
        N=args.N,
        alpha=args.alpha,
        num_points=args.points,
        run_phase_space=args.phase_space,
        edges=topology.edges,
        probes=probes,
        graph_name=topology.name,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        bond_cutoff=args.bond_cutoff,
        hotspot_multiplier=args.hotspot_multiplier,
        hotspot_time=args.hotspot_time,
        hotspot_edge_weights=hotspot_edge_weights,
        hotspot_stages=hotspot_stages,
        gamma_corr=args.gamma_corr,
        gamma_corr_diag=args.gamma_corr_diag,
        readout_gamma_corr=args.readout_gamma_corr,
        readout_gamma_corr_diag=args.readout_gamma_corr_diag,
    )
    if args.legacy_viz:
        run_legacy_visuals()


if __name__ == "__main__":
    main()
