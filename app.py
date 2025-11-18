from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

from core_qca import ExactQCA
from geometry import EmergentGeometryAnalyzer
from phase_analysis import PhaseAnalyzer
from runtime_config import configure_runtime, detect_accelerate
from scanners import ParameterScanner
from srqid import SRQIDValidators
from tester import QCATester, plot_ramsey_overlay
from topologies import get_topology


def _timestamped_dirs(N: int, alpha: float, graph: str) -> Dict[str, str]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = f"run_{ts}_N{N}_{graph}_alpha{alpha:.2f}"
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


def _write_summary(summary_path: str, context: Dict) -> None:
    with open(summary_path, "w") as f:
        f.write("QATNU/SRQID Production Run Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {context['timestamp']}\n")
        f.write(f"Run tag: {context['tag']}\n")
        f.write(
            f"N={context['N']}, graph={context.get('graph','path')}, α={context['alpha']}, points={context['num_points']}\n\n"
        )

        f.write("Critical Points:\n")
        for name in ("lambda_c1", "lambda_revival", "lambda_c2"):
            f.write(f"  {name}: {context['critical_points'].get(name)}\n")
        f.write(f"  residual_min: {context['critical_points'].get('residual_min')}\n\n")

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
) -> Dict:
    if edges is None:
        edges = [(i, i + 1) for i in range(max(N - 1, 0))]
    dirs = _timestamped_dirs(N, alpha, graph_name)
    timestamp = dirs["timestamp"]
    run_tag = dirs["tag"]

    print("=" * 60)
    print("QATNU/SRQID PRODUCTION RUN")
    print("=" * 60)
    print(f"Parameters: N={N}, α={alpha}, points={num_points}")
    print(f"Graph topology: {graph_name}, probes={probes}")
    print(f"System: {platform.system()} ({platform.machine()})")
    print(f"Run tag: {run_tag}")
    print("=" * 60)

    scanner = ParameterScanner()
    df = scanner.scan_lambda_parallel(
        N=N,
        alpha=alpha,
        num_points=num_points,
        output_dir=dirs["outputs"],
        run_tag=run_tag,
        edges=edges,
        probes=probes,
    )

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
        "probeOut": probes[0],
        "probeIn": probes[1],
        "edges": edges,
    }
    qca_val = ExactQCA(N, test_config, edges=edges)
    v_lr, _ = SRQIDValidators.extract_lr_velocity(qca_val)
    ns_violation = SRQIDValidators.no_signalling_quench(qca_val)
    energy_err = SRQIDValidators.energy_drift(qca_val)

    tester = QCATester(N=N, alpha=alpha, edges=edges, probes=probes)
    lambda_focus = _choose_lambda_focus(crit, default=df["lambda"].iloc[len(df) // 2])
    tester_config = tester.parameters.copy()
    tester_config["lambda"] = lambda_focus
    exact, mean_field, validation = tester.run_full_validation(config=tester_config)
    overlay_path = None
    if mean_field is not None:
        overlay_path = os.path.join(dirs["figures"], f"ramsey_overlay_{run_tag}.png")
        plot_ramsey_overlay(exact, mean_field, overlay_path, title_prefix="Ramsey")

    chi_profile = [d["chi"] for d in exact["bondDims"]]
    geom = EmergentGeometryAnalyzer(rng_seed=int(datetime.now().timestamp()))
    ks, psd = geom.spin2_from_chi_profile(chi_profile)
    spin2_fit = geom.analyze_spin2_scaling(ks, psd)
    spin2_path = os.path.join(dirs["figures"], f"spin2_psd_{run_tag}.png")
    geom.plot_spin2_tail(ks, psd, title="Spin-2 Tail (χ-informed)", save_path=spin2_path)

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
            output_dir=dirs["outputs"],
            run_tag=run_tag,
            edges=edges,
            probes=probes,
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
    )
    if args.legacy_viz:
        run_legacy_visuals()


if __name__ == "__main__":
    main()
