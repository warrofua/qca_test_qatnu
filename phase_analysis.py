"""
Phase diagram plotting and critical point detection utilities.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class PhaseAnalyzer:
    """Critical point detection and publication-quality plotting."""

    @staticmethod
    def analyze_critical_points(df: pd.DataFrame) -> Dict:
        try:
            lambda_c1 = df[df["residual"] > 0.05]["lambda"].iloc[0]
        except IndexError:
            lambda_c1 = None

        lambda_revival_first = None
        residual_min_first = None
        lambda_revival_global = None
        residual_min_global = None
        revival_method = "none"
        first_violation_lambda = None

        try:
            violation_region = df[df["residual"] > 0.10]
            if not violation_region.empty:
                first_viol_idx = violation_region.index[0]
                first_violation_lambda = float(df.loc[first_viol_idx, "lambda"])
                post_violation = df.loc[first_viol_idx:]

                if not post_violation.empty:
                    global_idx = post_violation["residual"].idxmin()
                    lambda_revival_global = float(df.loc[global_idx, "lambda"])
                    residual_min_global = float(df.loc[global_idx, "residual"])

                    minima_indices, _ = find_peaks(
                        -post_violation["residual"].values, height=-0.15, distance=5
                    )

                    if len(minima_indices) > 0:
                        first_idx = post_violation.index[minima_indices[0]]
                        lambda_revival_first = float(df.loc[first_idx, "lambda"])
                        residual_min_first = float(df.loc[first_idx, "residual"])
                        revival_method = "first_local_minimum"
                    else:
                        # No local revival detected; fall back to global minimum in the post-violation region.
                        lambda_revival_first = lambda_revival_global
                        residual_min_first = residual_min_global
                        revival_method = "global_minimum_fallback"
        except Exception:
            pass

        # Official reporting rule: use global minimum in the post-violation region.
        # Keep first-local values as diagnostics for method-sensitivity audits.
        lambda_revival = lambda_revival_global
        residual_min = residual_min_global

        try:
            if lambda_revival is not None:
                post_revival = df[df["lambda"] > lambda_revival]
                lambda_c2 = post_revival[post_revival["residual"] > 0.20]["lambda"].iloc[0]
            else:
                lambda_c2 = None
        except IndexError:
            lambda_c2 = None

        return {
            "lambda_c1": lambda_c1,
            "lambda_revival": lambda_revival,
            "residual_min": residual_min,
            "lambda_revival_first": lambda_revival_first,
            "residual_min_first": residual_min_first,
            "lambda_revival_global": lambda_revival_global,
            "residual_min_global": residual_min_global,
            "first_violation_lambda": first_violation_lambda,
            "revival_method": revival_method,
            "revival_reporting_rule": (
                "global_minimum_post_violation" if lambda_revival is not None else "none"
            ),
            "revival_gap": (
                abs(lambda_revival_first - lambda_revival_global)
                if lambda_revival_first is not None and lambda_revival_global is not None
                else None
            ),
            "lambda_c2": lambda_c2,
        }

    @staticmethod
    def plot_phase_diagram(
        df: pd.DataFrame,
        N: int,
        alpha: float,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Dict]:
        crit = PhaseAnalyzer.analyze_critical_points(df)

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df["lambda"], df["residual"] * 100, "o-", linewidth=2.5, markersize=8, color="darkblue", label="Postulate 1 Residual")
        ax1.axhline(5, color="red", linestyle="--", linewidth=2, alpha=0.7, label="5% threshold")
        ax1.axhline(10, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="10% threshold")
        ax1.fill_between(df["lambda"], 0, 5, alpha=0.15, color="green", label="Emergent (Phase I)")
        ax1.fill_between(df["lambda"], 5, 50, alpha=0.15, color="yellow", label="Breakdown (Phase II)")
        ax1.fill_between(df["lambda"], 50, 150, alpha=0.15, color="red", label="Catastrophic (Phase IV)")

        if crit["lambda_c1"] is not None:
            ax1.axvline(crit["lambda_c1"], color="black", linestyle="-", linewidth=1.5, alpha=0.7, label=f"λ_c1={crit['lambda_c1']:.3f}")
        if crit["lambda_revival"] is not None and crit["residual_min"] is not None:
            ax1.axvline(crit["lambda_revival"], color="purple", linestyle=":", linewidth=2, label=f"Revival (λ={crit['lambda_revival']:.3f})")
            ax1.annotate(
                f"Quantum Revival\nλ={crit['lambda_revival']:.3f}\nR={crit['residual_min']*100:.2f}%",
                xy=(crit["lambda_revival"], crit["residual_min"] * 100),
                xytext=(crit["lambda_revival"] + 0.15, crit["residual_min"] * 100 + 30),
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
                fontsize=11,
                ha="center",
                color="purple",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
        if crit["lambda_c2"] is not None:
            ax1.axvline(crit["lambda_c2"], color="gray", linestyle="-", linewidth=1.5, alpha=0.7, label=f"λ_c2={crit['lambda_c2']:.3f}")

        ax1.set_xlabel("λ (promotion strength)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Postulate 1 Residual (%)", fontsize=14, fontweight="bold")
        ax1.set_title(f"Quantum Phase Diagram: N={N}, α={alpha}", fontsize=16, fontweight="bold", pad=15)
        ax1.legend(loc="upper left", fontsize=10, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(df["lambda"].min(), df["lambda"].max())
        ax1.set_ylim(0, max(df["residual"] * 100) * 1.1)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df["lambda"], df["omega_out"], "o-", label="Measured ω_out", linewidth=2, markersize=6, color="steelblue")
        ax2.plot(df["lambda"], df["omega_in"], "s-", label="Measured ω_in", linewidth=2, markersize=6, color="coral")
        ax2.plot(df["lambda"], df["predicted_omega_out"], "--", alpha=0.6, color="steelblue", label="Predicted ω_out")
        ax2.plot(df["lambda"], df["predicted_omega_in"], "--", alpha=0.6, color="coral", label="Predicted ω_in")
        ax2.axhline(1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.7, label="Bare ω")
        if crit["lambda_revival"] is not None:
            ax2.axvline(crit["lambda_revival"], color="purple", linestyle=":", linewidth=1.5, alpha=0.5)
        ax2.set_xlabel("λ", fontsize=12, fontweight="bold")
        ax2.set_ylabel("ω_eff", fontsize=12, fontweight="bold")
        ax2.set_title("Frequency Scaling", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        high_lambda = df[df["lambda"] > 0.6]
        if not high_lambda.empty:
            ax3.plot(high_lambda["lambda"], high_lambda["omega_out"], "o-", label="ω_out", color="steelblue")
            ax3.plot(high_lambda["lambda"], high_lambda["omega_in"], "s-", label="ω_in", color="coral")
            ax3.axhline(1.0, color="black", linestyle=":", alpha=0.7)
            ax3.fill_between(high_lambda["lambda"], 1.0, high_lambda["omega_in"], alpha=0.2, color="red", label="Inversion region")
            ax3.set_xlabel("λ", fontsize=12, fontweight="bold")
            ax3.set_ylabel("ω_eff", fontsize=12, fontweight="bold")
            ax3.set_title("Frequency Inversion", fontsize=13, fontweight="bold")
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(df["lambda"], df["lambda_out"], "o-", label="Λ_out", linewidth=2, color="steelblue")
        ax4.plot(df["lambda"], df["lambda_in"], "s-", label="Λ_in", linewidth=2, color="coral")
        if crit["lambda_revival"] is not None:
            ax4.axvline(crit["lambda_revival"], color="purple", linestyle=":", linewidth=1.5, alpha=0.5)
        ax4.set_xlabel("λ", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Λ (circuit depth)", fontsize=12, fontweight="bold")
        ax4.set_title("Entanglement Proxy", fontsize=13, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[2, 0])
        measured_ratio = df["omega_in"] / df["omega_out"]
        predicted_ratio = df["predicted_omega_in"] / df["predicted_omega_out"]
        ax5.plot(df["lambda"], measured_ratio, "o-", label="Measured Ratio", linewidth=2, markersize=6, color="darkgreen")
        ax5.plot(df["lambda"], predicted_ratio, "s-", label="Predicted Ratio", linewidth=2, markersize=6, color="orange")
        ax5.axhline(1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.7, label="No scaling")
        if crit["lambda_revival"] is not None:
            ax5.axvline(crit["lambda_revival"], color="purple", linestyle=":", linewidth=1.5, alpha=0.5)
        ax5.set_xlabel("λ", fontsize=12, fontweight="bold")
        ax5.set_ylabel("ω_in / ω_out", fontsize=12, fontweight="bold")
        ax5.set_title("Frequency Ratio Deviation", fontsize=13, fontweight="bold")
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 1:])
        status_colors = {"✓": 0, "~": 1, "✗": 2}
        status_numeric = [status_colors.get(s, 1) for s in df["status"]]
        scatter = ax6.scatter(
            df["lambda"],
            [0] * len(df),
            c=status_numeric,
            cmap="RdYlGn_r",
            s=200,
            marker="s",
            edgecolors="black",
            alpha=0.8,
        )
        ax6.set_xlabel("λ", fontsize=12, fontweight="bold")
        ax6.set_yticks([])
        ax6.set_title("Phase Classification", fontsize=13, fontweight="bold")
        ax6.set_xlim(df["lambda"].min(), df["lambda"].max())
        cbar = plt.colorbar(scatter, ax=ax6, orientation="horizontal", pad=0.25, ticks=[0, 1, 2])
        cbar.ax.set_xticklabels(["✓ Emergent", "~ Within Tol.", "✗ Violated"], fontsize=10)

        lambda_c1_str = f"{crit['lambda_c1']:.3f}" if crit["lambda_c1"] is not None else "N/A"
        lambda_revival_str = f"{crit['lambda_revival']:.3f}" if crit["lambda_revival"] is not None else "N/A"
        residual_min_str = f"{crit['residual_min']*100:.2f}" if crit["residual_min"] is not None else "N/A"
        lambda_c2_str = f"{crit['lambda_c2']:.3f}" if crit["lambda_c2"] is not None else "N/A"

        summary_text = f"""
        Critical Points:
        • Scaling breakdown: λ_c1 ≈ {lambda_c1_str}
        • Quantum revival: λ_rev ≈ {lambda_revival_str} (R={residual_min_str}%)
        • Catastrophic fail: λ_c2 ≈ {lambda_c2_str}

        Regimes:
        Phase I (λ < {lambda_c1_str}): Postulate 1 emerges
        Phase II ({lambda_c1_str} < λ < {lambda_revival_str}): Breakdown
        Phase III (λ ≈ {lambda_revival_str}): Quantum revival
        Phase IV (λ > {lambda_c2_str}): Frequency inversion
        """

        fig.text(
            0.5,
            0.02,
            summary_text.strip(),
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7),
            fontfamily="monospace",
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"🖼️  Plot saved to {save_path}")

        return fig, crit
