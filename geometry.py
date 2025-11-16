"""
Emergent geometry toys (spin-2 PSD analysis).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class EmergentGeometryAnalyzer:
    """Spin-2 tail and dimensional flow analysis."""

    def __init__(self, rng_seed: int = 0):
        self.rng = np.random.default_rng(rng_seed)

    def spin2_power_spectrum_1d(
        self,
        N: int = 2048,
        steps: int = 3000,
        p0: float = 0.03,
        lam: float = 1e-5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        tier = np.zeros(N, int)
        r = np.arange(1, N + 1)
        p_prom = np.minimum(p0 / r**2, 0.15)

        for _ in range(steps):
            tier += self.rng.random(N) < p_prom
            chi = 2 ** tier
            dem = self.rng.random(N) < lam * (chi - 1) ** 3
            tier[dem & (tier > 0)] -= 1

        dchi = 2 ** tier - (2 ** tier).mean()
        k = np.fft.rfftfreq(N) * 2 * np.pi
        psd = np.abs(np.fft.rfft(dchi)) ** 2

        return k[1:], psd[1:]

    def spin2_from_chi_profile(
        self,
        chi_profile: List[float],
        repeats: int = 256,
        steps: int = 3000,
        lam: float = 1e-5,
        base_p0: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Promote/demote tiers using an actual χ profile tiled across a synthetic lattice.
        """
        profile = np.array(chi_profile, dtype=float)
        if profile.max() == 0:
            profile = np.ones_like(profile)
        weights = profile / profile.max()
        tiled = np.tile(weights, repeats)

        tier = np.zeros_like(tiled, dtype=int)
        for _ in range(steps):
            prom_prob = np.clip(base_p0 * (1 + tiled), 1e-4, 0.25)
            tier += self.rng.random(tiled.size) < prom_prob
            chi = 2 ** tier
            dem_prob = lam * (chi - 1) ** 3 * (0.5 + tiled / 2)
            dem = self.rng.random(tiled.size) < dem_prob
            tier[dem & (tier > 0)] -= 1

        detrended = 2 ** tier - np.mean(2 ** tier)
        k = np.fft.rfftfreq(tiled.size) * 2 * np.pi
        psd = np.abs(np.fft.rfft(detrended)) ** 2
        return k[1:], psd[1:]

    def analyze_spin2_scaling(
        self,
        ks: np.ndarray,
        psd: np.ndarray,
        expected_power: float = 2.0,
    ) -> Dict:
        log_k = np.log10(ks)
        log_psd = np.log10(psd)
        slope, intercept = np.polyfit(log_k, log_psd, 1)
        measured_power = -slope
        residual = abs(measured_power - expected_power)

        return {
            "measured_power": measured_power,
            "expected_power": expected_power,
            "residual": residual,
            "fit_quality": abs(intercept) < 1.0,
        }

    def plot_spin2_tail(
        self,
        ks: np.ndarray,
        psd: np.ndarray,
        title: str = "Spin-2 Tail",
        save_path: Optional[str] = None,
    ):
        plt.figure(figsize=(8, 6))
        plt.loglog(ks, psd, label="PSD", linewidth=2)

        ref_psd = psd[0] * (ks[0] / ks) ** 2
        plt.loglog(ks, ref_psd, "--", label="1/k² reference", alpha=0.7)

        plt.xlabel("k", fontsize=12, fontweight="bold")
        plt.ylabel("Power Spectral Density", fontsize=12, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"🖼️  Spin-2 plot saved to {save_path}")

        plt.close()
