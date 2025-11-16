"""
Single-point experiments combining exact QCA and mean-field comparators.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from core_qca import ExactQCA
from mean_field import QuantumChain


class QCATester:
    """Unified tester combining exact, mean-field, and validation."""

    def __init__(self, N: int = 4, alpha: float = 0.8, bond_cutoff: int = 4):
        self.parameters = {
            "N": N,
            "omega": 1.0,
            "alpha": alpha,
            "deltaB": 5.0,
            "lambda": 0.3,
            "kappa": 0.1,
            "k0": 4,
            "bondCutoff": bond_cutoff,
            "J0": 0.01,
            "gamma": 0.0,
            "probeOut": 0,
            "probeIn": 1,
            "tMax": 20.0,
        }

    def run_exact_experiment(self, config: Dict = None):
        if config is None:
            config = self.parameters

        qca = ExactQCA(config["N"], config, bond_cutoff=config["bondCutoff"])

        hotspot_config = config.copy()
        hotspot_config["lambda"] = config["lambda"] * 3.0
        qca_hotspot = ExactQCA(config["N"], hotspot_config, bond_cutoff=config["bondCutoff"])
        ground_state = qca.get_ground_state()
        frustrated_state = qca_hotspot.evolve_state(ground_state, 1.0)

        t_grid = np.linspace(0, config["tMax"], 100)

        data_out: List[Dict] = []
        data_in: List[Dict] = []

        for t in t_grid:
            for site, data_list in [(config["probeOut"], data_out), (config["probeIn"], data_in)]:
                psi = qca.apply_pi2_pulse(ground_state, site)
                psi_t = qca.evolve_state(psi, t)
                Z_t = qca.measure_Z(psi_t, site)
                data_list.append({"t": t, "Z": Z_t})

        bond_dims = []
        for edge in range(config["N"] - 1):
            chi = qca.get_bond_dimension(edge, frustrated_state)
            bond_dims.append({"edge": edge, "chi": chi})

        depth_out = qca.count_circuit_depth(config["probeOut"], frustrated_state)
        depth_in = qca.count_circuit_depth(config["probeIn"], frustrated_state)

        omega_out = self._fit_frequency(data_out)
        omega_in = self._fit_frequency(data_in)

        return {
            "dataOut": data_out,
            "dataIn": data_in,
            "omega_out": omega_out,
            "omega_in": omega_in,
            "bondDims": bond_dims,
            "depthOut": depth_out,
            "depthIn": depth_in,
            "config": config,
        }

    def run_mean_field_comparison(self, exact_results: Dict) -> Dict:
        p = exact_results["config"]
        chi_profile = [d["chi"] for d in exact_results["bondDims"]]

        chain = QuantumChain(
            N=p["N"],
            omega=p["omega"],
            alpha=p["alpha"],
            chi_profile=chi_profile,
            J0=p["J0"],
            gamma=p["gamma"],
        )

        t_grid = np.linspace(0, p["tMax"], 100)

        mf_out = [{"t": t, "Z": chain.evolve_ramsey_single_ion(p["probeOut"], t)} for t in t_grid]
        mf_in = [{"t": t, "Z": chain.evolve_ramsey_single_ion(p["probeIn"], t)} for t in t_grid]

        return {
            "dataOut": mf_out,
            "dataIn": mf_in,
            "omega_out": self._fit_frequency(mf_out),
            "omega_in": self._fit_frequency(mf_in),
        }

    @staticmethod
    def _fit_frequency(data: List[Dict]) -> float:
        if len(data) < 10:
            return 0.0

        times = np.array([d["t"] for d in data])
        signal = np.array([d["Z"] for d in data])
        signal_detrended = signal - np.mean(signal)

        fft = np.fft.fft(signal_detrended)
        freqs = np.fft.fftfreq(len(times), d=times[1] - times[0])

        pos_mask = freqs > 0
        peak_idx = np.argmax(np.abs(fft[pos_mask]))
        return 2 * np.pi * freqs[pos_mask][peak_idx]

    def validate_postulate(self, exact_results: Dict) -> Dict:
        p = exact_results["config"]

        measured_lambda_out = np.log2(1.0 + 2.0 ** exact_results["depthOut"] - 1.0)
        measured_lambda_in = np.log2(1.0 + 2.0 ** exact_results["depthIn"] - 1.0)

        predicted_out = p["omega"] / (1.0 + p["alpha"] * measured_lambda_out)
        predicted_in = p["omega"] / (1.0 + p["alpha"] * measured_lambda_in)

        residual = abs(
            (exact_results["omega_in"] / exact_results["omega_out"])
            - (predicted_in / predicted_out)
        )

        return {
            "measuredLambdaOut": measured_lambda_out,
            "measuredLambdaIn": measured_lambda_in,
            "predictedFreqOut": predicted_out,
            "predictedFreqIn": predicted_in,
            "residual": residual,
            "measuredFreqOut": exact_results["omega_out"],
            "measuredFreqIn": exact_results["omega_in"],
        }

    def run_full_validation(self, config: Dict = None):
        exact = self.run_exact_experiment(config)
        mean_field = self.run_mean_field_comparison(exact)
        validation = self.validate_postulate(exact)
        return exact, mean_field, validation


def plot_ramsey_overlay(
    exact_results: Dict,
    mean_field_results: Dict,
    save_path: str,
    title_prefix: str = "Ramsey",
):
    """Create exact vs mean-field overlays for inner/out probes."""
    plt.figure(figsize=(10, 6))
    for label, exact_data, mf_data in [
        ("Outer probe", exact_results["dataOut"], mean_field_results["dataOut"]),
        ("Inner probe", exact_results["dataIn"], mean_field_results["dataIn"]),
    ]:
        t_exact = [d["t"] for d in exact_data]
        z_exact = [d["Z"] for d in exact_data]
        t_mf = [d["t"] for d in mf_data]
        z_mf = [d["Z"] for d in mf_data]
        plt.plot(t_exact, z_exact, label=f"{label} – exact")
        plt.plot(t_mf, z_mf, "--", label=f"{label} – mean-field")

    plt.xlabel("Time (arb.)")
    plt.ylabel("⟨Z⟩")
    lam = exact_results["config"]["lambda"]
    plt.title(f"{title_prefix} Overlay (λ={lam:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"🖼️  Mean-field overlay saved to {save_path}")
