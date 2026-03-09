"""
Single-point experiments combining exact QCA and mean-field comparators.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from core_qca import ExactQCA
from mean_field import QuantumChain


def _apply_hotspot_protocol(
    *,
    N: int,
    base_config: Dict,
    bond_cutoff: int,
    edges: Optional[List[Tuple[int, int]]],
    ground_state: np.ndarray,
    hotspot_multiplier: float,
    hotspot_time: float,
    hotspot_edge_weights: Optional[Sequence[float]],
    hotspot_stages: Optional[Sequence[Dict]],
) -> np.ndarray:
    if hotspot_stages:
        state = ground_state
        for stage in hotspot_stages:
            stage_config = base_config.copy()
            stage_config["lambda"] = float(base_config["lambda"]) * float(stage.get("multiplier", hotspot_multiplier))
            stage_weights = stage.get("edge_weights")
            if stage_weights is not None:
                stage_config["lambda_edge_weights"] = list(stage_weights)
            qca_hotspot = ExactQCA(
                N,
                stage_config,
                bond_cutoff=bond_cutoff,
                edges=edges,
            )
            state = qca_hotspot.evolve_state(state, float(stage.get("time", hotspot_time)))
        return state

    hotspot_config = base_config.copy()
    hotspot_config["lambda"] = float(base_config["lambda"]) * hotspot_multiplier
    if hotspot_edge_weights is not None:
        hotspot_config["lambda_edge_weights"] = list(hotspot_edge_weights)
    qca_hotspot = ExactQCA(
        N,
        hotspot_config,
        bond_cutoff=bond_cutoff,
        edges=edges,
    )
    return qca_hotspot.evolve_state(ground_state, hotspot_time)


class QCATester:
    """Unified tester combining exact, mean-field, and validation."""

    def __init__(
        self,
        N: int = 4,
        alpha: float = 0.8,
        bond_cutoff: int = 4,
        hotspot_multiplier: float = 3.0,
        hotspot_time: float = 1.0,
        hotspot_edge_weights: Optional[Sequence[float]] = None,
        hotspot_stages: Optional[Sequence[Dict]] = None,
        gamma_corr: float = 0.0,
        gamma_corr_diag: float = 0.0,
        readout_gamma_corr: float = 0.0,
        readout_gamma_corr_diag: float = 0.0,
        edges: Optional[List[Tuple[int, int]]] = None,
        probes: Optional[Tuple[int, int]] = None,
    ):
        self.edges = edges
        self.probes = probes if probes is not None else (0, 1 if N > 1 else 0)
        self.parameters = {
            "N": N,
            "omega": 1.0,
            "alpha": alpha,
            "deltaB": 5.0,
            "lambda": 0.3,
            "hotspotMultiplier": hotspot_multiplier,
            "hotspotTime": hotspot_time,
            "hotspotEdgeWeights": list(hotspot_edge_weights) if hotspot_edge_weights is not None else None,
            "hotspotStages": list(hotspot_stages) if hotspot_stages is not None else None,
            "gamma_corr": gamma_corr,
            "gamma_corr_diag": gamma_corr_diag,
            "readoutGammaCorr": readout_gamma_corr,
            "readoutGammaCorrDiag": readout_gamma_corr_diag,
            "kappa": 0.1,
            "k0": 4,
            "bondCutoff": bond_cutoff,
            "J0": 0.01,
            "gamma": 0.0,
            "probeOut": self.probes[0],
            "probeIn": self.probes[1],
            "tMax": 20.0,
            "edges": self.edges,
        }
        self._is_path = self._edges_form_path()

    def _edges_form_path(self) -> bool:
        if not self.edges:
            return True
        expected = [(i, i + 1) for i in range(max(self.parameters["N"] - 1, 0))]
        return self.edges == expected

    def run_exact_experiment(self, config: Dict = None):
        if config is None:
            config = self.parameters

        qca = ExactQCA(
            config["N"],
            config,
            bond_cutoff=config["bondCutoff"],
            edges=config.get("edges"),
        )
        readout_config = dict(config)
        readout_config["gamma_corr"] = config.get(
            "readoutGammaCorr",
            self.parameters.get("readoutGammaCorr", config.get("gamma_corr", 0.0)),
        )
        readout_config["gamma_corr_diag"] = config.get(
            "readoutGammaCorrDiag",
            self.parameters.get("readoutGammaCorrDiag", config.get("gamma_corr_diag", 0.0)),
        )
        qca_readout = ExactQCA(
            config["N"],
            readout_config,
            bond_cutoff=config["bondCutoff"],
            edges=config.get("edges"),
        )

        hotspot_multiplier = config.get(
            "hotspotMultiplier",
            self.parameters.get("hotspotMultiplier", 3.0),
        )
        hotspot_time = config.get(
            "hotspotTime",
            self.parameters.get("hotspotTime", 1.0),
        )
        hotspot_edge_weights = config.get(
            "hotspotEdgeWeights",
            self.parameters.get("hotspotEdgeWeights"),
        )
        hotspot_stages = config.get(
            "hotspotStages",
            self.parameters.get("hotspotStages"),
        )
        ground_state = qca.get_ground_state()
        frustrated_state = _apply_hotspot_protocol(
            N=config["N"],
            base_config=config,
            bond_cutoff=config["bondCutoff"],
            edges=config.get("edges"),
            ground_state=ground_state,
            hotspot_multiplier=hotspot_multiplier,
            hotspot_time=hotspot_time,
            hotspot_edge_weights=hotspot_edge_weights,
            hotspot_stages=hotspot_stages,
        )

        t_grid = np.linspace(0, config["tMax"], 100)

        data_out: List[Dict] = []
        data_in: List[Dict] = []

        for t in t_grid:
            for site, data_list in [(config["probeOut"], data_out), (config["probeIn"], data_in)]:
                psi = qca_readout.apply_pi2_pulse(ground_state, site)
                psi_t = qca_readout.evolve_state(psi, t)
                Z_t = qca_readout.measure_Z(psi_t, site)
                data_list.append({"t": t, "Z": Z_t})

        bond_dims = []
        edge_total = len(qca.edges)
        for edge_idx in range(edge_total):
            chi = qca.get_bond_dimension(edge_idx, frustrated_state)
            bond_dims.append({"edge": edge_idx, "chi": chi})

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
            "frustrated_state": frustrated_state,
            "qca": qca_readout,
            "qca_base": qca,
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

    @staticmethod
    def _lambda_from_depth_metric(depth_metric: float) -> float:
        """
        The exact engine already reports the log-depth proxy Λ = log2(1 + degree).
        Keep this as identity to avoid accidental re-transformations.
        """
        return float(depth_metric)

    def validate_postulate(self, exact_results: Dict) -> Dict:
        p = exact_results["config"]

        depth_out = float(exact_results["depthOut"])
        depth_in = float(exact_results["depthIn"])
        measured_lambda_out = self._lambda_from_depth_metric(depth_out)
        measured_lambda_in = self._lambda_from_depth_metric(depth_in)

        predicted_out = p["omega"] / (1.0 + p["alpha"] * measured_lambda_out)
        predicted_in = p["omega"] / (1.0 + p["alpha"] * measured_lambda_in)

        residual = abs(
            (exact_results["omega_in"] / exact_results["omega_out"])
            - (predicted_in / predicted_out)
        )

        return {
            "measuredDepthOut": depth_out,
            "measuredDepthIn": depth_in,
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
        mean_field = None
        if self._is_path:
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
