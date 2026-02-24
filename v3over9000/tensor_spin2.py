"""
Tensor (TT-projected) spin-2 diagnostic from connected bond-bond correlators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from v3over9000.embeddings import GraphEmbedding, build_embedding


@dataclass(frozen=True)
class TTSpin2Spectrum:
    k: np.ndarray
    power: np.ndarray
    n_angles: int
    n_modes: int


class TensorSpin2Analyzer:
    """
    Build a TT-projected tensor spectrum using connected bond covariance.

    The scalar covariance C_ef is lifted to a rank-4 tensor via edge-direction
    basis tensors Q_e, then projected into the transverse-traceless subspace.
    """

    def _bond_covariance(self, qca, state: np.ndarray) -> np.ndarray:
        edge_count = int(getattr(qca, "edge_count", 0))
        if edge_count == 0:
            return np.zeros((0, 0), dtype=float)

        probabilities = np.abs(state) ** 2
        norm = float(probabilities.sum())
        if norm <= 0.0:
            return np.zeros((edge_count, edge_count), dtype=float)
        probabilities = probabilities / norm

        weights_by_bond = probabilities.reshape(qca.matter_dim, qca.bond_dim).sum(axis=0)

        bond_configs = np.zeros((qca.bond_dim, edge_count), dtype=float)
        for bond_idx in range(qca.bond_dim):
            bond_configs[bond_idx] = np.asarray(qca.decode_bond_config(bond_idx), dtype=float)

        mean_occ = weights_by_bond @ bond_configs
        weighted = bond_configs * weights_by_bond[:, None]
        second = bond_configs.T @ weighted
        return second - np.outer(mean_occ, mean_occ)

    @staticmethod
    def _tt_projector(khat: np.ndarray) -> np.ndarray:
        eye = np.eye(3, dtype=float)
        p = eye - np.outer(khat, khat)
        term1 = np.einsum("ac,bd->abcd", p, p)
        term2 = np.einsum("ad,bc->abcd", p, p)
        term3 = np.einsum("ab,cd->abcd", p, p)
        # 3D spatial projector: 2/(d-1) = 1 for d=3.
        return 0.5 * (term1 + term2 - term3)

    @staticmethod
    def _k_grid(edge_mid_xy: np.ndarray, n_modes: int) -> np.ndarray:
        if edge_mid_xy.size == 0:
            return np.zeros(0, dtype=float)
        spread = np.max(edge_mid_xy, axis=0) - np.min(edge_mid_xy, axis=0)
        max_scale = max(float(np.linalg.norm(spread)), 1.0)
        k_min = 2.0 * np.pi / (4.0 * max_scale)
        k_max = np.pi
        n = max(int(n_modes), 6)
        return np.linspace(k_min, k_max, num=n, dtype=float)

    def compute_spectrum(
        self,
        qca,
        state: np.ndarray,
        topology_name: str,
        n_modes: int = 16,
        n_angles: int = 24,
    ) -> TTSpin2Spectrum:
        edge_count = int(getattr(qca, "edge_count", 0))
        if edge_count < 2:
            return TTSpin2Spectrum(
                k=np.zeros(0, dtype=float),
                power=np.zeros(0, dtype=float),
                n_angles=int(n_angles),
                n_modes=int(n_modes),
            )

        embedding: GraphEmbedding = build_embedding(qca.N, topology_name, list(qca.edges))
        covariance = self._bond_covariance(qca, state)

        # Q_e,ab = d_a d_b - delta_ab / 3.
        eye3 = np.eye(3, dtype=float)
        directions = embedding.edge_dir_xyz
        q_tensors = np.einsum("ea,eb->eab", directions, directions) - (eye3[None, :, :] / 3.0)
        mids = embedding.edge_mid_xy

        k_vals = self._k_grid(mids, n_modes=n_modes)
        if k_vals.size == 0:
            return TTSpin2Spectrum(
                k=np.zeros(0, dtype=float),
                power=np.zeros(0, dtype=float),
                n_angles=int(n_angles),
                n_modes=int(n_modes),
            )

        power_vals = np.zeros_like(k_vals, dtype=float)
        angles = np.linspace(0.0, 2.0 * np.pi, num=max(int(n_angles), 6), endpoint=False)

        for idx, kval in enumerate(k_vals):
            powers = []
            for theta in angles:
                kvec = np.array([kval * np.cos(theta), kval * np.sin(theta), 0.0], dtype=float)
                knorm = float(np.linalg.norm(kvec))
                if knorm <= 1e-12:
                    continue
                khat = kvec / knorm

                # phase_e = exp(i k dot r_e), r_e from edge midpoints in XY.
                phase = np.exp(1j * (mids @ kvec[:2]))
                weighted_cov = covariance * np.outer(phase, np.conjugate(phase))
                corr = np.einsum("ef,eab,fcd->abcd", weighted_cov, q_tensors, q_tensors, optimize=True)

                tt = self._tt_projector(khat)
                # Positive semi-definite proxy in TT sector.
                tt_power = np.real(np.einsum("abcd,abcd->", tt, corr))
                powers.append(max(float(tt_power), 1e-20))

            power_vals[idx] = float(np.mean(powers)) if powers else 1e-20

        return TTSpin2Spectrum(
            k=k_vals,
            power=power_vals,
            n_angles=int(len(angles)),
            n_modes=int(len(k_vals)),
        )

    @staticmethod
    def fit_power_law(
        k: np.ndarray,
        power: np.ndarray,
        expected_power: float = 2.0,
        fit_fraction: Tuple[float, float] = (0.2, 0.8),
    ) -> Dict[str, float | bool]:
        valid = np.isfinite(k) & np.isfinite(power) & (k > 0.0) & (power > 0.0)
        if int(np.sum(valid)) < 4:
            return {
                "measured_power": float("nan"),
                "measured_slope": float("nan"),
                "expected_power": float(expected_power),
                "residual": float("nan"),
                "r2": float("nan"),
                "fit_quality": False,
            }

        k_v = k[valid]
        p_v = power[valid]
        order = np.argsort(k_v)
        k_v = k_v[order]
        p_v = p_v[order]

        lo_frac, hi_frac = fit_fraction
        lo_idx = int(max(0, min(len(k_v) - 2, np.floor(lo_frac * len(k_v)))))
        hi_idx = int(max(lo_idx + 2, min(len(k_v), np.ceil(hi_frac * len(k_v)))))
        k_fit = k_v[lo_idx:hi_idx]
        p_fit = p_v[lo_idx:hi_idx]

        x = np.log10(k_fit)
        y = np.log10(p_fit)
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        measured_power = -float(slope)
        residual = abs(measured_power - float(expected_power))
        return {
            "measured_power": measured_power,
            "measured_slope": float(slope),
            "expected_power": float(expected_power),
            "residual": residual,
            "r2": float(r2),
            "fit_quality": bool(np.isfinite(r2) and r2 >= 0.25),
        }

