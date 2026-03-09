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

    @staticmethod
    def site_lambda_profile(qca, state: np.ndarray) -> np.ndarray:
        return np.asarray(
            [float(qca.count_circuit_depth(site, state)) for site in range(int(getattr(qca, "N", 0)))],
            dtype=float,
        )

    @staticmethod
    def vertex_graph_distances(
        n_sites: int,
        edges: List[Tuple[int, int]],
        source_site: int,
    ) -> np.ndarray:
        if n_sites <= 0:
            return np.zeros(0, dtype=int)
        adjacency: List[List[int]] = [[] for _ in range(int(n_sites))]
        for u, v in edges:
            adjacency[int(u)].append(int(v))
            adjacency[int(v)].append(int(u))

        inf = int(n_sites) + 1
        dist = np.full(int(n_sites), inf, dtype=int)
        src = int(source_site)
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for nxt in adjacency[node]:
                if dist[nxt] > dist[node] + 1:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        return dist

    @staticmethod
    def shell_average(values: np.ndarray, shells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        shells = np.asarray(shells, dtype=int)
        if values.size == 0 or shells.size == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        valid = np.isfinite(values) & (shells >= 0)
        if not np.any(valid):
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        shell_ids = np.unique(shells[valid])
        means = np.zeros(shell_ids.size, dtype=float)
        for idx, shell in enumerate(shell_ids):
            mask = valid & (shells == shell)
            means[idx] = float(np.mean(values[mask])) if np.any(mask) else 0.0
        return shell_ids.astype(int), means

    @staticmethod
    def edge_shells_from_vertex_shells(
        edges: List[Tuple[int, int]],
        vertex_shells: np.ndarray,
    ) -> np.ndarray:
        shells = np.asarray(vertex_shells, dtype=int)
        if len(edges) == 0 or shells.size == 0:
            return np.zeros(0, dtype=int)
        out = np.zeros(len(edges), dtype=int)
        for idx, (u, v) in enumerate(edges):
            out[idx] = int(min(shells[int(u)], shells[int(v)]))
        return out

    @staticmethod
    def shell_pair_average_covariance(
        covariance: np.ndarray,
        edge_shells: np.ndarray,
    ) -> np.ndarray:
        covariance = np.asarray(covariance, dtype=float)
        shells = np.asarray(edge_shells, dtype=int)
        if covariance.size == 0 or shells.size == 0:
            return np.zeros_like(covariance, dtype=float)
        if covariance.shape != (shells.size, shells.size):
            raise ValueError("covariance shape must match edge_shells length.")

        unique_shells = np.unique(shells)
        shell_pair_means: Dict[Tuple[int, int], float] = {}
        for a in unique_shells:
            mask_a = shells == a
            for b in unique_shells:
                mask_b = shells == b
                vals = covariance[np.ix_(mask_a, mask_b)]
                shell_pair_means[(int(a), int(b))] = float(np.mean(vals)) if vals.size else 0.0

        bg = np.zeros_like(covariance, dtype=float)
        for i, a in enumerate(shells):
            for j, b in enumerate(shells):
                bg[i, j] = shell_pair_means[(int(a), int(b))]

        return 0.5 * (bg + bg.T)

    @staticmethod
    def _line_graph_distances(edges: List[Tuple[int, int]]) -> np.ndarray:
        """Shortest-path distances between edge nodes in the line graph."""
        edge_count = len(edges)
        if edge_count == 0:
            return np.zeros((0, 0), dtype=int)
        if edge_count == 1:
            return np.zeros((1, 1), dtype=int)

        adjacency: List[List[int]] = [[] for _ in range(edge_count)]
        for i in range(edge_count):
            u1, v1 = edges[i]
            for j in range(i + 1, edge_count):
                u2, v2 = edges[j]
                if u1 in (u2, v2) or v1 in (u2, v2):
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        inf = edge_count + 1
        dist = np.full((edge_count, edge_count), inf, dtype=int)
        for src in range(edge_count):
            dist[src, src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                for nxt in adjacency[node]:
                    if dist[src, nxt] > dist[src, node] + 1:
                        dist[src, nxt] = dist[src, node] + 1
                        queue.append(nxt)

        return dist

    @staticmethod
    def line_graph_adjacency(edges: List[Tuple[int, int]]) -> np.ndarray:
        edge_count = len(edges)
        if edge_count == 0:
            return np.zeros((0, 0), dtype=float)
        adjacency = np.zeros((edge_count, edge_count), dtype=float)
        for i in range(edge_count):
            u1, v1 = edges[i]
            for j in range(i + 1, edge_count):
                u2, v2 = edges[j]
                if u1 in (u2, v2) or v1 in (u2, v2):
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0
        return adjacency

    def line_graph_low_mode_projector(
        self,
        edges: List[Tuple[int, int]],
        n_modes: int,
    ) -> np.ndarray:
        adjacency = self.line_graph_adjacency(edges)
        edge_count = adjacency.shape[0]
        if edge_count == 0:
            return np.zeros((0, 0), dtype=float)
        if edge_count == 1:
            return np.ones((1, 1), dtype=float)

        degree = np.diag(np.sum(adjacency, axis=1))
        lap = degree - adjacency
        evals, evecs = np.linalg.eigh(lap)
        order = np.argsort(evals)
        keep = max(1, min(int(n_modes), edge_count))
        basis = evecs[:, order[:keep]]
        projector = basis @ basis.T
        return 0.5 * (projector + projector.T)

    @staticmethod
    def _bond_occupation_moments(qca, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ⟨n_e⟩ and connected covariance C_ee' from bond occupations.
        """
        edge_count = int(getattr(qca, "edge_count", 0))
        if edge_count == 0:
            return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)

        probabilities = np.abs(state) ** 2
        norm = float(probabilities.sum())
        if norm <= 0.0:
            return np.zeros(edge_count, dtype=float), np.zeros((edge_count, edge_count), dtype=float)
        probabilities = probabilities / norm

        # Marginalize matter subspace to obtain bond-sector probabilities.
        weights_by_bond = probabilities.reshape(qca.matter_dim, qca.bond_dim).sum(axis=0)

        bond_configs = np.zeros((qca.bond_dim, edge_count), dtype=float)
        for bond_idx in range(qca.bond_dim):
            bond_configs[bond_idx] = np.array(qca.decode_bond_config(bond_idx), dtype=float)

        mean_occ = weights_by_bond @ bond_configs
        weighted = bond_configs * weights_by_bond[:, None]
        second = bond_configs.T @ weighted
        covariance = second - np.outer(mean_occ, mean_occ)
        return mean_occ, covariance

    def bond_occupation_moments(self, qca, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._bond_occupation_moments(qca, state)

    @staticmethod
    def _radialize_covariance(covariance: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """
        Convert edge-edge covariance matrix into a distance-binned profile C(r).
        """
        if covariance.size == 0:
            return np.zeros(0, dtype=float)

        finite_mask = distances < (distances.shape[0] + 1)
        if not np.any(finite_mask):
            return np.zeros(0, dtype=float)

        max_dist = int(np.max(distances[finite_mask]))
        radial = np.zeros(max_dist + 1, dtype=float)
        for r in range(max_dist + 1):
            mask = distances == r
            vals = covariance[mask]
            radial[r] = float(np.mean(vals)) if vals.size else 0.0
        return radial

    @staticmethod
    def _coarse_grain_radial_profile(radial: np.ndarray, virtual_size: int) -> np.ndarray:
        """
        Interpolate sparse C(r) onto a larger translation-invariant virtual lattice.
        """
        if virtual_size < 8:
            virtual_size = 8
        half_size = virtual_size // 2 + 1
        if radial.size == 0:
            return np.zeros(virtual_size, dtype=float)

        src_x = np.arange(radial.size, dtype=float)
        tgt_x = np.linspace(0.0, max(float(radial.size - 1), 0.0), half_size)
        corr_half = np.interp(tgt_x, src_x, radial)

        kernel = np.zeros(virtual_size, dtype=float)
        kernel[:half_size] = corr_half
        for r in range(1, half_size):
            kernel[-r] = corr_half[r]
        return kernel

    def spin2_from_bond_correlators(
        self,
        qca,
        state: np.ndarray,
        virtual_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a spin-2 proxy spectrum from connected bond-bond correlators.

        This uses covariance of bond occupation numbers n_e and distance
        coarse-graining on the line graph to produce a PSD-like tail.
        """
        edge_count = int(getattr(qca, "edge_count", 0))
        if edge_count < 2:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        _, covariance = self._bond_occupation_moments(qca, state)
        distances = self._line_graph_distances(list(getattr(qca, "edges", [])))
        radial = self._radialize_covariance(covariance, distances)
        kernel = self._coarse_grain_radial_profile(radial, int(virtual_size))

        kernel = kernel - np.mean(kernel)
        psd = np.abs(np.fft.rfft(kernel)) ** 2
        k = np.fft.rfftfreq(kernel.size) * 2 * np.pi

        return k[1:], psd[1:]

    def analyze_spin2_scaling(
        self,
        ks: np.ndarray,
        psd: np.ndarray,
        expected_power: float = 2.0,
    ) -> Dict:
        valid = np.isfinite(ks) & np.isfinite(psd) & (ks > 0.0) & (psd > 0.0)
        if int(np.sum(valid)) < 3:
            return {
                "measured_power": float("nan"),
                "expected_power": expected_power,
                "residual": float("nan"),
                "fit_quality": False,
            }

        log_k = np.log10(ks[valid])
        log_psd = np.log10(psd[valid])
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
