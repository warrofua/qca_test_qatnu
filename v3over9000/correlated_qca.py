"""
Correlated-promotion extension of ExactQCA.

This introduces a pair-promotion channel across incident edges to test whether
explicit bond-bond coupling improves long-wavelength tensor behavior.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core_qca import ExactQCA


class CorrelatedPromotionQCA(ExactQCA):
    """
    ExactQCA plus correlated two-edge promotion/demotion terms.

    New config fields:
      - gamma_corr: amplitude of correlated edge-pair ladder terms
      - gamma_corr_diag: amplitude of diagonal n_e * n_f penalty/reward
    """

    def __init__(
        self,
        N: int,
        config: Dict,
        bond_cutoff: int = 4,
        edges: Optional[List[Tuple[int, int]]] = None,
    ):
        self.gamma_corr = float(config.get("gamma_corr", 0.0))
        self.gamma_corr_diag = float(config.get("gamma_corr_diag", 0.0))
        self._correlated_pairs: Optional[List[Tuple[int, int]]] = None
        super().__init__(N=N, config=config, bond_cutoff=bond_cutoff, edges=edges)

    def _get_correlated_pairs(self) -> List[Tuple[int, int]]:
        if self._correlated_pairs is not None:
            return self._correlated_pairs

        pairs: List[Tuple[int, int]] = []
        for e1, (u1, v1) in enumerate(self.edges):
            for e2 in range(e1 + 1, len(self.edges)):
                u2, v2 = self.edges[e2]
                if u1 in (u2, v2) or v1 in (u2, v2):
                    pairs.append((e1, e2))
        self._correlated_pairs = pairs
        return pairs

    def _build_hamiltonian(self):
        H = super()._build_hamiltonian()
        if self.edge_count == 0:
            return H
        if self.gamma_corr == 0.0 and self.gamma_corr_diag == 0.0:
            return H

        pairs = self._get_correlated_pairs()
        if not pairs:
            return H

        for idx in range(self.total_dim):
            matter_state = idx // self.bond_dim
            bond_config = self.decode_bond_config(idx % self.bond_dim)
            z_vals = self.Z_lookup[matter_state]

            for e1, e2 in pairs:
                i1, j1 = self.edges[e1]
                i2, j2 = self.edges[e2]
                f1 = 0.5 * (1.0 - z_vals[i1] * z_vals[j1])
                f2 = 0.5 * (1.0 - z_vals[i2] * z_vals[j2])
                amplitude = self.gamma_corr * f1 * f2

                if self.gamma_corr_diag != 0.0:
                    H[idx, idx] += self.gamma_corr_diag * bond_config[e1] * bond_config[e2]

                if amplitude == 0.0:
                    continue

                # Correlated double-promotion.
                if (
                    bond_config[e1] < self.bond_cutoff - 1
                    and bond_config[e2] < self.bond_cutoff - 1
                ):
                    new_cfg = bond_config.copy()
                    new_cfg[e1] += 1
                    new_cfg[e2] += 1
                    jdx = self.state_index(matter_state, new_cfg)
                    H[idx, jdx] += amplitude

                # Correlated double-demotion.
                if bond_config[e1] > 0 and bond_config[e2] > 0:
                    new_cfg = bond_config.copy()
                    new_cfg[e1] -= 1
                    new_cfg[e2] -= 1
                    jdx = self.state_index(matter_state, new_cfg)
                    H[idx, jdx] += amplitude

        return (H + H.T) * 0.5

