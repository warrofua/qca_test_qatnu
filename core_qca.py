"""
Core exact-diagonalization utilities for QATNU/SRQID simulations.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.linalg import eigh


class Utils:
    """Shared utilities and Pauli matrices."""

    I2 = np.eye(2, dtype=np.float64)
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)

    @staticmethod
    def kronN(ops: List[np.ndarray]) -> np.ndarray:
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out

    @staticmethod
    def local_operator(op: np.ndarray, site: int, N: int) -> np.ndarray:
        ops = [Utils.I2 if k != site else op for k in range(N)]
        return Utils.kronN(ops)

    @staticmethod
    def comm_norm(A: np.ndarray, B: np.ndarray) -> float:
        comm = A @ B - B @ A
        return np.linalg.norm(comm, 2)


class ExactQCA:
    """
    Optimized exact diagonalization engine for QATNU/SRQID Hamiltonian.
    Memory-efficient construction with pre-computed lookup tables.
    """

    def __init__(self, N: int, config: Dict, bond_cutoff: int = 4):
        self.N = N
        self.config = config
        self.bond_cutoff = bond_cutoff

        self.matter_dim = 2**N
        self.bond_dim = bond_cutoff ** (N - 1)
        self.total_dim = self.matter_dim * self.bond_dim

        self.bond_powers = [bond_cutoff**i for i in range(N - 1)][::-1]

        self.Z_lookup = np.ones((self.matter_dim, self.N), dtype=np.int8)
        for i in range(N):
            self.Z_lookup[:, i] = 1 - 2 * ((np.arange(self.matter_dim) >> i) & 1)

        self.H = self._build_hamiltonian()
        self._eig = None

    @staticmethod
    def get_entanglement_profile(qca: "ExactQCA", state: np.ndarray) -> List[float]:
        return [qca.get_bond_dimension(edge, state) for edge in range(qca.N - 1)]

    def decode_bond_config(self, bond_index: int) -> List[int]:
        config, remaining = [], bond_index
        for power in self.bond_powers:
            config.append(remaining // power)
            remaining %= power
        return config[::-1]

    def state_index(self, matter_state: int, bond_config: List[int]) -> int:
        bond_index = sum(val * self.bond_powers[i] for i, val in enumerate(bond_config))
        return matter_state * self.bond_dim + bond_index

    def _build_hamiltonian(self) -> np.ndarray:
        H = np.zeros((self.total_dim, self.total_dim), dtype=np.float64)

        for idx in range(self.total_dim):
            matter_state = idx // self.bond_dim
            bond_config = self.decode_bond_config(idx % self.bond_dim)
            Z_vals = self.Z_lookup[matter_state]

            for i in range(self.N):
                flipped = matter_state ^ (1 << i)
                j = self.state_index(flipped, bond_config)
                H[idx, j] += self.config["omega"] / 2.0

            for edge in range(self.N - 1):
                i, j = edge, edge + 1
                Zi, Zj = Z_vals[i], Z_vals[j]

                H[idx, idx] += self.config["J0"] * Zi * Zj
                H[idx, idx] += self.config["deltaB"] * bond_config[edge]

                d_i = self._calculate_degree(i, bond_config)
                d_j = self._calculate_degree(j, bond_config)
                penalty = (d_i - self.config["k0"]) ** 2 + (d_j - self.config["k0"]) ** 2
                H[idx, idx] += self.config["kappa"] * penalty

                F = 0.5 * (1.0 - Zi * Zj)

                if bond_config[edge] < self.bond_cutoff - 1:
                    new_config = bond_config.copy()
                    new_config[edge] += 1
                    jdx = self.state_index(matter_state, new_config)
                    H[idx, jdx] += self.config["lambda"] * F

                if bond_config[edge] > 0:
                    new_config = bond_config.copy()
                    new_config[edge] -= 1
                    jdx = self.state_index(matter_state, new_config)
                    H[idx, jdx] += self.config["lambda"] * F

        H = (H + H.T) * 0.5
        return H

    def _calculate_degree(self, site: int, bond_config: List[int]) -> int:
        degree = 0
        if site > 0 and bond_config[site - 1] > 0:
            degree += 1
        if site < self.N - 1 and bond_config[site] > 0:
            degree += 1
        return degree

    def diagonalize(self):
        if self._eig is None:
            eigenvalues, eigenvectors = eigh(self.H, overwrite_a=True)
            self._eig = {"eigenvectors": eigenvectors, "eigenvalues": eigenvalues}
        return self._eig

    def get_ground_state(self) -> np.ndarray:
        return self.diagonalize()["eigenvectors"][:, 0].astype(complex)

    def apply_pi2_pulse(self, state: np.ndarray, site: int) -> np.ndarray:
        new_state = np.zeros_like(state, dtype=complex)
        sqrt2 = np.sqrt(2.0)

        for idx in range(self.total_dim):
            if np.abs(state[idx]) < 1e-15:
                continue

            matter_state = idx // self.bond_dim
            bond_config = self.decode_bond_config(idx % self.bond_dim)
            bit = (matter_state >> site) & 1
            flipped_matter = matter_state ^ (1 << site)

            j1 = self.state_index(matter_state, bond_config)
            j2 = self.state_index(flipped_matter, bond_config)
            amplitude = state[idx]

            if bit == 0:
                new_state[j1] += amplitude / sqrt2
                new_state[j2] += 1j * amplitude / sqrt2
            else:
                new_state[j1] += amplitude / sqrt2
                new_state[j2] -= 1j * amplitude / sqrt2

        return new_state

    def evolve_state(self, state: np.ndarray, t: float) -> np.ndarray:
        eig = self.diagonalize()
        eigenvectors = eig["eigenvectors"]
        eigenvalues = eig["eigenvalues"]

        coeffs = eigenvectors.T @ state
        exp_coeffs = coeffs * np.exp(-1j * eigenvalues * t)

        return eigenvectors @ exp_coeffs

    def measure_Z(self, state: np.ndarray, site: int) -> float:
        probabilities = np.abs(state) ** 2
        Z_full = np.tile(self.Z_lookup[:, site], self.bond_dim)
        return float(np.sum(Z_full * probabilities))

    def get_bond_dimension(self, edge: int, state: np.ndarray) -> float:
        avg_n = 0.0
        norm = 0.0

        for idx in range(self.total_dim):
            prob = np.abs(state[idx]) ** 2
            if prob > 1e-15:
                bond_config = self.decode_bond_config(idx % self.bond_dim)
                avg_n += bond_config[edge] * prob
                norm += prob

        return 2.0 ** (avg_n / norm) if norm > 0 else 1.0

    def count_circuit_depth(self, site: int, state: np.ndarray) -> float:
        avg_degree = 0.0
        norm = 0.0

        for idx in range(self.total_dim):
            prob = np.abs(state[idx]) ** 2
            if prob > 1e-15:
                bond_config = self.decode_bond_config(idx % self.bond_dim)
                degree = self._calculate_degree(site, bond_config)
                avg_degree += degree * prob
                norm += prob

        d = avg_degree / norm if norm > 0 else 0
        return np.log2(1.0 + d)

    def commutator_norm(self, obs1: str, site1: int, obs2: str, site2: int, t: float) -> float:
        ops = {"X": Utils.X, "Y": Utils.Y, "Z": Utils.Z}
        A = Utils.local_operator(ops[obs1], site1, self.N)
        B = Utils.local_operator(ops[obs2], site2, self.N)

        A = np.kron(A, np.eye(self.bond_dim))
        B = np.kron(B, np.eye(self.bond_dim))

        eig = self.diagonalize()
        U = eig["eigenvectors"]
        U_dag = U.conj().T
        exp_E = np.diag(np.exp(-1j * eig["eigenvalues"] * t))

        A_t = U @ exp_E.conj().T @ U_dag @ A @ U @ exp_E @ U_dag
        return Utils.comm_norm(A_t, B)
