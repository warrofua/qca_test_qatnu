"""
Core exact-diagonalization utilities for QATNU/SRQID simulations.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix, issparse
from scipy.sparse.linalg import eigsh, expm_multiply


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

    def __init__(
        self,
        N: int,
        config: Dict,
        bond_cutoff: int = 4,
        edges: Optional[List[Tuple[int, int]]] = None,
        hamiltonian_mode: str = "dense",
    ):
        self.N = N
        self.config = config
        self.bond_cutoff = bond_cutoff
        self.edges = edges if edges is not None else [(i, i + 1) for i in range(max(N - 1, 0))]
        self.hamiltonian_mode = (hamiltonian_mode or "dense").lower()
        if self.hamiltonian_mode not in {"dense", "sparse"}:
            raise ValueError(f"Unsupported hamiltonian_mode '{hamiltonian_mode}'")
        self.edge_count = len(self.edges)

        self.matter_dim = 2**N
        if self.edge_count > 0:
            self.bond_dim = bond_cutoff ** self.edge_count
            self.bond_powers = [bond_cutoff**i for i in range(self.edge_count)][::-1]
        else:
            self.bond_dim = 1
            self.bond_powers = [1]
        self.total_dim = self.matter_dim * self.bond_dim

        self.Z_lookup = np.ones((self.matter_dim, self.N), dtype=np.int8)
        for i in range(N):
            self.Z_lookup[:, i] = 1 - 2 * ((np.arange(self.matter_dim) >> i) & 1)

        self.incident_edges = [[] for _ in range(self.N)]
        for edge_idx, (u, v) in enumerate(self.edges):
            self.incident_edges[u].append(edge_idx)
            self.incident_edges[v].append(edge_idx)

        self.H = self._build_hamiltonian(mode=self.hamiltonian_mode)
        self._eig = None

    @staticmethod
    def get_entanglement_profile(qca: "ExactQCA", state: np.ndarray) -> List[float]:
        return [qca.get_bond_dimension(edge_idx, state) for edge_idx in range(qca.edge_count)]

    def decode_bond_config(self, bond_index: int) -> List[int]:
        config, remaining = [], bond_index
        for power in self.bond_powers:
            config.append(remaining // power)
            remaining %= power
        if self.edge_count == 0:
            return []
        return config

    def state_index(self, matter_state: int, bond_config: List[int]) -> int:
        if self.edge_count == 0:
            bond_index = 0
        else:
            bond_index = sum(
                val * self.bond_powers[i] for i, val in enumerate(bond_config[: self.edge_count])
            )
        return matter_state * self.bond_dim + bond_index

    def _hamiltonian_entries(self) -> Iterable[Tuple[int, int, float]]:
        for idx in range(self.total_dim):
            matter_state = idx // self.bond_dim
            bond_config = self.decode_bond_config(idx % self.bond_dim)
            Z_vals = self.Z_lookup[matter_state]
            diag_val = 0.0

            for i in range(self.N):
                flipped = matter_state ^ (1 << i)
                j = self.state_index(flipped, bond_config)
                yield idx, j, float(self.config["omega"]) / 2.0

            for edge_idx, (site_i, site_j) in enumerate(self.edges):
                Zi, Zj = Z_vals[site_i], Z_vals[site_j]

                diag_val += float(self.config["J0"]) * float(Zi * Zj)
                if self.edge_count > 0:
                    diag_val += float(self.config["deltaB"]) * float(bond_config[edge_idx])

                d_i = self._calculate_degree(site_i, bond_config)
                d_j = self._calculate_degree(site_j, bond_config)
                k0 = float(self.config["k0"])
                penalty = (float(d_i) - k0) ** 2 + (float(d_j) - k0) ** 2
                diag_val += float(self.config["kappa"]) * penalty

                F = 0.5 * (1.0 - Zi * Zj)
                lam_f = float(self.config["lambda"]) * float(F)

                if self.edge_count > 0:
                    if bond_config[edge_idx] < self.bond_cutoff - 1:
                        new_config = bond_config.copy()
                        new_config[edge_idx] += 1
                        jdx = self.state_index(matter_state, new_config)
                        yield idx, jdx, lam_f

                    if bond_config[edge_idx] > 0:
                        new_config = bond_config.copy()
                        new_config[edge_idx] -= 1
                        jdx = self.state_index(matter_state, new_config)
                        yield idx, jdx, lam_f

            yield idx, idx, diag_val

    def _build_hamiltonian(self, mode: str = "dense"):
        mode = (mode or "dense").lower()
        if mode == "dense":
            H = np.zeros((self.total_dim, self.total_dim), dtype=np.float64)
            for row, col, value in self._hamiltonian_entries():
                H[row, col] += value
            H = (H + H.T) * 0.5
            return H

        if mode == "sparse":
            rows: List[int] = []
            cols: List[int] = []
            data: List[float] = []
            for row, col, value in self._hamiltonian_entries():
                rows.append(row)
                cols.append(col)
                data.append(value)

            H = coo_matrix(
                (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
                shape=(self.total_dim, self.total_dim),
            ).tocsr()
            H = (H + H.T) * 0.5
            H.sum_duplicates()
            return H.tocsr()

        raise ValueError(f"Unsupported hamiltonian build mode '{mode}'")

    def _calculate_degree(self, site: int, bond_config: List[int]) -> int:
        degree = 0
        for edge_idx in self.incident_edges[site]:
            if bond_config and bond_config[edge_idx] > 0:
                degree += 1
        return degree

    def diagonalize(self):
        if self._eig is None:
            if issparse(self.H):
                dense_h = self.H.toarray()
                eigenvalues, eigenvectors = eigh(dense_h, overwrite_a=True)
            else:
                eigenvalues, eigenvectors = eigh(self.H, overwrite_a=True)
            self._eig = {"eigenvectors": eigenvectors, "eigenvalues": eigenvalues}
        return self._eig

    def get_ground_state(
        self,
        method: str = "auto",
        tol: float = 1e-9,
        maxiter: Optional[int] = None,
    ) -> np.ndarray:
        mode = (method or "auto").lower()
        if mode not in {"auto", "dense", "iterative", "krylov"}:
            raise ValueError(f"Unsupported ground-state method '{method}'")

        if mode == "auto":
            mode = "iterative" if issparse(self.H) else "dense"

        if mode == "dense":
            return self.diagonalize()["eigenvectors"][:, 0].astype(np.complex128)

        op = self.H if issparse(self.H) else csr_matrix(self.H)
        evals, evecs = eigsh(op, k=1, which="SA", tol=float(tol), maxiter=maxiter)
        ground = evecs[:, int(np.argmin(evals))].astype(np.complex128)
        # Fix global phase to make runs deterministic.
        anchor = int(np.argmax(np.abs(ground)))
        if np.abs(ground[anchor]) > 0.0:
            ground *= np.exp(-1j * np.angle(ground[anchor]))
        return ground

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

    def evolve_state(self, state: np.ndarray, t: float, method: str = "auto") -> np.ndarray:
        mode = (method or "auto").lower()
        if mode not in {"auto", "dense", "iterative", "krylov"}:
            raise ValueError(f"Unsupported evolution method '{method}'")
        if mode == "auto":
            mode = "krylov" if issparse(self.H) else "dense"

        psi = np.asarray(state, dtype=np.complex128)
        if mode == "dense":
            eig = self.diagonalize()
            eigenvectors = eig["eigenvectors"]
            eigenvalues = eig["eigenvalues"]

            coeffs = eigenvectors.T @ psi
            exp_coeffs = coeffs * np.exp(-1j * eigenvalues * float(t))
            return eigenvectors @ exp_coeffs

        op = self.H if issparse(self.H) else csr_matrix(self.H)
        return np.asarray(expm_multiply((-1j * float(t)) * op, psi), dtype=np.complex128)

    def evolve_states(
        self,
        state: np.ndarray,
        times: np.ndarray,
        method: str = "auto",
    ) -> np.ndarray:
        """
        Return matrix with shape [dim, T].
        """
        t_arr = np.asarray(times, dtype=float)
        if t_arr.ndim != 1:
            raise ValueError("times must be a 1D array")
        if t_arr.size == 0:
            return np.zeros((self.total_dim, 0), dtype=np.complex128)

        mode = (method or "auto").lower()
        if mode not in {"auto", "dense", "iterative", "krylov"}:
            raise ValueError(f"Unsupported evolution method '{method}'")
        if mode == "auto":
            mode = "krylov" if issparse(self.H) else "dense"

        psi = np.asarray(state, dtype=np.complex128)
        if mode == "dense":
            eig = self.diagonalize()
            eigenvectors = eig["eigenvectors"]
            eigenvalues = eig["eigenvalues"]
            coeffs = eigenvectors.T @ psi
            phases = np.exp(-1j * np.outer(eigenvalues, t_arr))
            return eigenvectors @ (coeffs[:, None] * phases)

        op = self.H if issparse(self.H) else csr_matrix(self.H)
        if t_arr.size == 1:
            single = np.asarray(expm_multiply((-1j * float(t_arr[0])) * op, psi), dtype=np.complex128)
            return single[:, None]

        dts = np.diff(t_arr)
        dt0 = float(dts[0])
        uniform = np.allclose(dts, dt0, rtol=0.0, atol=1e-12 * max(1.0, abs(dt0)))
        if uniform:
            states_t = expm_multiply(
                -1j * op,
                psi,
                start=float(t_arr[0]),
                stop=float(t_arr[-1]),
                num=int(t_arr.size),
                endpoint=True,
            )
            states_t = np.asarray(states_t, dtype=np.complex128)
            if states_t.ndim == 1:
                states_t = states_t[None, :]
        else:
            states_t = np.vstack(
                [
                    np.asarray(expm_multiply((-1j * float(t)) * op, psi), dtype=np.complex128)
                    for t in t_arr
                ]
            )
        return states_t.T

    def measure_Z(self, state: np.ndarray, site: int) -> float:
        probabilities = np.abs(state) ** 2
        Z_full = np.repeat(self.Z_lookup[:, site], self.bond_dim)
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
