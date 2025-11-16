"""
SRQID structural validation utilities.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from core_qca import ExactQCA, Utils


class SRQIDValidators:
    """SRQID structural validation tests."""

    @staticmethod
    def extract_lr_velocity(qca: ExactQCA, threshold: float = 1e-3, max_dist: int = None) -> Tuple[float, Tuple]:
        if max_dist is None:
            max_dist = qca.N // 2

        distances = range(1, max_dist + 1)
        times = np.linspace(0, 5.0, 50)
        arrival_times = []

        print(f"Computing LR velocity for N={qca.N}...")
        for r in distances:
            for t in times:
                norm = qca.commutator_norm("Z", 0, "Z", r, t)
                if norm > threshold:
                    arrival_times.append((r, t))
                    print(f"  Distance {r}: t_arrival = {t:.3f} (norm={norm:.6f})")
                    break

        if len(arrival_times) >= 2:
            r_vals, t_vals = zip(*arrival_times)
            v_LR, intercept = np.polyfit(t_vals, r_vals, 1)
            print(f"✓ Extracted v_LR = {v_LR:.3f} (intercept={intercept:.3f})")
            return v_LR, (r_vals, t_vals)

        print("✗ Could not extract velocity (insufficient data)")
        return None, None

    @staticmethod
    def no_signalling_quench(qca: ExactQCA, site: int = 0, far_site: int = None, t_max: float = 3.0) -> float:
        if far_site is None:
            far_site = qca.N - 1

        ground = qca.get_ground_state()
        X_op = Utils.local_operator(Utils.X, site, qca.N)
        X_op = np.kron(X_op, np.eye(qca.bond_dim))
        quenched = X_op @ ground

        ts = np.linspace(0, t_max, 61)
        Z_far = Utils.local_operator(Utils.Z, far_site, qca.N)
        Z_far = np.kron(Z_far, np.eye(qca.bond_dim))

        exp_ground = []
        exp_quenched = []
        for t in ts:
            psi_ground = qca.evolve_state(ground, t)
            psi_quenched = qca.evolve_state(quenched, t)

            exp_ground.append(np.real(psi_ground.conj().T @ (Z_far @ psi_ground)))
            exp_quenched.append(np.real(psi_quenched.conj().T @ (Z_far @ psi_quenched)))

        diff = np.abs(np.array(exp_quenched) - np.array(exp_ground))
        max_violation = np.max(diff)

        print(f"No-signalling test: max|Δ⟨Z_r⟩| = {max_violation:.3e} (r={far_site})")
        return max_violation

    @staticmethod
    def energy_drift(qca: ExactQCA, t_max: float = 3.0, ngrid: int = 61) -> float:
        state = qca.get_ground_state()
        ts = np.linspace(0, t_max, ngrid)

        energies = []
        for t in ts:
            psi_t = qca.evolve_state(state, t)
            energy = np.real(psi_t.conj().T @ (qca.H @ psi_t))
            energies.append(energy)

        energies = np.array(energies)
        drift = np.ptp(energies)
        print(f"Energy drift: ΔE = {drift:.3e}")
        return drift
