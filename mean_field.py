"""
Mean-field comparator utilities based on the entanglement profile.
"""
from __future__ import annotations

from typing import List

import numpy as np


class QuantumChain:
    """Mean-field comparator using exact χ profile."""

    def __init__(
        self,
        N: int,
        omega: float,
        alpha: float,
        chi_profile: List[float],
        J0: float,
        gamma: float,
    ):
        self.N = N
        self.omega = omega
        self.alpha = alpha
        self.chi_profile = chi_profile
        self.J0 = J0
        self.gamma = gamma
        self._compute_lambda()

    def _compute_lambda(self) -> None:
        self.Lambda = np.zeros(self.N)
        for i in range(self.N):
            if i > 0:
                self.Lambda[i] += np.log2(max(1.0, self.chi_profile[i - 1]))
            if i < self.N - 1:
                self.Lambda[i] += np.log2(max(1.0, self.chi_profile[i]))

    def get_effective_frequency(self, i: int) -> float:
        return self.omega / (1.0 + self.alpha * self.Lambda[i])

    def evolve_ramsey_single_ion(self, probe: int, t: float) -> float:
        omega_eff = self.get_effective_frequency(probe)
        dt = min(0.1, np.pi / (4.0 * omega_eff))
        steps = max(1, int(np.floor(t / dt)))

        state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)

        for _ in range(steps):
            angle = omega_eff * dt / 2.0
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            new_state = np.array(
                [
                    cos_a * state[0] - sin_a * state[2],
                    cos_a * state[1] - sin_a * state[3],
                    cos_a * state[2] + sin_a * state[0],
                    cos_a * state[3] + sin_a * state[1],
                ],
                dtype=complex,
            )

            if self.gamma > 0.0:
                decay = np.exp(-self.gamma * dt / 2.0)
                new_state[1] *= decay
                new_state[3] *= decay

            norm = np.sqrt(np.sum(np.abs(new_state) ** 2))
            state = new_state / norm

        sqrt2 = np.sqrt(2.0)
        final = np.array(
            [
                (state[0] - state[3]) / sqrt2,
                (state[1] + state[2]) / sqrt2,
                (state[2] + state[1]) / sqrt2,
                (state[3] - state[0]) / sqrt2,
            ],
            dtype=complex,
        )

        prob_up = np.abs(final[0]) ** 2 + np.abs(final[1]) ** 2
        prob_down = np.abs(final[2]) ** 2 + np.abs(final[3]) ** 2
        return prob_up - prob_down
