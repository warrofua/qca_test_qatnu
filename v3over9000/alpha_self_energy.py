"""
Operational alpha extraction from static self-energy response.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class AlphaSample:
    lambda_control: float
    lambda_local: float
    omega_eff: float
    sigma0: float


def _fit_frequency_fft(times: np.ndarray, signal: np.ndarray) -> float:
    if len(times) < 10:
        return 0.0
    dt = float(times[1] - times[0])
    if dt <= 0.0:
        return 0.0
    centered = signal - np.mean(signal)
    fft = np.fft.fft(centered)
    freqs = np.fft.fftfreq(len(times), d=dt)
    mask = freqs > 0
    if not np.any(mask):
        return 0.0
    amps = np.abs(fft[mask])
    pos_freqs = freqs[mask]
    idx = int(np.argmax(amps))
    base_freq = float(pos_freqs[idx])

    # Sub-bin quadratic interpolation around the FFT peak for smoother estimates.
    if 0 < idx < len(amps) - 1:
        a = float(amps[idx - 1])
        b = float(amps[idx])
        c = float(amps[idx + 1])
        denom = (a - 2.0 * b + c)
        if abs(denom) > 1e-14:
            delta = 0.5 * (a - c) / denom
            step = float(pos_freqs[1] - pos_freqs[0]) if len(pos_freqs) > 1 else 0.0
            base_freq = base_freq + delta * step

    return float(2.0 * np.pi * base_freq)


def _measure_probe_frequency(qca, base_state: np.ndarray, site: int, t_max: float, n_points: int) -> float:
    times = np.linspace(0.0, float(t_max), int(max(n_points, 20)))
    z_vals = np.zeros_like(times, dtype=float)
    for i, t in enumerate(times):
        psi = qca.apply_pi2_pulse(base_state, int(site))
        psi_t = qca.evolve_state(psi, float(t))
        z_vals[i] = float(qca.measure_Z(psi_t, int(site)))
    return _fit_frequency_fft(times, z_vals)


def estimate_alpha_from_self_energy(
    qca_builder: Callable[[float], object],
    lambdas: Iterable[float],
    probe_site: int,
    t_max: float = 20.0,
    n_points: int = 120,
    state_builder: Optional[Callable[[object, float], np.ndarray]] = None,
) -> Dict[str, object]:
    """
    Estimate alpha from sigma(0) = omega - omega_eff response slope versus local Lambda.

    qca_builder(lambda_control) must return a configured QCA instance.
    """
    samples: List[AlphaSample] = []
    omega_ref = None

    for lam in lambdas:
        qca = qca_builder(float(lam))
        if omega_ref is None:
            omega_ref = float(qca.config["omega"])
        ground = qca.get_ground_state()
        base_state = state_builder(qca, float(lam)) if state_builder is not None else ground
        lambda_local = float(qca.count_circuit_depth(int(probe_site), base_state))
        omega_eff = _measure_probe_frequency(
            qca=qca,
            base_state=base_state,
            site=int(probe_site),
            t_max=float(t_max),
            n_points=int(n_points),
        )
        sigma0 = float(omega_ref - omega_eff)
        samples.append(
            AlphaSample(
                lambda_control=float(lam),
                lambda_local=lambda_local,
                omega_eff=float(omega_eff),
                sigma0=sigma0,
            )
        )

    valid = [s for s in samples if np.isfinite(s.lambda_local) and np.isfinite(s.sigma0)]
    if omega_ref is None or len(valid) < 3:
        return {
            "alpha_self_energy": float("nan"),
            "omega_ref": float(omega_ref) if omega_ref is not None else float("nan"),
            "slope_sigma_vs_lambda": float("nan"),
            "intercept_sigma_vs_lambda": float("nan"),
            "r2": float("nan"),
            "num_points": len(valid),
            "samples": [s.__dict__ for s in samples],
        }

    valid_sorted = sorted(valid, key=lambda s: s.lambda_local)
    fit_count = max(3, len(valid_sorted) // 2)
    fit_set = valid_sorted[:fit_count]

    x = np.array([s.lambda_local for s in fit_set], dtype=float)
    y = np.array([s.sigma0 for s in fit_set], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    alpha_self_energy = float(slope / omega_ref) if abs(float(omega_ref)) > 1e-12 else float("nan")
    return {
        "alpha_self_energy": alpha_self_energy,
        "omega_ref": float(omega_ref),
        "slope_sigma_vs_lambda": float(slope),
        "intercept_sigma_vs_lambda": float(intercept),
        "r2": float(r2),
        "num_points": len(valid),
        "fit_points": fit_count,
        "samples": [s.__dict__ for s in samples],
    }
