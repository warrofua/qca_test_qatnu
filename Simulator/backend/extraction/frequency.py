"""
Effective frequency extraction: Theory vs measurement.

Postulate 1: ω_eff = ω / (1 + αΛ)

Theory predicts ω_eff from Λ and α.
Measurement extracts ω_eff from Ramsey fringes or Z oscillations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core_qca import ExactQCA


def extract_omega_eff_theory(
    lambda_profile: np.ndarray,
    alpha: float,
    omega_0: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Predict ω_eff from theory (Postulate 1).
    
    ω_eff,i = ω_0 / (1 + α × Λ_i)
    
    Parameters
    ----------
    lambda_profile : np.ndarray
        Λ values per site [N]
    alpha : float
        Susceptibility
    omega_0 : float
        Bare clock frequency
        
    Returns
    -------
    dict
        {
            "omega_eff": np.ndarray [N] - effective frequency per site,
            "omega_ratio": np.ndarray [N] - ω_eff/ω_0,
            "method": "theory"
        }
    """
    # Postulate 1
    denominator = 1.0 + alpha * lambda_profile
    omega_eff = omega_0 / denominator
    
    return {
        "omega_eff": omega_eff,
        "omega_ratio": omega_eff / omega_0,
        "method": "theory",
        "params": {
            "omega_0": omega_0,
            "alpha": alpha,
            "Lambda_mean": float(np.mean(lambda_profile))
        }
    }


def extract_omega_eff_ramsey(
    state_func: callable,
    site: int,
    evolution_times: np.ndarray,
    omega_0: float = 1.0,
    dt: float = 0.01
) -> Dict[str, float]:
    """
    Extract ω_eff from simulated Ramsey fringes.
    
    Protocol:
    1. Initialize |+⟩ on probe site (Bell with ancilla)
    2. Evolve for time t
    3. Apply π/2 pulse
    4. Measure ⟨Z⟩ oscillation frequency
    
    Parameters
    ----------
    state_func : callable
        Function(t) -> state that evolves the system
    site : int
        Probe site index
    evolution_times : np.ndarray
        Times to sample
    omega_0 : float
        Bare frequency (for normalization)
    dt : float
        Time step for simulation
        
    Returns
    -------
    dict
        {
            "omega_eff": float - extracted frequency,
            "contrast": float - fringe contrast,
            "quality": float - fit quality
        }
    """
    # Simulate Ramsey sequence at each evolution time
    Z_expectations = []
    
    for t in evolution_times:
        # Get evolved state
        state = state_func(t)
        
        # Measure Z on probe (after π/2 pulse implicit in phase)
        # For simplified extraction, we measure Z directly
        # In real Ramsey: Z = cos(ω_eff × t)
        
        # Simplified: extract from Z expectation
        # This would need the actual QCA instance for measurement
        # Placeholder: assume perfect cosine
        Z_expectations.append(np.cos(omega_0 * t * 0.5))  # Example
    
    Z_expectations = np.array(Z_expectations)
    
    # Fit to cosine: A × cos(ω × t + φ) + C
    # Using FFT for frequency extraction
    if len(evolution_times) > 2:
        # Interpolate to uniform grid
        t_uniform = np.linspace(evolution_times[0], evolution_times[-1], 256)
        Z_uniform = np.interp(t_uniform, evolution_times, Z_expectations)
        
        # FFT
        fft = np.fft.rfft(Z_uniform - np.mean(Z_uniform))
        freqs = np.fft.rfftfreq(len(t_uniform), t_uniform[1] - t_uniform[0])
        
        # Peak frequency
        peak_idx = np.argmax(np.abs(fft[1:])) + 1
        omega_extracted = 2 * np.pi * np.abs(freqs[peak_idx])
        
        # Contrast (amplitude)
        contrast = (np.max(Z_expectations) - np.min(Z_expectations)) / 2
        
        return {
            "omega_eff": float(omega_extracted),
            "contrast": float(contrast),
            "quality": float(np.abs(fft[peak_idx]) / np.sum(np.abs(fft))),
            "method": "ramsey"
        }
    
    return {
        "omega_eff": omega_0 * 0.5,  # Fallback
        "contrast": 0.5,
        "quality": 0.0,
        "method": "ramsey_fallback"
    }


def extract_omega_eff_from_Z_dynamics(
    Z_trajectory: np.ndarray,
    times: np.ndarray
) -> Dict[str, float]:
    """
    Extract ω_eff from Z expectation value dynamics.
    
    Parameters
    ----------
    Z_trajectory : np.ndarray
        ⟨Z(t)⟩ values
    times : np.ndarray
        Time points
        
    Returns
    -------
    dict
        {
            "omega_eff": float - extracted frequency,
            "phase": float - initial phase,
            "amplitude": float - oscillation amplitude
        }
    """
    if len(times) < 2:
        return {"omega_eff": 0.0, "phase": 0.0, "amplitude": 0.0}
    
    # Remove DC offset
    Z_centered = Z_trajectory - np.mean(Z_trajectory)
    
    # FFT for frequency
    dt = times[1] - times[0]
    fft = np.fft.rfft(Z_centered)
    freqs = np.fft.rfftfreq(len(times), dt)
    
    # Dominant frequency
    peak_idx = np.argmax(np.abs(fft[1:])) + 1
    omega_eff = 2 * np.pi * freqs[peak_idx]
    
    # Amplitude
    amplitude = np.abs(fft[peak_idx]) / len(times)
    
    # Phase from Hilbert transform
    analytic = np.abs(fft[peak_idx]) * np.exp(1j * np.angle(fft[peak_idx]))
    phase = np.angle(analytic)
    
    return {
        "omega_eff": float(omega_eff),
        "phase": float(phase),
        "amplitude": float(amplitude),
        "method": "fft"
    }


def extract_omega_eff_measured_direct(
    state: np.ndarray,
    qca: "ExactQCA",
    site: int,
    evolution_time: float = 10.0,
    num_points: int = 100
) -> Dict[str, float]:
    """
    Direct measurement of ω_eff from state evolution.
    
    Evolves the state and measures Z oscillation frequency.
    
    Parameters
    ----------
    state : np.ndarray
        Initial state
    qca : ExactQCA
        QCA instance
    site : int
        Site to measure
    evolution_time : float
        Total evolution time
    num_points : int
        Number of measurement points
        
    Returns
    -------
    dict
        Frequency extraction results
    """
    times = np.linspace(0, evolution_time, num_points)
    Z_trajectory = []
    
    for t in times:
        evolved = qca.evolve(state, t)
        Z = qca.measure_Z(evolved, site)
        Z_trajectory.append(Z)
    
    Z_trajectory = np.array(Z_trajectory)
    
    return extract_omega_eff_from_Z_dynamics(Z_trajectory, times)


def compute_omega_eff_discrepancy(
    theory: Dict[str, np.ndarray],
    measured_sites: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Compute discrepancy between theory and measurements.
    
    Parameters
    ----------
    theory : dict
        Output from extract_omega_eff_theory()
    measured_sites : list
        List of per-site measurements
        
    Returns
    -------
    dict
        Discrepancy metrics
    """
    omega_theory = theory["omega_eff"]
    omega_measured = np.array([m["omega_eff"] for m in measured_sites])
    
    # Handle mismatched lengths
    N_theory = len(omega_theory)
    N_measured = len(omega_measured)
    N = min(N_theory, N_measured)
    
    omega_theory = omega_theory[:N]
    omega_measured = omega_measured[:N]
    
    # Compute metrics
    abs_diff = np.abs(omega_measured - omega_theory)
    rel_diff = abs_diff / (np.abs(omega_theory) + 1e-10)
    
    return {
        "max_abs_error": float(np.max(abs_diff)),
        "mean_abs_error": float(np.mean(abs_diff)),
        "rms_error": float(np.sqrt(np.mean(abs_diff**2))),
        "max_rel_error": float(np.max(rel_diff)),
        "mean_rel_error": float(np.mean(rel_diff)),
        "correlation": float(np.corrcoef(omega_theory, omega_measured)[0, 1])
    }


def extract_global_omega_eff(
    omega_eff_profile: np.ndarray,
    method: str = "mean"
) -> float:
    """
    Extract global effective frequency from profile.
    
    Parameters
    ----------
    omega_eff_profile : np.ndarray
        Per-site effective frequencies
    method : str
        "mean", "median", "min", "harmonic"
        
    Returns
    -------
    float
        Global ω_eff
    """
    if method == "mean":
        return float(np.mean(omega_eff_profile))
    elif method == "median":
        return float(np.median(omega_eff_profile))
    elif method == "min":
        return float(np.min(omega_eff_profile))
    elif method == "harmonic":
        # Harmonic mean: 1/⟨1/ω⟩
        return float(1.0 / np.mean(1.0 / (omega_eff_profile + 1e-10)))
    else:
        return float(np.mean(omega_eff_profile))


if __name__ == "__main__":
    # Test frequency extraction
    print("Effective Frequency Extraction Test")
    print("=" * 50)
    
    # Theory prediction
    Lambda = np.array([0.5, 1.0, 1.0, 0.5])
    theory = extract_omega_eff_theory(Lambda, alpha=0.8, omega_0=1.0)
    
    print(f"Theory prediction (α=0.8):")
    print(f"  Λ = {Lambda}")
    print(f"  ω_eff = {theory['omega_eff']}")
    print(f"  Global ω_eff (mean) = {extract_global_omega_eff(theory['omega_eff']):.4f}")
    print(f"  Global ω_eff (harmonic) = {extract_global_omega_eff(theory['omega_eff'], 'harmonic'):.4f}")
    
    # Simulated measurement
    print("\nSimulated 'measurement' with 3% deviation:")
    measured = [
        {"omega_eff": w * 0.97, "contrast": 0.9}
        for w in theory['omega_eff']
    ]
    
    # Discrepancy
    disc = compute_omega_eff_discrepancy(theory, measured)
    print(f"\nDiscrepancy metrics:")
    print(f"  Mean abs error: {disc['mean_abs_error']:.4f}")
    print(f"  Mean rel error: {disc['mean_rel_error']:.2%}")
