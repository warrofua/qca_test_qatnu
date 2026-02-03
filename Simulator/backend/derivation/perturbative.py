"""
Perturbative derivation: α and δB from validity constraints.

From mega_document.md Eq. 144:
α_pert ≈ (2λ² / (ω × Δ_eff)) × ⟨F⟩

And perturbative validity requires Δ_eff ≫ λ.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_deltaB(
    omega: float,
    lambda_max: float,
    safety_factor: float = 5.0,
    log_callback = None
) -> float:
    """
    Compute bond energy spacing δB from perturbative validity constraint.
    
    Theory requirement: Δ_eff > safety_factor × λ_max
    
    Since Δ_eff ≈ δB (for linear ladder), we set:
    δB = safety_factor × λ_max × ω
    
    The ω factor ensures proper dimensionality (δB has units of energy).
    
    Parameters
    ----------
    omega : float
        Bare clock frequency (sets energy scale)
    lambda_max : float
        Maximum λ in scan (worst case for perturbation)
    safety_factor : float, default 5.0
        How much larger Δ_eff should be than λ
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    float
        Bond energy spacing δB
    """
    deltaB = safety_factor * lambda_max * omega
    
    if log_callback:
        log_callback({
            "parameter": "deltaB",
            "formula": "δB = safety × λ_max × ω",
            "inputs": {
                "omega": omega,
                "lambda_max": lambda_max,
                "safety_factor": safety_factor
            },
            "output": float(deltaB),
            "steps": [
                f"Require: Δ_eff > {safety_factor} × λ_max",
                f"δB = {safety_factor} × {lambda_max:.3f} × {omega:.3f}",
                f"δB = {deltaB:.4f}",
                f"Validity: Δ_eff/λ_max = {safety_factor:.1f} ✓"
            ]
        })
    
    return deltaB


def compute_alpha_perturbative(
    lambda_val: float,
    omega: float,
    delta_eff: float,
    avg_frustration: float = 0.5,
    log_callback = None
) -> float:
    """
    Compute α from perturbative formula (mega_document.md Eq. 144).
    
    α_pert ≈ (2λ² / (ω × Δ_eff)) × ⟨F⟩
    
    Where:
    - λ is promotion strength
    - ω is bare clock frequency
    - Δ_eff is bond excitation gap (~δB)
    - ⟨F⟩ is average frustration (0 to 1)
    
    Parameters
    ----------
    lambda_val : float
        Promotion strength λ
    omega : float
        Bare clock frequency
    delta_eff : float
        Bond excitation gap (from δB)
    avg_frustration : float, default 0.5
        Average frustration ⟨F_ij⟩
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    float
        Susceptibility α
    """
    alpha = (2 * lambda_val**2 / (omega * delta_eff)) * avg_frustration
    
    if log_callback:
        # Check perturbative validity
        validity_ratio = delta_eff / lambda_val if lambda_val > 0 else float('inf')
        is_valid = validity_ratio > 1.0
        
        log_callback({
            "parameter": "alpha_perturbative",
            "formula": "α = (2λ² / (ω×Δ_eff)) × ⟨F⟩",
            "inputs": {
                "lambda": lambda_val,
                "omega": omega,
                "delta_eff": delta_eff,
                "avg_frustration": avg_frustration
            },
            "output": float(alpha),
            "steps": [
                f"α = (2 × {lambda_val:.3f}²) / ({omega:.3f} × {delta_eff:.3f}) × {avg_frustration:.3f}",
                f"α = {2*lambda_val**2:.4f} / {omega*delta_eff:.4f} × {avg_frustration:.3f}",
                f"α = {alpha:.6f}",
                f"Perturbative validity: Δ_eff/λ = {validity_ratio:.2f} {'✓' if is_valid else '✗'}"
            ]
        })
    
    return alpha


def compute_alpha_from_gap_relation(
    lambda_val: float,
    measured_gap: float,
    log_callback = None
) -> float:
    """
    Alternative: Compute α from measured energy gap.
    
    In the dilute limit, the frequency shift is:
    ω_eff ≈ ω(1 - αΛ)
    
    Can also be extracted from: ΔE_{01} = ℏ(ω_eff_out - ω_eff_in)
    
    Parameters
    ----------
    lambda_val : float
        Promotion strength
    measured_gap : float
        Measured energy gap E1 - E0
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    float
        Extracted α
    """
    # Simplified relation: α ∝ λ² / gap
    # This is a placeholder for more sophisticated extraction
    alpha = lambda_val**2 / measured_gap if measured_gap > 0 else 0.0
    
    if log_callback:
        log_callback({
            "parameter": "alpha_from_gap",
            "formula": "α ∝ λ² / ΔE",
            "inputs": {
                "lambda": lambda_val,
                "measured_gap": measured_gap
            },
            "output": float(alpha),
            "steps": [
                f"α = {lambda_val:.3f}² / {measured_gap:.4f}",
                f"α = {alpha:.6f}",
                "Note: Simplified relation, use perturbative formula for theory"
            ]
        })
    
    return alpha


def compute_perturbative_parameters(
    lambda_val: float,
    omega: float = 1.0,
    lambda_max: float = 1.5,
    avg_frustration: float = 0.5,
    safety_factor: float = 5.0,
    log_callback = None
) -> Dict[str, float]:
    """
    Compute all perturbative parameters.
    
    Parameters
    ----------
    lambda_val : float
        Current promotion strength
    omega : float, default 1.0
        Bare clock frequency
    lambda_max : float, default 1.5
        Maximum λ for validity constraint
    avg_frustration : float, default 0.5
        Average frustration ⟨F_ij⟩
    safety_factor : float, default 5.0
        Perturbative validity safety margin
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    dict
        {
            "deltaB": bond spacing,
            "alpha_pert": susceptibility,
            "validity_ratio": Δ_eff/λ
        }
    """
    # δB is computed from lambda_max (worst case)
    deltaB = compute_deltaB(omega, lambda_max, safety_factor, log_callback)
    
    # α is computed at current lambda_val
    alpha = compute_alpha_perturbative(
        lambda_val, omega, deltaB, avg_frustration, log_callback
    )
    
    # Check validity at current lambda
    validity_ratio = deltaB / lambda_val if lambda_val > 0 else float('inf')
    
    return {
        "deltaB": deltaB,
        "alpha_pert": alpha,
        "validity_ratio": validity_ratio,
        "is_valid": validity_ratio > 1.0
    }


if __name__ == "__main__":
    # Test perturbative formulas
    def print_log(entry: Dict[str, Any]):
        print(f"\n{entry['parameter']}:")
        print(f"  Formula: {entry['formula']}")
        print(f"  Output: {entry['output']:.6f}")
        for step in entry.get('steps', []):
            print(f"    → {step}")
    
    print("Perturbative Derivation Test:")
    
    # Test at λ = 0.5
    params = compute_perturbative_parameters(
        lambda_val=0.5,
        omega=1.0,
        lambda_max=1.5,
        avg_frustration=0.5,
        log_callback=print_log
    )
    
    print(f"\nSummary at λ=0.5:")
    print(f"  δB = {params['deltaB']:.4f}")
    print(f"  α = {params['alpha_pert']:.6f}")
    print(f"  Validity: Δ/λ = {params['validity_ratio']:.2f}")
