"""
Newtonian limit: Compute κ from target Newton's constant.

From mega_document.md Appendix D:
4πG_eff = α × c² × κ / a²

Solving for κ:
κ = 4πG_eff × a² / (α × c²)
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_kappa_from_Geff(
    G_eff: float,
    alpha: float,
    c: float,
    a: float,
    log_callback: Optional[callable] = None
) -> float:
    """
    Compute degree penalty κ from target Newton's constant.
    
    From the Poisson equation in the Newtonian limit:
    (∇² - μ²)Λ = -κρ_ent
    
    The effective Newton constant is:
    4πG_eff = αc²κ/a²
    
    Parameters
    ----------
    G_eff : float
        Target Newton's constant (dimensionless in lattice units)
    alpha : float
        Susceptibility (from Stinespring or perturbative)
    c : float
        Speed of light
    a : float
        Lattice spacing (from scale anchoring)
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    float
        Degree penalty κ
    """
    numerator = 4 * np.pi * G_eff * a**2
    denominator = alpha * c**2
    
    if denominator == 0:
        raise ValueError("alpha and c must be non-zero")
    
    kappa = numerator / denominator
    
    if log_callback:
        # Verify the relation
        G_check = alpha * c**2 * kappa / (4 * np.pi * a**2)
        
        log_callback({
            "parameter": "kappa",
            "formula": "κ = 4πG_eff × a² / (α × c²)",
            "inputs": {
                "G_eff": G_eff,
                "alpha": alpha,
                "c": c,
                "a": a
            },
            "output": float(kappa),
            "steps": [
                f"κ = 4π × {G_eff:.6e} × {a:.4f}² / ({alpha:.6f} × {c:.4f}²)",
                f"κ = {numerator:.6e} / {denominator:.6f}",
                f"κ = {kappa:.6e}",
                f"Verification: G_eff = {G_check:.6e} (target: {G_eff:.6e})",
                f"Match: {'✓' if np.isclose(G_check, G_eff) else '✗'}"
            ]
        })
    
    return kappa


def compute_effective_G(
    alpha: float,
    kappa: float,
    c: float,
    a: float,
    log_callback: Optional[callable] = None
) -> float:
    """
    Compute effective G from microscopic parameters.
    
    Inverse of compute_kappa_from_Geff.
    
    Parameters
    ----------
    alpha : float
        Susceptibility
    kappa : float
        Degree penalty
    c : float
        Speed of light
    a : float
        Lattice spacing
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    float
        Effective Newton's constant
    """
    G_eff = alpha * c**2 * kappa / (4 * np.pi * a**2)
    
    if log_callback:
        log_callback({
            "parameter": "G_eff_effective",
            "formula": "G_eff = αc²κ / (4πa²)",
            "inputs": {"alpha": alpha, "kappa": kappa, "c": c, "a": a},
            "output": float(G_eff),
            "steps": [
                f"G_eff = {alpha:.6f} × {c:.4f}² × {kappa:.6e} / (4π × {a:.4f}²)",
                f"G_eff = {G_eff:.6e}"
            ]
        })
    
    return G_eff


def compute_target_degree(
    topology: str,
    N: int,
    log_callback: Optional[callable] = None
) -> float:
    """
    Compute target degree k0 from topology.
    
    For path/cycle: k0 = 2 (interior sites)
    For star: k0 ~ 2 (balance center vs leaves)
    For grid: k0 = 4 (2D coordination)
    
    Parameters
    ----------
    topology : str
        Graph topology
    N : int
        Number of sites
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    float
        Target degree k0
    """
    k0_map = {
        'path': 2.0,
        'cycle': 2.0,
        'star': min(2.0, N - 1),  # Effective average
        'diamond': 2.5,  # Mixed degree
        'bowtie': 2.5,
        'grid': 4.0,  # 2D
    }
    
    k0 = k0_map.get(topology, 2.0)
    
    if log_callback:
        log_callback({
            "parameter": "k0_target_degree",
            "formula": "Topology-dependent average degree",
            "inputs": {"topology": topology, "N": N},
            "output": float(k0),
            "steps": [
                f"Topology: {topology}",
                f"Target degree k0 = {k0}",
                "Note: Only relevant for dynamical geometry"
            ]
        })
    
    return k0


def check_newtonian_consistency(
    G_eff_measured: float,
    G_eff_target: float,
    tolerance: float = 0.1,
    log_callback: Optional[callable] = None
) -> bool:
    """
    Check if measured G_eff matches target within tolerance.
    
    Parameters
    ----------
    G_eff_measured : float
        Measured from simulation
    G_eff_target : float
        Target value
    tolerance : float, default 0.1
        Relative tolerance
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    bool
        True if consistent
    """
    relative_error = abs(G_eff_measured - G_eff_target) / G_eff_target
    is_consistent = relative_error < tolerance
    
    if log_callback:
        log_callback({
            "parameter": "G_consistency_check",
            "formula": "|G_measured - G_target| / G_target < tolerance",
            "inputs": {
                "G_measured": G_eff_measured,
                "G_target": G_eff_target,
                "tolerance": tolerance
            },
            "output": is_consistent,
            "steps": [
                f"G_measured = {G_eff_measured:.6e}",
                f"G_target = {G_eff_target:.6e}",
                f"Relative error = {relative_error:.2%}",
                f"Tolerance = {tolerance:.1%}",
                f"Consistent: {'✓' if is_consistent else '✗'}"
            ]
        })
    
    return is_consistent


if __name__ == "__main__":
    # Test Newtonian derivation
    def print_log(entry: Dict[str, Any]):
        print(f"\n{entry['parameter']}:")
        print(f"  Formula: {entry['formula']}")
        if isinstance(entry['output'], bool):
            print(f"  Output: {entry['output']}")
        else:
            print(f"  Output: {entry['output']:.6e}")
        for step in entry.get('steps', []):
            print(f"    → {step}")
    
    print("Newtonian Limit Derivation Test:")
    
    # Typical values
    G_eff = 6.674e-11  # Newton's constant (would be dimensionless in code)
    alpha = 0.8
    c = 1.0
    a = 1.386  # From chi_max=4
    
    kappa = compute_kappa_from_Geff(G_eff, alpha, c, a, print_log)
    
    # Verify
    G_check = compute_effective_G(alpha, kappa, c, a, print_log)
    
    # Check consistency
    check_newtonian_consistency(G_check, G_eff, log_callback=print_log)
    
    # Target degree
    k0 = compute_target_degree('path', 4, print_log)
    
    print(f"\nSummary:")
    print(f"  κ = {kappa:.6e}")
    print(f"  k0 = {k0}")
