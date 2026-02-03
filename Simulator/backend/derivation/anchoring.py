"""
Scale anchoring: Compute lattice spacing and time step from physical constants.

From mega_document.md Section 6:
- Vacuum anchor: Group velocity c_0 in lattice units → physical c
- Black hole anchor: Bond entropy γ = ln(χ_max) → Bekenstein-Hawking entropy
"""

import numpy as np
from typing import Dict, Any


def compute_lattice_spacing(
    chi_max: int,
    beta: float = 1.0,
    log_callback = None
) -> float:
    """
    Compute lattice spacing a from Bekenstein-Hawking entropy anchor.
    
    From S_QATNU = S_BH:
    - S_QATNU = γ × N_∂ = ln(χ_max) × N_∂
    - S_BH = A / (4ℓ_P²) = β × N_∂ × a² / (4ℓ_P²)
    
    Equating: γ = β × a² / (4ℓ_P²)
    
    Therefore: a = 2ℓ_P √(γ/β)
    
    Parameters
    ----------
    chi_max : int
        Bond dimension cutoff
    beta : float, default 1.0
        Geometric factor relating N_∂ to area
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    float
        Lattice spacing a in Planck units (ℓ_P = 1)
    """
    gamma = np.log(chi_max)
    a = 2 * np.sqrt(gamma / beta)
    
    if log_callback:
        log_callback({
            "parameter": "lattice_spacing_a",
            "formula": "a = 2ℓ_P √(γ/β)",
            "inputs": {
                "chi_max": chi_max,
                "gamma": float(gamma),
                "beta": beta
            },
            "output": float(a),
            "steps": [
                f"γ = ln(χ_max) = ln({chi_max}) = {gamma:.4f}",
                f"a = 2 × √({gamma:.4f}/{beta}) = {a:.4f} ℓ_P"
            ]
        })
    
    return a


def compute_time_step(
    c_lattice: float,
    a: float,
    c_physical: float = 1.0,
    log_callback = None
) -> float:
    """
    Compute time step τ from vacuum anchor.
    
    Physical group velocity: v_phys = c_lattice × a/τ
    Vacuum anchor requires: v_phys = c_physical
    
    Therefore: τ = c_lattice × a / c_physical
    
    Parameters
    ----------
    c_lattice : float
        Group velocity in lattice units (~1/√2 for QATNU)
    a : float
        Lattice spacing (from compute_lattice_spacing)
    c_physical : float, default 1.0
        Physical speed of light (c = 1 in natural units)
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    float
        Time step τ in Planck units
    """
    tau = c_lattice * a / c_physical
    
    if log_callback:
        log_callback({
            "parameter": "time_step_tau",
            "formula": "τ = c_lattice × a / c",
            "inputs": {
                "c_lattice": c_lattice,
                "a": a,
                "c_physical": c_physical
            },
            "output": float(tau),
            "steps": [
                f"c_lattice = {c_lattice:.4f} (lattice units)",
                f"a = {a:.4f} ℓ_P",
                f"τ = {c_lattice:.4f} × {a:.4f} / {c_physical} = {tau:.4f} t_P"
            ]
        })
    
    return tau


def compute_bond_entropy_gamma(
    chi_max: int,
    log_callback = None
) -> float:
    """
    Compute bond entropy γ = ln(χ_max).
    
    This is the entropy per boundary site in the QATNU model,
    related to Bekenstein-Hawking entropy through S = γ × N_∂.
    
    Parameters
    ----------
    chi_max : int
        Bond dimension cutoff
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    float
        Bond entropy γ = ln(χ_max)
    """
    gamma = np.log(chi_max)
    
    if log_callback:
        log_callback({
            "parameter": "bond_entropy_gamma",
            "formula": "γ = ln(χ_max)",
            "inputs": {"chi_max": chi_max},
            "output": float(gamma),
            "steps": [
                f"γ = ln({chi_max}) = {gamma:.4f}"
            ]
        })
    
    return gamma


def compute_UV_energy_scale(
    tau: float,
    log_callback = None
) -> float:
    """
    Compute UV energy scale E_UV = ℏ/τ.
    
    In Planck units (ℏ = 1): E_UV = 1/τ
    
    Parameters
    ----------
    tau : float
        Time step in Planck units
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    float
        UV energy scale in Planck units
    """
    E_UV = 1.0 / tau
    
    if log_callback:
        log_callback({
            "parameter": "UV_energy",
            "formula": "E_UV = ℏ/τ",
            "inputs": {"tau": tau},
            "output": float(E_UV),
            "steps": [
                f"τ = {tau:.4f} t_P",
                f"E_UV = 1/{tau:.4f} = {E_UV:.4f} E_P"
            ]
        })
    
    return E_UV


def compute_all_scales(
    chi_max: int,
    c_lattice: float = 1/np.sqrt(2),
    c_physical: float = 1.0,
    beta: float = 1.0,
    log_callback = None
) -> Dict[str, float]:
    """
    Compute all scale anchoring parameters.
    
    Parameters
    ----------
    chi_max : int
        Bond dimension cutoff
    c_lattice : float, default 1/sqrt(2)
        Group velocity in lattice units
    c_physical : float, default 1.0
        Physical speed of light
    beta : float, default 1.0
        Geometric factor for horizon area
    log_callback : callable, optional
        Function to log derivation steps
        
    Returns
    -------
    dict
        {
            "lattice_spacing": a in ℓ_P,
            "time_step": τ in t_P,
            "UV_energy": E_UV in E_P
        }
    """
    a = compute_lattice_spacing(chi_max, beta, log_callback)
    tau = compute_time_step(c_lattice, a, c_physical, log_callback)
    E_UV = compute_UV_energy_scale(tau, log_callback)
    
    return {
        "lattice_spacing": a,
        "time_step": tau,
        "UV_energy": E_UV
    }


if __name__ == "__main__":
    # Test scale anchoring
    def print_log(entry: Dict[str, Any]):
        print(f"\n{entry['parameter']}:")
        print(f"  Formula: {entry['formula']}")
        print(f"  Output: {entry['output']:.4f}")
        for step in entry['steps']:
            print(f"    → {step}")
    
    print("Scale Anchoring Test (χ_max=4):")
    scales = compute_all_scales(chi_max=4, log_callback=print_log)
    
    print(f"\nSummary:")
    print(f"  a = {scales['lattice_spacing']:.4f} ℓ_P")
    print(f"  τ = {scales['time_step']:.4f} t_P")
    print(f"  E_UV = {scales['UV_energy']:.4f} E_P")
