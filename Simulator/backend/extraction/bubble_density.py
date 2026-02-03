"""
Lambda (bubble density) extraction: Theory vs measurement.

Λ_i = log₂(χ_{i-1}) + log₂(χ_i)  [bubble density at site i]

Theory predicts Λ from the frustration protocol and α.
Measurement extracts Λ from the quantum state bond dimensions.
"""

import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core_qca import ExactQCA


def extract_lambda_theory(
    lambda_val: float,
    alpha: float,
    N: int,
    chi_max: int,
    hotspot_multiplier: float = 3.0,
    frustration_time: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Predict Λ profile from theory.
    
    Theory model:
    - After frustration protocol with hotspot M×λ
    - Bond dimensions scale as χ ~ 1 + (M×λ)² × t_f
    - Λ_i = log₂(χ_{i-1}) + log₂(χ_i)
    
    Parameters
    ----------
    lambda_val : float
        Nominal promotion strength
    alpha : float
        Susceptibility
    N : int
        Number of sites
    chi_max : int
        Bond dimension cutoff
    hotspot_multiplier : float
        Frustration hotspot strength M
    frustration_time : float
        Duration of frustration protocol
        
    Returns
    -------
    dict
        {
            "Lambda": np.ndarray [N] - bubble density per site,
            "chi_bonds": np.ndarray [N-1] - bond dimensions,
            "method": "theory"
        }
    """
    # Effective hotspot strength
    lambda_hot = hotspot_multiplier * lambda_val
    
    # Theoretical bond dimension growth from frustration
    # χ grows with (λt)², saturates at chi_max
    # Simple model: χ = min(χ_max, 1 + (λ_hot)² × frustration_time)
    chi_ideal = 1.0 + (lambda_hot ** 2) * frustration_time
    chi_effective = min(chi_ideal, chi_max)
    
    # Bond dimensions (uniform for simplicity)
    # In reality, edges may have lower χ due to boundary effects
    chi_bonds = np.full(max(N - 1, 0), chi_effective)
    
    # Apply boundary corrections
    if len(chi_bonds) > 0:
        chi_bonds[0] = np.sqrt(chi_effective)  # Left boundary
        chi_bonds[-1] = np.sqrt(chi_effective)  # Right boundary
    
    # Compute Λ from bond dimensions
    Lambda = np.zeros(N)
    for i in range(N):
        if i > 0:
            Lambda[i] += np.log2(max(1.0, chi_bonds[i - 1]))
        if i < N - 1:
            Lambda[i] += np.log2(max(1.0, chi_bonds[i]))
    
    return {
        "Lambda": Lambda,
        "chi_bonds": chi_bonds,
        "method": "theory",
        "params": {
            "lambda_val": lambda_val,
            "alpha": alpha,
            "hotspot_multiplier": hotspot_multiplier,
            "chi_effective": chi_effective
        }
    }


def extract_lambda_measured(
    state: np.ndarray,
    qca: "ExactQCA"
) -> Dict[str, np.ndarray]:
    """
    Extract Λ profile from measured quantum state.
    
    Uses ExactQCA.get_bond_dimension() for each edge,
    then computes Λ_i = log₂(χ_{i-1}) + log₂(χ_i).
    
    Parameters
    ----------
    state : np.ndarray
        Quantum state vector
    qca : ExactQCA
        QCA instance with edge information
        
    Returns
    -------
    dict
        {
            "Lambda": np.ndarray [N] - bubble density per site,
            "chi_bonds": np.ndarray [edges] - measured bond dimensions,
            "method": "measured"
        }
    """
    N = qca.N
    
    # Measure bond dimensions for each edge
    chi_bonds = np.array([
        qca.get_bond_dimension(edge_idx, state)
        for edge_idx in range(qca.edge_count)
    ])
    
    # Compute Λ from bond dimensions
    # For path graph: edges = [(0,1), (1,2), ...]
    Lambda = np.zeros(N)
    
    for edge_idx, (u, v) in enumerate(qca.edges):
        chi = chi_bonds[edge_idx]
        # Both endpoints get contribution from this edge
        Lambda[u] += np.log2(max(1.0, chi))
        Lambda[v] += np.log2(max(1.0, chi))
    
    return {
        "Lambda": Lambda,
        "chi_bonds": chi_bonds,
        "method": "measured"
    }


def compute_lambda_discrepancy(
    theory: Dict[str, np.ndarray],
    measured: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute discrepancy between theory and measurement.
    
    Parameters
    ----------
    theory : dict
        Output from extract_lambda_theory()
    measured : dict
        Output from extract_lambda_measured()
        
    Returns
    -------
    dict
        Discrepancy metrics
    """
    Lambda_theory = theory["Lambda"]
    Lambda_measured = measured["Lambda"]
    
    # Absolute differences
    abs_diff = np.abs(Lambda_measured - Lambda_theory)
    rel_diff = np.abs(Lambda_measured - Lambda_theory) / (np.abs(Lambda_theory) + 1e-10)
    
    # Metrics
    return {
        "max_abs_error": float(np.max(abs_diff)),
        "mean_abs_error": float(np.mean(abs_diff)),
        "rms_error": float(np.sqrt(np.mean(abs_diff**2))),
        "max_rel_error": float(np.max(rel_diff)),
        "mean_rel_error": float(np.mean(rel_diff)),
        "correlation": float(np.corrcoef(Lambda_theory, Lambda_measured)[0, 1])
    }


def extract_lambda_total(
    lambda_profile: np.ndarray
) -> float:
    """
    Extract total bubble count from Λ profile.
    
    Λ_total = Σ_i Λ_i / N = average bubble density
    
    Parameters
    ----------
    lambda_profile : np.ndarray
        Λ values per site
        
    Returns
    -------
    float
        Average bubble density
    """
    return float(np.mean(lambda_profile))


def extract_lambda_gradient(
    lambda_profile: np.ndarray,
    edges: List[tuple]
) -> np.ndarray:
    """
    Extract spatial gradient of Λ across edges.
    
    Useful for identifying inhomogeneities in the bubble distribution.
    
    Parameters
    ----------
    lambda_profile : np.ndarray
        Λ values per site
    edges : list of (i, j)
        Graph edges
        
    Returns
    -------
    np.ndarray
        |Λ_i - Λ_j| for each edge
    """
    gradients = []
    for i, j in edges:
        gradients.append(abs(lambda_profile[i] - lambda_profile[j]))
    return np.array(gradients)


if __name__ == "__main__":
    # Test lambda extraction
    print("Lambda Extraction Test")
    print("=" * 50)
    
    # Theory prediction
    theory = extract_lambda_theory(
        lambda_val=0.5,
        alpha=0.8,
        N=4,
        chi_max=4,
        hotspot_multiplier=3.0
    )
    
    print(f"Theory prediction (λ=0.5, α=0.8):")
    print(f"  Λ = {theory['Lambda']}")
    print(f"  χ_bonds = {theory['chi_bonds']}")
    print(f"  Total Λ = {extract_lambda_total(theory['Lambda']):.4f}")
    
    # Simulate a measurement
    # (Would need actual QCA state for real test)
    print("\nSimulated 'measurement' with same values:")
    measured = {
        "Lambda": theory["Lambda"] * 0.95,  # 5% deviation
        "chi_bonds": theory["chi_bonds"] * 0.95,
        "method": "measured"
    }
    
    # Compute discrepancy
    disc = compute_lambda_discrepancy(theory, measured)
    print(f"\nDiscrepancy metrics:")
    print(f"  Mean abs error: {disc['mean_abs_error']:.4f}")
    print(f"  Mean rel error: {disc['mean_rel_error']:.2%}")
    print(f"  Correlation: {disc['correlation']:.4f}")
