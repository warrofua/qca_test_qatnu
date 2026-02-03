"""
SRQID validation checks: Verify SRQID physical claims.

Checks:
1. Lieb-Robinson velocity bound
2. No-signaling (causality)
3. Energy drift (unitarity)
4. Entropy scaling (Page curve)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core_qca import ExactQCA


def check_lieb_robinson_velocity(
    commutator_norms: np.ndarray,
    distances: np.ndarray,
    times: np.ndarray,
    c_expected: float = 2.0
) -> Dict[str, float]:
    """
    Verify Lieb-Robinson bound: v_LR ≤ c_expected.
    
    The bound states: ||[A(t), B]|| ≤ C × exp(-(d - v×t)/ξ)
    
    Parameters
    ----------
    commutator_norms : np.ndarray [distances, times]
        ||[X_i(t), X_j]|| for various i,j distances and times
    distances : np.ndarray
        Spatial separations
    times : np.ndarray
        Evolution times
    c_expected : float
        Expected light cone velocity
        
    Returns
    -------
    dict
        Validation results
    """
    # Find where commutator becomes significant (> threshold)
    threshold = 0.1
    
    v_estimates = []
    for d_idx, d in enumerate(distances):
        for t_idx, t in enumerate(times):
            if t > 0 and commutator_norms[d_idx, t_idx] > threshold:
                v_estimates.append(d / t)
                break
    
    if v_estimates:
        v_measured = np.percentile(v_estimates, 95)  # Conservative estimate
    else:
        v_measured = 0.0
    
    return {
        "v_measured": float(v_measured),
        "v_expected": float(c_expected),
        "is_satisfied": v_measured <= c_expected * 1.2,  # 20% tolerance
        "margin": float((c_expected - v_measured) / c_expected),
        "confidence": float(min(1.0, len(v_estimates) / len(distances)))
    }


def check_no_signaling(
    Z_expectations: Dict[int, List[float]],
    times: np.ndarray,
    perturbation_site: int,
    tolerance: float = 1e-15
) -> Dict[str, float]:
    """
    Verify no-signaling: causality is respected.
    
    If we perturb at site A, sites outside light cone should show
    no change in Z expectation (within numerical precision).
    
    Parameters
    ----------
    Z_expectations : dict
        {site: [Z(t) for t in times]} for each site
    times : np.ndarray
        Evolution times
    perturbation_site : int
        Site where perturbation was applied
    tolerance : float
        Numerical tolerance for no-change detection
        
    Returns
    -------
    dict
        Validation results
    """
    max_violations = []
    
    for site, Z_traj in Z_expectations.items():
        if site == perturbation_site:
            continue
        
        # Check if Z changes significantly
        Z_initial = Z_traj[0]
        max_change = max(abs(Z - Z_initial) for Z in Z_traj)
        max_violations.append(max_change)
    
    if max_violations:
        max_violation = max(max_violations)
        mean_violation = np.mean(max_violations)
    else:
        max_violation = 0.0
        mean_violation = 0.0
    
    return {
        "max_violation": float(max_violation),
        "mean_violation": float(mean_violation),
        "tolerance": float(tolerance),
        "is_satisfied": max_violation < tolerance,
        "num_sites_checked": len(max_violations)
    }


def check_energy_drift(
    energies: np.ndarray,
    tolerance: float = 1e-14
) -> Dict[str, float]:
    """
    Verify energy is conserved (unitarity check).
    
    For exact diagonalization, energy should be conserved to machine precision.
    
    Parameters
    ----------
    energies : np.ndarray
        Energy at different times
    tolerance : float
        Acceptable relative drift
        
    Returns
    -------
    dict
        Validation results
    """
    if len(energies) < 2:
        return {
            "drift_relative": 0.0,
            "is_conserved": True,
            "max_deviation": 0.0
        }
    
    E_initial = energies[0]
    E_final = energies[-1]
    
    # Relative drift
    if abs(E_initial) > 1e-15:
        drift_relative = abs(E_final - E_initial) / abs(E_initial)
    else:
        drift_relative = abs(E_final - E_initial)
    
    # Max deviation from initial
    max_deviation = max(abs(E - E_initial) for E in energies)
    
    return {
        "drift_relative": float(drift_relative),
        "max_deviation": float(max_deviation),
        "E_initial": float(E_initial),
        "E_final": float(E_final),
        "is_conserved": drift_relative < tolerance,
        "tolerance": float(tolerance)
    }


def check_entropy_scaling(
    entropies: np.ndarray,
    subsystem_sizes: np.ndarray,
    total_N: int,
    expected_page: bool = True
) -> Dict[str, float]:
    """
    Verify entropy scaling matches Page curve.
    
    For random states: S(A) ≈ min(|A|, |B|) × log(D) - constant
    
    Parameters
    ----------
    entropies : np.ndarray
        Entropy for each subsystem size
    subsystem_sizes : np.ndarray
        Sizes of subsystems
    total_N : int
        Total system size
    expected_page : bool
        Whether Page curve is expected (thermalization)
        
    Returns
    -------
    dict
        Validation results
    """
    # Fit to Page curve: S = a × min(k, N-k) + b
    min_sizes = np.minimum(subsystem_sizes, total_N - subsystem_sizes)
    
    # Linear fit
    if len(entropies) > 2:
        coeffs = np.polyfit(min_sizes, entropies, 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        # R²
        predicted = slope * min_sizes + intercept
        ss_res = np.sum((entropies - predicted) ** 2)
        ss_tot = np.sum((entropies - np.mean(entropies)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope, intercept = 0.0, 0.0
        r_squared = 0.0
    
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "matches_page": r_squared > 0.95,
        "expected_slope": float(np.log(2)),  # For qubits
        "slope_discrepancy": float(abs(slope - np.log(2)))
    }


def check_postulate1_residuals(
    omega_measured: np.ndarray,
    omega_theory: np.ndarray,
    tolerance: float = 0.1
) -> Dict[str, float]:
    """
    Check residuals of Postulate 1 predictions.
    
    Parameters
    ----------
    omega_measured : np.ndarray
        Measured effective frequencies
    omega_theory : np.ndarray
        Theoretical predictions from Postulate 1
    tolerance : float
        Acceptable relative error
        
    Returns
    -------
    dict
        Residual statistics
    """
    abs_residuals = omega_measured - omega_theory
    rel_residuals = abs_residuals / (np.abs(omega_theory) + 1e-10)
    
    return {
        "max_abs_residual": float(np.max(np.abs(abs_residuals))),
        "mean_abs_residual": float(np.mean(np.abs(abs_residuals))),
        "rms_residual": float(np.sqrt(np.mean(abs_residuals**2))),
        "max_rel_residual": float(np.max(np.abs(rel_residuals))),
        "mean_rel_residual": float(np.mean(np.abs(rel_residuals))),
        "within_tolerance": float(np.mean(np.abs(rel_residuals) < tolerance)),
        "is_satisfied": np.max(np.abs(rel_residuals)) < tolerance
    }


def run_srqid_validation_suite(
    qca: "ExactQCA",
    state: np.ndarray,
    evolution_times: np.ndarray,
    check_sites: Optional[List[int]] = None
) -> Dict[str, Dict]:
    """
    Run full SRQID validation suite.
    
    Parameters
    ----------
    qca : ExactQCA
        QCA instance
    state : np.ndarray
        State to validate
    evolution_times : np.ndarray
        Evolution times
    check_sites : list, optional
        Sites to check (default: all)
        
    Returns
    -------
    dict
        All validation results
    """
    if check_sites is None:
        check_sites = list(range(qca.N))
    
    results = {}
    
    # 1. Energy conservation
    energies = []
    for t in evolution_times:
        evolved = qca.evolve(state, t)
        # Energy = <ψ|H|ψ>
        energy = np.real(np.vdot(evolved, qca.H @ evolved))
        energies.append(energy)
    
    results["energy"] = check_energy_drift(np.array(energies))
    
    # 2. No-signaling (measure Z at all sites)
    Z_expectations = {site: [] for site in check_sites}
    for t in evolution_times[:5]:  # Sample first few times
        evolved = qca.evolve(state, t)
        for site in check_sites:
            Z = qca.measure_Z(evolved, site)
            Z_expectations[site].append(Z)
    
    results["no_signaling"] = check_no_signaling(
        Z_expectations, evolution_times[:5], perturbation_site=0
    )
    
    # 3. Placeholder for LR velocity (would need commutator computations)
    results["lieb_robinson"] = {
        "status": "requires_commutator_data",
        "note": "Compute ||[X_i(t), X_j]|| for various separations"
    }
    
    return results


def summarize_validation(
    results: Dict[str, Dict]
) -> Dict[str, any]:
    """
    Create summary of validation results.
    
    Parameters
    ----------
    results : dict
        Output from run_srqid_validation_suite()
        
    Returns
    -------
    dict
        Summary with pass/fail status
    """
    summary = {
        "all_passed": True,
        "checks": {},
        "warnings": [],
        "errors": []
    }
    
    # Energy
    if "energy" in results:
        e = results["energy"]
        passed = e.get("is_conserved", False)
        summary["checks"]["energy"] = "PASS" if passed else "FAIL"
        summary["all_passed"] = summary["all_passed"] and passed
        if not passed:
            summary["errors"].append(f"Energy drift: {e.get('drift_relative', 0):.2e}")
    
    # No-signaling
    if "no_signaling" in results:
        ns = results["no_signaling"]
        passed = ns.get("is_satisfied", False)
        summary["checks"]["no_signaling"] = "PASS" if passed else "FAIL"
        summary["all_passed"] = summary["all_passed"] and passed
        if not passed:
            summary["errors"].append(f"No-signaling violation: {ns.get('max_violation', 0):.2e}")
    
    # Lieb-Robinson
    if "lieb_robinson" in results:
        lr = results["lieb_robinson"]
        if "is_satisfied" in lr:
            passed = lr["is_satisfied"]
            summary["checks"]["lieb_robinson"] = "PASS" if passed else "FAIL"
            summary["all_passed"] = summary["all_passed"] and passed
        else:
            summary["checks"]["lieb_robinson"] = "SKIPPED"
    
    return summary


if __name__ == "__main__":
    # Test validation checks
    print("SRQID Validation Tests")
    print("=" * 50)
    
    # Test energy drift
    energies = np.ones(100) + 1e-16 * np.random.randn(100)
    energy_check = check_energy_drift(energies)
    print(f"Energy conservation:")
    print(f"  Drift: {energy_check['drift_relative']:.2e}")
    print(f"  Pass: {energy_check['is_conserved']}")
    
    # Test no-signaling
    Z_expectations = {
        0: [1.0, 0.9, 0.8],  # Changed (perturbation site)
        1: [0.0, 1e-16, -1e-16],  # Unchanged
        2: [0.0, 0.0, 0.0]  # Unchanged
    }
    ns_check = check_no_signaling(Z_expectations, np.array([0, 1, 2]), 0)
    print(f"\nNo-signaling:")
    print(f"  Max violation: {ns_check['max_violation']:.2e}")
    print(f"  Pass: {ns_check['is_satisfied']}")
    
    # Test Postulate 1 residuals
    omega_theory = np.array([1.0, 0.5, 0.5, 1.0])
    omega_measured = omega_theory * 1.03  # 3% error
    p1_check = check_postulate1_residuals(omega_measured, omega_theory)
    print(f"\nPostulate 1 residuals:")
    print(f"  Mean rel error: {p1_check['mean_rel_residual']:.2%}")
    print(f"  Within tolerance: {p1_check['within_tolerance']:.1%}")
