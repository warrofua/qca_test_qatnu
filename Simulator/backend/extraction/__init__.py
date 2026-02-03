"""
Measurement extraction package: Theory vs simulation comparison.

This package provides tools to extract physical quantities from both
theoretical predictions and measured quantum states, enabling direct
comparison between QATNU theory and exact diagonalization simulations.

Modules:
- lambda: Bubble density (Λ) extraction
- frequency: Effective frequency (ω_eff) extraction
- validation: SRQID consistency checks
- coordinator: Unified measurement interface
"""

from .bubble_density import (
    extract_lambda_theory,
    extract_lambda_measured,
    compute_lambda_discrepancy,
    extract_lambda_total,
    extract_lambda_gradient,
)

from .frequency import (
    extract_omega_eff_theory,
    extract_omega_eff_ramsey,
    extract_omega_eff_from_Z_dynamics,
    extract_omega_eff_measured_direct,
    compute_omega_eff_discrepancy,
    extract_global_omega_eff,
)

from .validation import (
    check_lieb_robinson_velocity,
    check_no_signaling,
    check_energy_drift,
    check_entropy_scaling,
    check_postulate1_residuals,
    run_srqid_validation_suite,
    summarize_validation,
)

__all__ = [
    # Lambda extraction
    "extract_lambda_theory",
    "extract_lambda_measured",
    "compute_lambda_discrepancy",
    "extract_lambda_total",
    "extract_lambda_gradient",
    # Frequency extraction
    "extract_omega_eff_theory",
    "extract_omega_eff_ramsey",
    "extract_omega_eff_from_Z_dynamics",
    "extract_omega_eff_measured_direct",
    "compute_omega_eff_discrepancy",
    "extract_global_omega_eff",
    # Validation
    "check_lieb_robinson_velocity",
    "check_no_signaling",
    "check_energy_drift",
    "check_entropy_scaling",
    "check_postulate1_residuals",
    "run_srqid_validation_suite",
    "summarize_validation",
]
