"""
First principles derivation engine for QATNU parameters.

All parameters are derived from theory, not fitted:
- Scale anchoring: a, τ from χ_max and c
- Perturbative: α(λ), δB from validity constraints  
- Frustration: hotspot from ⟨F⟩ target
- Newtonian: κ from G_eff
"""

from .anchoring import (
    compute_lattice_spacing,
    compute_time_step,
    compute_UV_energy_scale,
    compute_all_scales
)

from .perturbative import (
    compute_alpha_perturbative,
    compute_deltaB,
    compute_alpha_from_gap_relation,
    compute_perturbative_parameters
)

from .frustration import (
    compute_hotspot_multiplier,
    compute_hotspot_simple,
    estimate_frustration_timescale,
    measure_frustration
)

from .newtonian import (
    compute_kappa_from_Geff,
    compute_effective_G,
    compute_target_degree,
    check_newtonian_consistency
)

__all__ = [
    # Anchoring
    "compute_lattice_spacing",
    "compute_time_step",
    "compute_UV_energy_scale",
    "compute_all_scales",
    
    # Perturbative
    "compute_alpha_perturbative",
    "compute_deltaB",
    "compute_alpha_from_gap_relation",
    "compute_perturbative_parameters",
    
    # Frustration
    "compute_hotspot_multiplier",
    "compute_hotspot_simple",
    "estimate_frustration_timescale",
    "measure_frustration",
    
    # Newtonian
    "compute_kappa_from_Geff",
    "compute_effective_G",
    "compute_target_degree",
    "check_newtonian_consistency",
]
