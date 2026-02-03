"""
Master derivation coordinator: Computes all parameters for a given λ.

This module coordinates the derivation engine, computing all parameters
from first principles and logging steps to the database.
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from .anchoring import compute_all_scales
from .perturbative import compute_perturbative_parameters
from .frustration import compute_hotspot_simple, estimate_frustration_timescale
from .newtonian import compute_kappa_from_Geff, compute_target_degree


@dataclass
class DerivedParameters:
    """Container for all theory-derived parameters."""
    
    # Scale anchoring
    lattice_spacing: float
    time_step: float
    UV_energy: float
    
    # Perturbative
    alpha: float
    deltaB: float
    validity_ratio: float
    is_valid: bool
    
    # Frustration protocol
    hotspot_multiplier: float
    frustration_timescale: float
    
    # Newtonian
    kappa: float
    k0: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lattice_spacing": self.lattice_spacing,
            "time_step": self.time_step,
            "UV_energy": self.UV_energy,
            "alpha": self.alpha,
            "deltaB": self.deltaB,
            "validity_ratio": self.validity_ratio,
            "is_valid": self.is_valid,
            "hotspot_multiplier": self.hotspot_multiplier,
            "frustration_timescale": self.frustration_timescale,
            "kappa": self.kappa,
            "k0": self.k0,
        }


class DerivationCoordinator:
    """
    Coordinates parameter derivation for a simulation run.
    
    Usage:
        coordinator = DerivationCoordinator(run_id, db_session)
        params = coordinator.derive_all(lambda_val=0.5)
    """
    
    def __init__(
        self,
        run_id: int,
        db_session,
        topology: str = "path",
        N: int = 4,
        chi_max: int = 4,
        G_eff: float = 6.674e-11,
        c: float = 1.0,
        omega: float = 1.0,
        lambda_max: float = 1.5,
    ):
        self.run_id = run_id
        self.db = db_session
        
        # Ground truth inputs
        self.topology = topology
        self.N = N
        self.chi_max = chi_max
        self.G_eff = G_eff
        self.c = c
        self.omega = omega
        self.lambda_max = lambda_max
        
        # Step counter for logging
        self.step_number = 0
    
    def log_step(
        self,
        parameter: str,
        formula: str,
        inputs: Dict[str, Any],
        output: float,
        steps: list
    ):
        """Log a derivation step to the database."""
        from ..models import DerivationStep
        
        self.step_number += 1
        
        step = DerivationStep(
            run_id=self.run_id,
            step_number=self.step_number,
            parameter_name=parameter,
            formula_used=formula,
            inputs=inputs,
            output_value=output,
            intermediate_steps="\n".join(steps)
        )
        
        self.db.add(step)
        self.db.commit()
    
    def derive_scales(self) -> Dict[str, float]:
        """Derive scale anchoring parameters (once per run)."""
        scales = compute_all_scales(
            chi_max=self.chi_max,
            c_lattice=1.0 / (2**0.5),  # ~1/√2 for QATNU
            c_physical=self.c,
            log_callback=lambda entry: self.log_step(
                entry["parameter"],
                entry["formula"],
                entry["inputs"],
                entry["output"],
                entry["steps"]
            )
        )
        
        return scales
    
    def derive_at_lambda(self, lambda_val: float) -> DerivedParameters:
        """
        Derive all parameters at a specific λ value.
        
        Parameters
        ----------
        lambda_val : float
            Current promotion strength
            
        Returns
        -------
        DerivedParameters
            All derived parameters
        """
        # 1. Scale anchoring (computed once, but include for completeness)
        scales = self.derive_scales()
        
        # 2. Perturbative parameters
        pert = compute_perturbative_parameters(
            lambda_val=lambda_val,
            omega=self.omega,
            lambda_max=self.lambda_max,
            avg_frustration=0.5,  # Could be derived from topology
            log_callback=lambda entry: self.log_step(
                entry["parameter"],
                entry["formula"],
                entry["inputs"],
                entry["output"],
                entry["steps"]
            )
        )
        
        # 3. Hotspot multiplier
        hotspot = compute_hotspot_simple(
            lambda_val=lambda_val,
            chi_max=self.chi_max,
            log_callback=lambda entry: self.log_step(
                entry["parameter"],
                entry["formula"],
                entry["inputs"],
                entry["output"],
                entry["steps"]
            )
        )
        
        # 4. Frustration timescale
        t_scale = estimate_frustration_timescale(
            lambda_val=lambda_val,
            omega=self.omega,
            log_callback=lambda entry: self.log_step(
                entry["parameter"],
                entry["formula"],
                entry["inputs"],
                entry["output"],
                entry["steps"]
            )
        )
        
        # 5. Newtonian κ
        kappa = compute_kappa_from_Geff(
            G_eff=self.G_eff,
            alpha=pert["alpha_pert"],
            c=self.c,
            a=scales["lattice_spacing"],
            log_callback=lambda entry: self.log_step(
                entry["parameter"],
                entry["formula"],
                entry["inputs"],
                entry["output"],
                entry["steps"]
            )
        )
        
        # 6. Target degree
        k0 = compute_target_degree(
            topology=self.topology,
            N=self.N,
            log_callback=lambda entry: self.log_step(
                entry["parameter"],
                entry["formula"],
                entry["inputs"],
                entry["output"],
                entry["steps"]
            )
        )
        
        return DerivedParameters(
            lattice_spacing=scales["lattice_spacing"],
            time_step=scales["time_step"],
            UV_energy=scales["UV_energy"],
            alpha=pert["alpha_pert"],
            deltaB=pert["deltaB"],
            validity_ratio=pert["validity_ratio"],
            is_valid=pert["is_valid"],
            hotspot_multiplier=hotspot,
            frustration_timescale=t_scale,
            kappa=kappa,
            k0=k0
        )
    
    def validate_derivations(self, params: DerivedParameters) -> Dict[str, Any]:
        """
        Check if derived parameters are self-consistent.
        
        Returns warnings if any checks fail.
        """
        warnings = []
        
        # Check perturbative validity
        if not params.is_valid:
            warnings.append(
                f"Perturbative validity marginal: Δ/λ = {params.validity_ratio:.2f}"
            )
        
        # Check κ is positive
        if params.kappa <= 0:
            warnings.append(f"κ = {params.kappa:.6e} is non-positive!")
        
        # Check hotspot is reasonable
        if params.hotspot_multiplier < 1.5 or params.hotspot_multiplier > 5.0:
            warnings.append(
                f"Hotspot M = {params.hotspot_multiplier:.2f} outside typical range [1.5, 5.0]"
            )
        
        return {
            "is_valid": len(warnings) == 0,
            "warnings": warnings
        }


def derive_all_parameters(
    run_id: int,
    db_session,
    lambda_val: float,
    topology: str = "path",
    N: int = 4,
    chi_max: int = 4,
    G_eff: float = 6.674e-11,
    c: float = 1.0,
    omega: float = 1.0,
    lambda_max: float = 1.5,
) -> DerivedParameters:
    """
    Convenience function: Derive all parameters at once.
    
    Parameters
    ----------
    run_id : int
        Database run ID for logging
    db_session : Session
        SQLAlchemy database session
    lambda_val : float
        Current promotion strength
    topology, N, chi_max, G_eff, c, omega, lambda_max
        Ground truth inputs
        
    Returns
    -------
    DerivedParameters
        All derived parameters
    """
    coordinator = DerivationCoordinator(
        run_id=run_id,
        db_session=db_session,
        topology=topology,
        N=N,
        chi_max=chi_max,
        G_eff=G_eff,
        c=c,
        omega=omega,
        lambda_max=lambda_max
    )
    
    return coordinator.derive_at_lambda(lambda_val)


if __name__ == "__main__":
    # Test the coordinator
    from ..database import get_session
    from ..models import Run
    
    print("Derivation Coordinator Test")
    print("=" * 50)
    
    # Create a test run
    with get_session() as db:
        run = Run(
            status="testing",
            N=4,
            topology="path",
            chi_max=4,
            G_eff=6.674e-11,
            c=1.0,
            omega=1.0
        )
        db.add(run)
        db.commit()
        
        print(f"Test run ID: {run.id}")
        
        # Derive parameters at λ = 0.5
        coordinator = DerivationCoordinator(
            run_id=run.id,
            db_session=db,
            topology="path",
            N=4,
            chi_max=4
        )
        
        params = coordinator.derive_at_lambda(lambda_val=0.5)
        
        print(f"\nDerived Parameters at λ=0.5:")
        for key, value in params.to_dict().items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # Check consistency
        checks = coordinator.validate_derivations(params)
        print(f"\nValidation: {'✓' if checks['is_valid'] else '✗'}")
        if checks['warnings']:
            for w in checks['warnings']:
                print(f"  ⚠ {w}")
        
        # Cleanup
        db.delete(run)
        db.commit()
    
    print("\nTest complete!")
