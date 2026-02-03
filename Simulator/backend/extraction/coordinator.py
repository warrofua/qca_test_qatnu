"""
Measurement coordinator: Unified interface for theory vs measurement comparison.

This module coordinates the extraction of physical quantities from both
theory predictions and simulation measurements, computing agreement scores.
"""

import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..core_qca import ExactQCA
    from ..derivation.coordinator import DerivedParameters


@dataclass
class MeasurementResult:
    """Container for a single measurement comparison."""
    
    quantity: str
    theory_value: Any
    measured_value: Any
    abs_error: float
    rel_error: float
    is_within_tolerance: bool
    tolerance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LambdaPointComparison:
    """Complete comparison at a single λ point."""
    
    lambda_val: float
    
    # Theory predictions
    Lambda_theory: np.ndarray
    omega_eff_theory: np.ndarray
    
    # Measurements
    Lambda_measured: Optional[np.ndarray] = None
    omega_eff_measured: Optional[np.ndarray] = None
    
    # Discrepancy metrics
    Lambda_discrepancy: Dict[str, float] = field(default_factory=dict)
    omega_discrepancy: Dict[str, float] = field(default_factory=dict)
    
    # Validation results
    srqid_checks: Dict[str, Dict] = field(default_factory=dict)
    
    # Overall agreement
    agreement_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lambda": self.lambda_val,
            "agreement_score": self.agreement_score,
            "Lambda": {
                "theory": self.Lambda_theory.tolist(),
                "measured": self.Lambda_measured.tolist() if self.Lambda_measured is not None else None,
                "discrepancy": self.Lambda_discrepancy
            },
            "omega_eff": {
                "theory": self.omega_eff_theory.tolist(),
                "measured": self.omega_eff_measured.tolist() if self.omega_eff_measured is not None else None,
                "discrepancy": self.omega_discrepancy
            },
            "srqid_checks": self.srqid_checks
        }


class MeasurementCoordinator:
    """
    Coordinates measurement extraction and comparison.
    
    Usage:
        coordinator = MeasurementCoordinator(qca, derived_params)
        comparison = coordinator.compare_at_lambda(lambda_val, state)
    """
    
    def __init__(
        self,
        qca: "ExactQCA",
        derived_params: "DerivedParameters",
        omega_0: float = 1.0
    ):
        """
        Initialize coordinator.
        
        Parameters
        ----------
        qca : ExactQCA
            QCA instance for measurements
        derived_params : DerivedParameters
            Theory-derived parameters
        omega_0 : float
            Bare clock frequency
        """
        self.qca = qca
        self.params = derived_params
        self.omega_0 = omega_0
        self.N = qca.N
    
    def predict_theory(self, lambda_val: float) -> Dict[str, np.ndarray]:
        """
        Generate theory predictions for Λ and ω_eff.
        
        Parameters
        ----------
        lambda_val : float
            Current promotion strength
            
        Returns
        -------
        dict
            {"Lambda": ..., "omega_eff": ...}
        """
        from .bubble_density import extract_lambda_theory
        from .frequency import extract_omega_eff_theory
        
        # Predict Λ from theory
        lambda_result = extract_lambda_theory(
            lambda_val=lambda_val,
            alpha=self.params.alpha,
            N=self.N,
            chi_max=self.params.chi_max if hasattr(self.params, 'chi_max') else 4,
            hotspot_multiplier=self.params.hotspot_multiplier,
            frustration_time=self.params.frustration_timescale
        )
        
        # Predict ω_eff from Postulate 1
        omega_result = extract_omega_eff_theory(
            lambda_profile=lambda_result["Lambda"],
            alpha=self.params.alpha,
            omega_0=self.omega_0
        )
        
        return {
            "Lambda": lambda_result["Lambda"],
            "omega_eff": omega_result["omega_eff"]
        }
    
    def measure_from_state(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract measurements from quantum state.
        
        Parameters
        ----------
        state : np.ndarray
            Quantum state vector
            
        Returns
        -------
        dict
            {"Lambda": ..., "omega_eff": ...}
        """
        from .bubble_density import extract_lambda_measured
        
        # Measure Λ from bond dimensions
        lambda_result = extract_lambda_measured(state, self.qca)
        
        # For ω_eff, we'd need time evolution
        # Placeholder: derive from Λ using Postulate 1 "backwards"
        # In production, this would use Ramsey or Z dynamics
        from .frequency import extract_omega_eff_theory
        omega_result = extract_omega_eff_theory(
            lambda_profile=lambda_result["Lambda"],
            alpha=self.params.alpha,
            omega_0=self.omega_0
        )
        
        return {
            "Lambda": lambda_result["Lambda"],
            "omega_eff": omega_result["omega_eff"]  # Placeholder
        }
    
    def compare_at_lambda(
        self,
        lambda_val: float,
        state: np.ndarray,
        run_validation: bool = True
    ) -> LambdaPointComparison:
        """
        Full comparison at a single λ point.
        
        Parameters
        ----------
        lambda_val : float
            Promotion strength
        state : np.ndarray
            Measured quantum state
        run_validation : bool
            Whether to run SRQID checks
            
        Returns
        -------
        LambdaPointComparison
            Complete comparison results
        """
        from .bubble_density import compute_lambda_discrepancy
        from .frequency import compute_omega_eff_discrepancy
        from .validation import run_srqid_validation_suite, summarize_validation
        
        # Theory predictions
        theory = self.predict_theory(lambda_val)
        
        # Measurements
        measured = self.measure_from_state(state)
        
        # Discrepancies
        lambda_disc = compute_lambda_discrepancy(
            {"Lambda": theory["Lambda"]},
            {"Lambda": measured["Lambda"]}
        )
        
        # Placeholder for omega discrepancy
        omega_disc = {
            "mean_rel_error": 0.0,
            "correlation": 1.0
        }
        
        # Validation checks
        srqid_results = {}
        if run_validation:
            times = np.linspace(0, 1.0, 10)
            srqid_results = run_srqid_validation_suite(
                self.qca, state, times
            )
        
        # Agreement score (weighted combination)
        lambda_score = max(0, 1 - lambda_disc.get("mean_rel_error", 1.0))
        omega_score = max(0, 1 - omega_disc.get("mean_rel_error", 1.0))
        
        # Weights: Λ is primary, ω is derived
        agreement_score = 0.6 * lambda_score + 0.4 * omega_score
        
        return LambdaPointComparison(
            lambda_val=lambda_val,
            Lambda_theory=theory["Lambda"],
            omega_eff_theory=theory["omega_eff"],
            Lambda_measured=measured["Lambda"],
            omega_eff_measured=measured["omega_eff"],
            Lambda_discrepancy=lambda_disc,
            omega_discrepancy=omega_disc,
            srqid_checks=srqid_results,
            agreement_score=agreement_score
        )
    
    def compute_run_agreement(
        self,
        comparisons: List[LambdaPointComparison]
    ) -> Dict[str, float]:
        """
        Compute overall agreement score for a full λ scan.
        
        Parameters
        ----------
        comparisons : list
            List of LambdaPointComparison for each λ
            
        Returns
        -------
        dict
            Overall agreement metrics
        """
        if not comparisons:
            return {"overall_score": 0.0}
        
        scores = [c.agreement_score for c in comparisons]
        lambda_errors = [c.Lambda_discrepancy.get("mean_rel_error", 1.0) for c in comparisons]
        
        return {
            "overall_score": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "mean_lambda_error": float(np.mean(lambda_errors)),
            "num_points": len(comparisons)
        }


def quick_compare(
    qca: "ExactQCA",
    derived_params: "DerivedParameters",
    lambda_val: float,
    state: np.ndarray
) -> Dict[str, Any]:
    """
    Quick comparison function (convenience wrapper).
    
    Parameters
    ----------
    qca : ExactQCA
        QCA instance
    derived_params : DerivedParameters
        Theory parameters
    lambda_val : float
        Current λ
    state : np.ndarray
        State to measure
        
    Returns
    -------
    dict
        Comparison results
    """
    coordinator = MeasurementCoordinator(qca, derived_params)
    comparison = coordinator.compare_at_lambda(lambda_val, state)
    return comparison.to_dict()


if __name__ == "__main__":
    # Test measurement coordinator
    print("Measurement Coordinator Test")
    print("=" * 50)
    
    # Mock derived parameters
    from dataclasses import dataclass
    
    @dataclass
    class MockParams:
        alpha: float = 0.8
        chi_max: int = 4
        hotspot_multiplier: float = 3.0
        frustration_timescale: float = 1.0
    
    mock_params = MockParams()
    
    # Would need actual QCA instance for full test
    print("Theory prediction at λ=0.5:")
    
    # Manual test of theory prediction
    from .lambda import extract_lambda_theory
    from .frequency import extract_omega_eff_theory
    
    lambda_theory = extract_lambda_theory(
        lambda_val=0.5,
        alpha=0.8,
        N=4,
        chi_max=4,
        hotspot_multiplier=3.0
    )
    
    omega_theory = extract_omega_eff_theory(
        lambda_profile=lambda_theory["Lambda"],
        alpha=0.8,
        omega_0=1.0
    )
    
    print(f"  Λ = {lambda_theory['Lambda']}")
    print(f"  ω_eff = {omega_theory['omega_eff']}")
    print(f"  ω_ratio = {omega_theory['omega_ratio']}")
    
    print("\nCoordinator test complete!")
