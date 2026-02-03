"""
Agreement calculator: Compute overall theory-simulation agreement.

This is the main interface for calculating how well theory predictions
match simulation measurements, producing a final grade and detailed report.
"""

import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..extraction.coordinator import LambdaPointComparison

from .metrics import compute_all_metrics, compute_agreement_grade, MetricResult
from .weights import WeightConfig, PhysicsPriority, get_weight_config, GradeScale


@dataclass
class QuantityAgreement:
    """Agreement results for a single quantity (e.g., Λ or ω_eff)."""
    
    name: str
    metrics: Dict[str, MetricResult]
    grade: str
    score: float
    is_acceptable: bool
    theory_mean: float
    measured_mean: float
    relative_difference: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "grade": self.grade,
            "score": self.score,
            "acceptable": self.is_acceptable,
            "theory_mean": self.theory_mean,
            "measured_mean": self.measured_mean,
            "relative_diff": self.relative_difference,
            "metrics": {k: {
                "value": v.value,
                "normalized": v.normalized,
                "acceptable": v.is_acceptable
            } for k, v in self.metrics.items()}
        }


@dataclass
class RunAgreementReport:
    """Complete agreement report for a simulation run."""
    
    run_id: int
    overall_grade: str
    overall_score: float
    
    # Per-quantity agreement
    quantities: Dict[str, QuantityAgreement]
    
    # Per-lambda-point results
    lambda_points: List[Dict[str, Any]] = field(default_factory=list)
    
    # SRQID validation status
    srqid_passed: bool = True
    srqid_failures: List[str] = field(default_factory=list)
    
    # Critical points detected
    critical_points: Dict[str, float] = field(default_factory=dict)
    
    # Configuration used
    weight_config: Dict[str, Any] = field(default_factory=dict)
    
    # Summary statistics
    num_points: int = 0
    lambda_range: tuple = (0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "overall_grade": self.overall_grade,
            "overall_score": self.overall_score,
            "quantities": {k: v.to_dict() for k, v in self.quantities.items()},
            "lambda_points": self.lambda_points,
            "srqid": {
                "passed": self.srqid_passed,
                "failures": self.srqid_failures
            },
            "critical_points": self.critical_points,
            "statistics": {
                "num_points": self.num_points,
                "lambda_range": self.lambda_range
            }
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"Agreement Report for Run {self.run_id}",
            "=" * 60,
            f"Overall: Grade {self.overall_grade} (Score: {self.overall_score:.3f})",
            "",
            "Per-Quantity Agreement:",
        ]
        
        for name, qty in self.quantities.items():
            status = "✓" if qty.is_acceptable else "✗"
            lines.append(f"  {status} {name:15s}: Grade {qty.grade} (Score: {qty.score:.3f})")
        
        lines.extend([
            "",
            f"SRQID Validation: {'PASSED' if self.srqid_passed else 'FAILED'}",
        ])
        
        if self.critical_points:
            lines.append(f"Critical Points: {self.critical_points}")
        
        lines.extend([
            "",
            f"Lambda scan: {self.num_points} points from {self.lambda_range[0]:.2f} to {self.lambda_range[1]:.2f}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class AgreementCalculator:
    """
    Main calculator for theory-simulation agreement.
    
    Usage:
        calculator = AgreementCalculator(PhysicsPriority.BALANCED)
        report = calculator.compute_run_agreement(run_data)
    """
    
    def __init__(
        self,
        priority: PhysicsPriority = PhysicsPriority.BALANCED,
        custom_config: Optional[WeightConfig] = None
    ):
        """
        Initialize calculator.
        
        Parameters
        ----------
        priority : PhysicsPriority
            Weighting priority scheme
        custom_config : WeightConfig, optional
            Override with custom configuration
        """
        self.config = custom_config or get_weight_config(priority)
        self.priority = priority
    
    def compute_quantity_agreement(
        self,
        name: str,
        theory: np.ndarray,
        measured: np.ndarray
    ) -> QuantityAgreement:
        """
        Compute agreement for a single quantity.
        
        Parameters
        ----------
        name : str
            Quantity name (e.g., "Lambda", "omega_eff")
        theory : np.ndarray
            Theoretical predictions
        measured : np.ndarray
            Measured values
            
        Returns
        -------
        QuantityAgreement
            Agreement results
        """
        # Compute all metrics
        metrics = compute_all_metrics(theory, measured, normalize=True)
        
        # Filter to weighted metrics
        weighted_metrics = {
            k: metrics[k] for k in self.config.metric_weights.keys()
            if k in metrics
        }
        
        # Compute grade and score
        grade, score = compute_agreement_grade(
            weighted_metrics,
            self.config.metric_weights
        )
        
        # Check acceptance
        is_acceptable = score >= self.config.grade_thresholds.get("B", 0.8)
        
        return QuantityAgreement(
            name=name,
            metrics=metrics,
            grade=grade,
            score=score,
            is_acceptable=is_acceptable,
            theory_mean=float(np.mean(theory)),
            measured_mean=float(np.mean(measured)),
            relative_difference=float(
                (np.mean(measured) - np.mean(theory)) / (np.mean(theory) + 1e-10)
            )
        )
    
    def compute_lambda_point_agreement(
        self,
        comparison: "LambdaPointComparison"
    ) -> Dict[str, Any]:
        """
        Compute agreement for a single lambda point.
        
        Parameters
        ----------
        comparison : LambdaPointComparison
            Comparison data from measurement coordinator
            
        Returns
        -------
        dict
            Agreement results for this point
        """
        results = {
            "lambda": comparison.lambda_val,
            "quantities": {}
        }
        
        # Lambda agreement
        if comparison.Lambda_measured is not None:
            lambda_agreement = self.compute_quantity_agreement(
                "Lambda",
                comparison.Lambda_theory,
                comparison.Lambda_measured
            )
            results["quantities"]["Lambda"] = lambda_agreement.to_dict()
        
        # Omega_eff agreement
        if comparison.omega_eff_measured is not None:
            omega_agreement = self.compute_quantity_agreement(
                "omega_eff",
                comparison.omega_eff_theory,
                comparison.omega_eff_measured
            )
            results["quantities"]["omega_eff"] = omega_agreement.to_dict()
        
        # Overall point score (weighted by quantity weights)
        scores = []
        weights = []
        for qty_name, qty_data in results["quantities"].items():
            scores.append(qty_data["score"])
            weights.append(self.config.quantity_weights.get(qty_name, 0.1))
        
        if scores:
            results["point_score"] = np.average(scores, weights=weights)
            results["point_grade"] = GradeScale.get_grade(results["point_score"])
        
        return results
    
    def compute_run_agreement(
        self,
        run_id: int,
        lambda_comparisons: List["LambdaPointComparison"],
        critical_points: Optional[Dict[str, float]] = None
    ) -> RunAgreementReport:
        """
        Compute overall agreement for a complete run.
        
        Parameters
        ----------
        run_id : int
            Run identifier
        lambda_comparisons : list
            List of per-lambda comparisons
        critical_points : dict, optional
            Detected critical points
            
        Returns
        -------
        RunAgreementReport
            Complete agreement report
        """
        if not lambda_comparisons:
            return RunAgreementReport(
                run_id=run_id,
                overall_grade="F",
                overall_score=0.0,
                quantities={}
            )
        
        # Collect all theory and measured values
        all_lambda_theory = []
        all_lambda_measured = []
        all_omega_theory = []
        all_omega_measured = []
        
        lambda_point_results = []
        srqid_failures = []
        
        for comp in lambda_comparisons:
            # Collect arrays
            all_lambda_theory.extend(comp.Lambda_theory)
            if comp.Lambda_measured is not None:
                all_lambda_measured.extend(comp.Lambda_measured)
            
            all_omega_theory.extend(comp.omega_eff_theory)
            if comp.omega_eff_measured is not None:
                all_omega_measured.extend(comp.omega_eff_measured)
            
            # Per-point agreement
            point_result = self.compute_lambda_point_agreement(comp)
            
            # Critical point boost
            if critical_points:
                for cp_name, cp_lambda in critical_points.items():
                    if abs(comp.lambda_val - cp_lambda) < self.config.critical_region_width:
                        point_result["point_score"] *= self.config.critical_point_boost
                        point_result["point_score"] = min(1.0, point_result["point_score"])
                        point_result["point_grade"] = GradeScale.get_grade(point_result["point_score"])
                        point_result["critical_boost"] = True
            
            lambda_point_results.append(point_result)
            
            # SRQID check failures
            if comp.srqid_checks:
                for check_name, check_result in comp.srqid_checks.items():
                    if isinstance(check_result, dict) and not check_result.get("is_satisfied", True):
                        srqid_failures.append(f"λ={comp.lambda_val:.2f}:{check_name}")
        
        # Compute quantity agreements
        quantities = {}
        
        if all_lambda_measured:
            quantities["Lambda"] = self.compute_quantity_agreement(
                "Lambda",
                np.array(all_lambda_theory),
                np.array(all_lambda_measured)
            )
        
        if all_omega_measured:
            quantities["omega_eff"] = self.compute_quantity_agreement(
                "omega_eff",
                np.array(all_omega_theory),
                np.array(all_omega_measured)
            )
        
        # Overall score (weighted by quantity)
        overall_scores = []
        overall_weights = []
        
        for qty_name, qty_agreement in quantities.items():
            overall_scores.append(qty_agreement.score)
            overall_weights.append(self.config.quantity_weights.get(qty_name, 0.1))
        
        # Include point scores
        point_scores = [p["point_score"] for p in lambda_point_results if "point_score" in p]
        if point_scores:
            overall_scores.append(np.mean(point_scores))
            overall_weights.append(0.2)  # Lambda scan consistency weight
        
        if overall_scores:
            overall_score = np.average(overall_scores, weights=overall_weights)
        else:
            overall_score = 0.0
        
        overall_grade = GradeScale.get_grade(overall_score)
        
        # Adjust grade if SRQID failed
        srqid_passed = len(srqid_failures) == 0
        if not srqid_passed and overall_grade in ["A+", "A", "B+"]:
            # Downgrade if SRQID checks fail
            grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]
            idx = grade_order.index(overall_grade)
            overall_grade = grade_order[min(len(grade_order)-1, idx + 1)]
        
        # Lambda range
        lambda_vals = [c.lambda_val for c in lambda_comparisons]
        
        return RunAgreementReport(
            run_id=run_id,
            overall_grade=overall_grade,
            overall_score=float(overall_score),
            quantities=quantities,
            lambda_points=lambda_point_results,
            srqid_passed=srqid_passed,
            srqid_failures=srqid_failures,
            critical_points=critical_points or {},
            weight_config=self.config.to_dict(),
            num_points=len(lambda_comparisons),
            lambda_range=(min(lambda_vals), max(lambda_vals))
        )
    
    def quick_assessment(
        self,
        theory: np.ndarray,
        measured: np.ndarray
    ) -> str:
        """
        Quick grade assessment (convenience method).
        
        Parameters
        ----------
        theory : np.ndarray
            Theory predictions
        measured : np.ndarray
            Measured values
            
        Returns
        -------
        str
            Grade letter
        """
        agreement = self.compute_quantity_agreement("quick", theory, measured)
        return f"Grade {agreement.grade} (Score: {agreement.score:.3f})"


def compare_theory_measurement(
    theory: np.ndarray,
    measured: np.ndarray,
    priority: PhysicsPriority = PhysicsPriority.BALANCED
) -> Dict[str, Any]:
    """
    Standalone comparison function.
    
    Parameters
    ----------
    theory : np.ndarray
        Theory predictions
    measured : np.ndarray
        Measured values
    priority : PhysicsPriority
        Weighting scheme
        
    Returns
    -------
    dict
        Full comparison results
    """
    calculator = AgreementCalculator(priority)
    agreement = calculator.compute_quantity_agreement("quantity", theory, measured)
    return agreement.to_dict()


if __name__ == "__main__":
    # Test agreement calculator
    print("Agreement Calculator Test")
    print("=" * 60)
    
    # Test data: good agreement
    theory = np.linspace(0, 1, 20)
    measured = theory + 0.02 * np.random.randn(20)  # Small noise
    
    # Test with different priorities
    for priority in PhysicsPriority:
        print(f"\n{priority.value.upper()}:")
        calculator = AgreementCalculator(priority)
        agreement = calculator.compute_quantity_agreement("test", theory, measured)
        print(f"  Grade: {agreement.grade} (Score: {agreement.score:.3f})")
        print(f"  Top metrics: pearson={agreement.metrics['pearson'].value:.3f}, "
              f"mae={agreement.metrics['mae'].value:.3f}")
    
    # Test poor agreement
    print("\n" + "=" * 60)
    print("Poor agreement test:")
    measured_bad = theory * 1.5 + 0.5  # Large systematic error
    calculator = AgreementCalculator(PhysicsPriority.BALANCED)
    agreement_bad = calculator.compute_quantity_agreement("bad", theory, measured_bad)
    print(f"  Grade: {agreement_bad.grade} (Score: {agreement_bad.score:.3f})")
    print(f"  Bias: {agreement_bad.relative_difference:.2%}")
    
    # Test quick assessment
    print("\n" + "=" * 60)
    print("Quick assessment:")
    result = calculator.quick_assessment(theory, measured)
    print(f"  {result}")
