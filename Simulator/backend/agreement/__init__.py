"""
Agreement calculation package: Theory vs measurement quality assessment.

This package provides comprehensive tools for quantifying the agreement
between QATNU theory predictions and simulation measurements, including
statistical metrics, configurable weighting, and letter-grade scoring.

Modules:
- metrics: Statistical agreement metrics (MAE, RMSE, correlation, R², etc.)
- weights: Configurable weighting schemes and grade scales
- calculator: Main calculator interface for computing agreement

Usage:
    from backend.agreement import AgreementCalculator, PhysicsPriority
    
    calculator = AgreementCalculator(PhysicsPriority.BALANCED)
    report = calculator.compute_run_agreement(run_id, comparisons)
    print(f"Overall grade: {report.overall_grade}")
"""

from .metrics import (
    MetricResult,
    mean_absolute_error,
    root_mean_square_error,
    pearson_correlation,
    spearman_correlation,
    max_absolute_deviation,
    coefficient_of_determination,
    relative_bias,
    variance_ratio,
    compute_all_metrics,
    compute_agreement_grade,
)

from .weights import (
    PhysicsPriority,
    WeightConfig,
    GradeScale,
    get_weight_config,
    customize_weights,
    BALANCED_CONFIG,
    FREQUENCY_PRIORITY_CONFIG,
    LAMBDA_PRIORITY_CONFIG,
    SRQID_PRIORITY_CONFIG,
    CRITICAL_POINT_CONFIG,
)

from .calculator import (
    QuantityAgreement,
    RunAgreementReport,
    AgreementCalculator,
    compare_theory_measurement,
)

__all__ = [
    # Metrics
    "MetricResult",
    "mean_absolute_error",
    "root_mean_square_error",
    "pearson_correlation",
    "spearman_correlation",
    "max_absolute_deviation",
    "coefficient_of_determination",
    "relative_bias",
    "variance_ratio",
    "compute_all_metrics",
    "compute_agreement_grade",
    # Weights
    "PhysicsPriority",
    "WeightConfig",
    "GradeScale",
    "get_weight_config",
    "customize_weights",
    "BALANCED_CONFIG",
    "FREQUENCY_PRIORITY_CONFIG",
    "LAMBDA_PRIORITY_CONFIG",
    "SRQID_PRIORITY_CONFIG",
    "CRITICAL_POINT_CONFIG",
    # Calculator
    "QuantityAgreement",
    "RunAgreementReport",
    "AgreementCalculator",
    "compare_theory_measurement",
]
