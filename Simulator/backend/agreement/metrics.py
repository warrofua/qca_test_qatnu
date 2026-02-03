"""
Agreement metrics: Statistical measures of theory-measurement concordance.

Provides various metrics for quantifying how well theory predictions
match simulation measurements, with appropriate physics weighting.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for a single metric result."""
    value: float
    normalized: float  # 0-1 scale (1 = perfect agreement)
    is_acceptable: bool
    threshold: float
    description: str


def mean_absolute_error(
    theory: np.ndarray,
    measured: np.ndarray,
    normalize: bool = False
) -> MetricResult:
    """
    Compute Mean Absolute Error (MAE).
    
    Parameters
    ----------
    theory : np.ndarray
        Theoretical predictions
    measured : np.ndarray
        Measured values
    normalize : bool
        If True, normalize by theory mean
        
    Returns
    -------
    MetricResult
        MAE with normalized score
    """
    mae = np.mean(np.abs(measured - theory))
    
    if normalize and np.mean(np.abs(theory)) > 1e-10:
        mae_rel = mae / np.mean(np.abs(theory))
        normalized = max(0, 1 - mae_rel)
    else:
        normalized = max(0, 1 - mae)
    
    return MetricResult(
        value=float(mae),
        normalized=float(normalized),
        is_acceptable=normalized > 0.8,
        threshold=0.2,
        description="Mean Absolute Error"
    )


def root_mean_square_error(
    theory: np.ndarray,
    measured: np.ndarray,
    normalize: bool = False
) -> MetricResult:
    """
    Compute Root Mean Square Error (RMSE).
    
    More sensitive to outliers than MAE.
    
    Parameters
    ----------
    theory : np.ndarray
        Theoretical predictions
    measured : np.ndarray
        Measured values
    normalize : bool
        If True, normalize by theory std
        
    Returns
    -------
    MetricResult
        RMSE with normalized score
    """
    rmse = np.sqrt(np.mean((measured - theory) ** 2))
    
    if normalize and np.std(theory) > 1e-10:
        rmse_rel = rmse / np.std(theory)
        normalized = max(0, 1 - rmse_rel)
    else:
        normalized = max(0, 1 - rmse)
    
    return MetricResult(
        value=float(rmse),
        normalized=float(normalized),
        is_acceptable=normalized > 0.8,
        threshold=0.2,
        description="Root Mean Square Error"
    )


def pearson_correlation(
    theory: np.ndarray,
    measured: np.ndarray
) -> MetricResult:
    """
    Compute Pearson correlation coefficient.
    
    Measures linear correlation (-1 to 1, 1 = perfect).
    
    Parameters
    ----------
    theory : np.ndarray
        Theoretical predictions
    measured : np.ndarray
        Measured values
        
    Returns
    -------
    MetricResult
        Correlation with normalized score
    """
    if len(theory) < 2 or np.std(theory) < 1e-10 or np.std(measured) < 1e-10:
        corr = 0.0
    else:
        corr = np.corrcoef(theory, measured)[0, 1]
    
    # Normalize: 0 to 1 (handle negative correlations)
    normalized = (corr + 1) / 2 if not np.isnan(corr) else 0.0
    
    return MetricResult(
        value=float(corr),
        normalized=float(normalized),
        is_acceptable=corr > 0.9,
        threshold=0.9,
        description="Pearson Correlation"
    )


def spearman_correlation(
    theory: np.ndarray,
    measured: np.ndarray
) -> MetricResult:
    """
    Compute Spearman rank correlation.
    
    Measures monotonic relationship (robust to outliers).
    
    Parameters
    ----------
    theory : np.ndarray
        Theoretical predictions
    measured : np.ndarray
        Measured values
        
    Returns
    -------
    MetricResult
        Spearman correlation
    """
    from scipy import stats
    
    if len(theory) < 2:
        rho = 0.0
    else:
        rho, _ = stats.spearmanr(theory, measured)
        if np.isnan(rho):
            rho = 0.0
    
    normalized = (rho + 1) / 2
    
    return MetricResult(
        value=float(rho),
        normalized=float(normalized),
        is_acceptable=rho > 0.9,
        threshold=0.9,
        description="Spearman Rank Correlation"
    )


def max_absolute_deviation(
    theory: np.ndarray,
    measured: np.ndarray,
    normalize: bool = False
) -> MetricResult:
    """
    Compute maximum absolute deviation.
    
    Identifies worst-case discrepancy.
    
    Parameters
    ----------
    theory : np.ndarray
        Theoretical predictions
    measured : np.ndarray
        Measured values
    normalize : bool
        If True, normalize by theory range
        
    Returns
    -------
    MetricResult
        Max deviation
    """
    mad = np.max(np.abs(measured - theory))
    
    if normalize:
        theory_range = np.max(theory) - np.min(theory)
        if theory_range > 1e-10:
            mad_rel = mad / theory_range
            normalized = max(0, 1 - mad_rel)
        else:
            normalized = 1.0 if mad < 1e-10 else 0.0
    else:
        normalized = max(0, 1 - mad)
    
    return MetricResult(
        value=float(mad),
        normalized=float(normalized),
        is_acceptable=normalized > 0.9,
        threshold=0.1,
        description="Maximum Absolute Deviation"
    )


def coefficient_of_determination(
    theory: np.ndarray,
    measured: np.ndarray
) -> MetricResult:
    """
    Compute R² (coefficient of determination).
    
    R² = 1 - SS_res/SS_tot, where 1 is perfect fit.
    
    Parameters
    ----------
    theory : np.ndarray
        Theoretical predictions (treated as "true" for R²)
    measured : np.ndarray
        Measured values (treated as predictions)
        
    Returns
    -------
    MetricResult
        R² score
    """
    ss_res = np.sum((theory - measured) ** 2)
    ss_tot = np.sum((theory - np.mean(theory)) ** 2)
    
    if ss_tot < 1e-15:
        r2 = 1.0 if ss_res < 1e-15 else 0.0
    else:
        r2 = 1 - ss_res / ss_tot
    
    # Clip to [0, 1]
    r2 = max(0, min(1, r2))
    
    return MetricResult(
        value=float(r2),
        normalized=float(r2),
        is_acceptable=r2 > 0.9,
        threshold=0.9,
        description="R² (Coefficient of Determination)"
    )


def relative_bias(
    theory: np.ndarray,
    measured: np.ndarray
) -> MetricResult:
    """
    Compute relative bias (systematic offset).
    
    bias = (measured - theory) / theory
    
    Parameters
    ----------
    theory : np.ndarray
        Theoretical predictions
    measured : np.ndarray
        Measured values
        
    Returns
    -------
    MetricResult
        Relative bias
    """
    mean_theory = np.mean(theory)
    mean_measured = np.mean(measured)
    
    if abs(mean_theory) > 1e-10:
        bias = (mean_measured - mean_theory) / mean_theory
    else:
        bias = mean_measured - mean_theory
    
    # Normalized: 1 = no bias, 0 = large bias
    normalized = max(0, 1 - abs(bias))
    
    return MetricResult(
        value=float(bias),
        normalized=float(normalized),
        is_acceptable=abs(bias) < 0.1,
        threshold=0.1,
        description="Relative Bias"
    )


def variance_ratio(
    theory: np.ndarray,
    measured: np.ndarray
) -> MetricResult:
    """
    Compute variance ratio (measured/theory).
    
    Should be close to 1 for good agreement.
    
    Parameters
    ----------
    theory : np.ndarray
        Theoretical predictions
    measured : np.ndarray
        Measured values
        
    Returns
    -------
    MetricResult
        Variance ratio
    """
    var_theory = np.var(theory)
    var_measured = np.var(measured)
    
    if var_theory > 1e-10:
        ratio = var_measured / var_theory
    else:
        ratio = 1.0 if var_measured < 1e-10 else float('inf')
    
    # Normalized: 1 = ratio is 1, decreases as ratio deviates
    normalized = max(0, 1 - abs(np.log(ratio + 1e-10)))
    
    return MetricResult(
        value=float(ratio),
        normalized=float(normalized),
        is_acceptable=0.5 < ratio < 2.0,
        threshold=2.0,
        description="Variance Ratio (measured/theory)"
    )


def compute_all_metrics(
    theory: np.ndarray,
    measured: np.ndarray,
    normalize: bool = True
) -> Dict[str, MetricResult]:
    """
    Compute all available metrics.
    
    Parameters
    ----------
    theory : np.ndarray
        Theoretical predictions
    measured : np.ndarray
        Measured values
    normalize : bool
        Whether to normalize metrics
        
    Returns
    -------
    dict
        All metric results
    """
    return {
        "mae": mean_absolute_error(theory, measured, normalize),
        "rmse": root_mean_square_error(theory, measured, normalize),
        "pearson": pearson_correlation(theory, measured),
        "spearman": spearman_correlation(theory, measured),
        "max_dev": max_absolute_deviation(theory, measured, normalize),
        "r2": coefficient_of_determination(theory, measured),
        "bias": relative_bias(theory, measured),
        "variance_ratio": variance_ratio(theory, measured),
    }


def compute_agreement_grade(
    metrics: Dict[str, MetricResult],
    weights: Optional[Dict[str, float]] = None
) -> Tuple[str, float]:
    """
    Compute letter grade from metrics.
    
    Parameters
    ----------
    metrics : dict
        Metric results
    weights : dict, optional
        Custom weights for each metric
        
    Returns
    -------
    tuple
        (grade, weighted_score)
    """
    default_weights = {
        "pearson": 0.25,
        "r2": 0.20,
        "mae": 0.15,
        "rmse": 0.15,
        "max_dev": 0.10,
        "bias": 0.10,
        "variance_ratio": 0.05,
    }
    
    w = weights or default_weights
    
    # Compute weighted score
    total_score = 0.0
    total_weight = 0.0
    
    for key, metric in metrics.items():
        if key in w:
            total_score += metric.normalized * w[key]
            total_weight += w[key]
    
    if total_weight > 0:
        weighted_score = total_score / total_weight
    else:
        weighted_score = 0.0
    
    # Assign grade
    if weighted_score >= 0.95:
        grade = "A+"
    elif weighted_score >= 0.90:
        grade = "A"
    elif weighted_score >= 0.85:
        grade = "B+"
    elif weighted_score >= 0.80:
        grade = "B"
    elif weighted_score >= 0.70:
        grade = "C"
    elif weighted_score >= 0.60:
        grade = "D"
    else:
        grade = "F"
    
    return grade, float(weighted_score)


if __name__ == "__main__":
    # Test metrics
    print("Agreement Metrics Test")
    print("=" * 60)
    
    # Generate test data with known agreement
    theory = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    measured = theory * 1.05 + 0.1  # 5% error + offset
    
    print(f"Theory:    {theory}")
    print(f"Measured:  {measured}")
    print()
    
    # Compute all metrics
    metrics = compute_all_metrics(theory, measured, normalize=True)
    
    print("Metrics:")
    for name, result in metrics.items():
        print(f"  {name:15s}: {result.value:8.4f} (normalized: {result.normalized:.3f})")
    
    # Grade
    grade, score = compute_agreement_grade(metrics)
    print(f"\nOverall: Grade {grade} (score: {score:.3f})")
    
    # Perfect agreement test
    print("\n" + "=" * 60)
    print("Perfect agreement test:")
    metrics_perfect = compute_all_metrics(theory, theory)
    grade, score = compute_agreement_grade(metrics_perfect)
    print(f"Grade: {grade} (score: {score:.3f})")
