"""
Agreement weighting configuration.

Provides configurable weighting schemes for combining different
metrics into an overall agreement score. Weights can be customized
for different physics priorities.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class PhysicsPriority(Enum):
    """Predefined physics priorities for weighting."""
    BALANCED = "balanced"
    FREQUENCY_MATCH = "frequency_match"  # Prioritize ω_eff agreement
    LAMBDA_MATCH = "lambda_match"        # Prioritize Λ agreement
    SRQID_VALIDATION = "srqid"           # Prioritize SRQID checks
    CRITICAL_POINT = "critical"          # Prioritize near critical points


@dataclass
class WeightConfig:
    """Configuration for metric weights and thresholds."""
    
    # Metric weights (should sum to 1.0)
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "pearson": 0.25,
        "r2": 0.20,
        "mae": 0.15,
        "rmse": 0.15,
        "max_dev": 0.10,
        "bias": 0.10,
        "variance_ratio": 0.05,
    })
    
    # Quantity weights for overall score
    quantity_weights: Dict[str, float] = field(default_factory=lambda: {
        "Lambda": 0.40,      # Bubble density is primary observable
        "omega_eff": 0.30,   # Effective frequency is derived
        "critical_points": 0.20,  # Phase transitions
        "srqid": 0.10,       # Validation checks
    })
    
    # Grade thresholds
    grade_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "A+": 0.95,
        "A": 0.90,
        "B+": 0.85,
        "B": 0.80,
        "C": 0.70,
        "D": 0.60,
        "F": 0.0,
    })
    
    # Per-metric acceptance thresholds
    acceptance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "pearson": 0.90,
        "r2": 0.85,
        "mae": 0.20,
        "rmse": 0.20,
        "max_dev": 0.15,
        "bias": 0.10,
        "variance_ratio": 2.0,
    })
    
    # Lambda-scan specific: weight near critical points more
    critical_point_boost: float = 1.5
    critical_region_width: float = 0.1  # λ range around critical point
    
    def validate(self) -> bool:
        """Check that weights sum appropriately."""
        metric_sum = sum(self.metric_weights.values())
        quantity_sum = sum(self.quantity_weights.values())
        
        return (
            abs(metric_sum - 1.0) < 0.01 and
            abs(quantity_sum - 1.0) < 0.01
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metric_weights": self.metric_weights,
            "quantity_weights": self.quantity_weights,
            "grade_thresholds": self.grade_thresholds,
            "acceptance_thresholds": self.acceptance_thresholds,
            "critical_point_boost": self.critical_point_boost,
            "critical_region_width": self.critical_region_width,
        }


# Predefined configurations
BALANCED_CONFIG = WeightConfig()

FREQUENCY_PRIORITY_CONFIG = WeightConfig(
    quantity_weights={
        "Lambda": 0.25,
        "omega_eff": 0.50,  # Boost frequency
        "critical_points": 0.15,
        "srqid": 0.10,
    },
    metric_weights={
        "pearson": 0.30,  # Correlation matters most for frequencies
        "r2": 0.25,
        "mae": 0.15,
        "rmse": 0.15,
        "max_dev": 0.05,
        "bias": 0.05,
        "variance_ratio": 0.05,
    }
)

LAMBDA_PRIORITY_CONFIG = WeightConfig(
    quantity_weights={
        "Lambda": 0.60,  # Boost bubble density
        "omega_eff": 0.15,
        "critical_points": 0.15,
        "srqid": 0.10,
    }
)

SRQID_PRIORITY_CONFIG = WeightConfig(
    quantity_weights={
        "Lambda": 0.30,
        "omega_eff": 0.25,
        "critical_points": 0.20,
        "srqid": 0.25,  # Boost validation
    },
    acceptance_thresholds={
        "pearson": 0.95,
        "r2": 0.90,
        "mae": 0.15,
        "rmse": 0.15,
        "max_dev": 0.10,
        "bias": 0.05,
        "variance_ratio": 1.5,
    }
)

CRITICAL_POINT_CONFIG = WeightConfig(
    critical_point_boost=2.0,  # Strong boost
    critical_region_width=0.15,
    quantity_weights={
        "Lambda": 0.35,
        "omega_eff": 0.25,
        "critical_points": 0.30,  # Boost critical points
        "srqid": 0.10,
    }
)


def get_weight_config(priority: PhysicsPriority) -> WeightConfig:
    """
    Get predefined weight configuration.
    
    Parameters
    ----------
    priority : PhysicsPriority
        Desired priority scheme
        
    Returns
    -------
    WeightConfig
        Configuration for the priority
    """
    configs = {
        PhysicsPriority.BALANCED: BALANCED_CONFIG,
        PhysicsPriority.FREQUENCY_MATCH: FREQUENCY_PRIORITY_CONFIG,
        PhysicsPriority.LAMBDA_MATCH: LAMBDA_PRIORITY_CONFIG,
        PhysicsPriority.SRQID_VALIDATION: SRQID_PRIORITY_CONFIG,
        PhysicsPriority.CRITICAL_POINT: CRITICAL_POINT_CONFIG,
    }
    return configs.get(priority, BALANCED_CONFIG)


def customize_weights(
    base: WeightConfig,
    metric_adjustments: Optional[Dict[str, float]] = None,
    quantity_adjustments: Optional[Dict[str, float]] = None
) -> WeightConfig:
    """
    Create customized weights from base config.
    
    Parameters
    ----------
    base : WeightConfig
        Base configuration
    metric_adjustments : dict, optional
        Adjustments to metric weights (additive)
    quantity_adjustments : dict, optional
        Adjustments to quantity weights (additive)
        
    Returns
    -------
    WeightConfig
        Customized configuration (normalized)
    """
    import copy
    config = copy.deepcopy(base)
    
    # Apply adjustments
    if metric_adjustments:
        for key, delta in metric_adjustments.items():
            if key in config.metric_weights:
                config.metric_weights[key] += delta
        # Normalize
        total = sum(config.metric_weights.values())
        config.metric_weights = {k: v/total for k, v in config.metric_weights.items()}
    
    if quantity_adjustments:
        for key, delta in quantity_adjustments.items():
            if key in config.quantity_weights:
                config.quantity_weights[key] += delta
        # Normalize
        total = sum(config.quantity_weights.values())
        config.quantity_weights = {k: v/total for k, v in config.quantity_weights.items()}
    
    return config


class GradeScale:
    """Letter grade scale with descriptions."""
    
    GRADES = {
        "A+": (0.95, "Exceptional agreement"),
        "A": (0.90, "Excellent agreement"),
        "B+": (0.85, "Very good agreement"),
        "B": (0.80, "Good agreement"),
        "C": (0.70, "Acceptable agreement"),
        "D": (0.60, "Marginal agreement"),
        "F": (0.00, "Poor agreement - theory rejected"),
    }
    
    @classmethod
    def get_grade(cls, score: float) -> str:
        """Get grade letter from score."""
        for grade, (threshold, _) in sorted(cls.GRADES.items(), key=lambda x: -x[1][0]):
            if score >= threshold:
                return grade
        return "F"
    
    @classmethod
    def get_description(cls, grade: str) -> str:
        """Get description for grade."""
        return cls.GRADES.get(grade, (0, "Unknown"))[1]
    
    @classmethod
    def get_color(cls, grade: str) -> str:
        """Get display color for grade."""
        colors = {
            "A+": "bright_green",
            "A": "green",
            "B+": "light_green",
            "B": "yellow",
            "C": "orange",
            "D": "red",
            "F": "bright_red",
        }
        return colors.get(grade, "white")


if __name__ == "__main__":
    # Test weight configurations
    print("Weight Configuration Test")
    print("=" * 60)
    
    # Test default config
    print("\nBalanced config:")
    config = BALANCED_CONFIG
    print(f"  Valid: {config.validate()}")
    print(f"  Metric weights: {config.metric_weights}")
    print(f"  Quantity weights: {config.quantity_weights}")
    
    # Test priority configs
    for priority in PhysicsPriority:
        config = get_weight_config(priority)
        print(f"\n{priority.value}:")
        print(f"  Valid: {config.validate()}")
    
    # Test grade scale
    print("\n" + "=" * 60)
    print("Grade Scale:")
    for score in [0.97, 0.92, 0.87, 0.82, 0.75, 0.65, 0.50]:
        grade = GradeScale.get_grade(score)
        desc = GradeScale.get_description(grade)
        print(f"  Score {score:.2f}: {grade} - {desc}")
    
    # Test customization
    print("\n" + "=" * 60)
    print("Customization test:")
    custom = customize_weights(
        BALANCED_CONFIG,
        quantity_adjustments={"Lambda": 0.2, "omega_eff": -0.2}
    )
    print(f"  Original Lambda weight: {BALANCED_CONFIG.quantity_weights['Lambda']:.2f}")
    print(f"  Custom Lambda weight: {custom.quantity_weights['Lambda']:.2f}")
    print(f"  Valid: {custom.validate()}")
