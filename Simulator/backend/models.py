"""
SQLAlchemy models for QATNU First Principles Simulator.

These tables store:
- Runs: Theory inputs and overall results
- LambdaPoints: Per-lambda measurements
- Validations: Agreement metrics between theory and simulation
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, ForeignKey, Text, UniqueConstraint
from sqlalchemy.types import Integer, Float, String, DateTime, JSON
from sqlalchemy.orm import (
    declarative_base, sessionmaker, relationship, Session, Mapped, mapped_column
)

Base = declarative_base()


class Run(Base):
    """A single simulation run with theory-derived parameters."""
    
    __tablename__ = "runs"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(
        String, 
        default="queued",
        doc="queued|running|paused|completed|failed|cancelled"
    )
    
    # Ground truth inputs (what the user specifies)
    N: Mapped[int] = mapped_column(doc="Number of matter sites")
    topology: Mapped[str] = mapped_column(default="path")
    chi_max: Mapped[int] = mapped_column(default=4, doc="Bond dimension cutoff")
    G_eff: Mapped[float] = mapped_column(default=6.674e-11, doc="Target Newton's constant (dimensionless)")
    c: Mapped[float] = mapped_column(default=1.0, doc="Speed of light")
    omega: Mapped[float] = mapped_column(default=1.0, doc="Bare clock frequency")

    # Tunable scan and heuristic parameters (no hardcoded knobs)
    lambda_min: Mapped[float] = mapped_column(default=0.1, doc="Minimum lambda in scan")
    lambda_max: Mapped[float] = mapped_column(default=1.5, doc="Maximum lambda in scan")
    num_points: Mapped[int] = mapped_column(default=30, doc="Number of lambda points")
    avg_frustration: Mapped[float] = mapped_column(default=0.5, doc="⟨F⟩ heuristic for perturbative α")
    safety_factor: Mapped[float] = mapped_column(default=5.0, doc="Δ_eff/λ_max safety factor")
    target_frustration: Mapped[float] = mapped_column(default=0.9, doc="Target ⟨F_ij⟩ for hotspot")
    max_multiplier: Mapped[float] = mapped_column(default=5.0, doc="Max hotspot multiplier search bound")
    frustration_time: Mapped[float] = mapped_column(default=1.0, doc="Frustration protocol duration")
    use_hotspot_search: Mapped[int] = mapped_column(default=0, doc="Use binary search for hotspot (0/1)")
    hotspot_tolerance: Mapped[float] = mapped_column(default=0.05, doc="Target frustration tolerance")
    hotspot_max_iterations: Mapped[int] = mapped_column(default=6, doc="Max hotspot search iterations")
    J0: Mapped[float] = mapped_column(default=0.01, doc="Ising coupling strength")
    
    # Scale anchoring outputs
    lattice_spacing: Mapped[Optional[float]] = mapped_column(doc="a in Planck units")
    time_step: Mapped[Optional[float]] = mapped_column(doc="tau in Planck units")
    
    # Critical points (observed from simulation)
    lambda_c1: Mapped[Optional[float]] = mapped_column(doc="Onset of curvature")
    lambda_revival: Mapped[Optional[float]] = mapped_column(doc="Quantum revival point")
    lambda_c2: Mapped[Optional[float]] = mapped_column(doc="Catastrophic failure")
    residual_min: Mapped[Optional[float]] = mapped_column()
    
    # Overall agreement score
    agreement_score: Mapped[Optional[float]] = mapped_column(doc="0-1 overall theory match")
    
    # Full config backup
    config_json: Mapped[Optional[dict]] = mapped_column(JSON, doc="Complete run configuration")
    
    # Relationships
    lambda_points: Mapped[List["LambdaPoint"]] = relationship(
        back_populates="run", 
        cascade="all, delete-orphan",
        order_by="LambdaPoint.point_index"
    )
    validation: Mapped[Optional["Validation"]] = relationship(
        back_populates="run", 
        uselist=False,
        cascade="all, delete-orphan"
    )
    derivation_steps: Mapped[List["DerivationStep"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="DerivationStep.step_number"
    )
    
    def __repr__(self):
        return f"<Run(id={self.id}, N={self.N}, status={self.status}, agreement={self.agreement_score})>"


class LambdaPoint(Base):
    """A single lambda point in a scan with theory vs measurement comparison."""
    
    __tablename__ = "lambda_points"
    __table_args__ = (
        UniqueConstraint("run_id", "point_index", name="unique_run_point"),
    )
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    point_index: Mapped[int] = mapped_column(doc="Order in scan")
    
    # The scan value
    lambda_val: Mapped[float] = mapped_column()
    
    # === THEORY (derived from first principles) ===
    theory_alpha: Mapped[Optional[float]] = mapped_column(doc="α from Stinespring/perturbative")
    theory_deltaB: Mapped[Optional[float]] = mapped_column(doc="Bond spacing from validity")
    theory_hotspot: Mapped[Optional[float]] = mapped_column(doc="Multiplier from frustration target")
    theory_kappa: Mapped[Optional[float]] = mapped_column(doc="Degree penalty from G_eff")
    theory_omega_out: Mapped[Optional[float]] = mapped_column(doc="Predicted outer frequency")
    theory_omega_in: Mapped[Optional[float]] = mapped_column(doc="Predicted inner frequency")
    
    # === SIMULATION (measured from exact diagonalization) ===
    measured_omega_out: Mapped[Optional[float]] = mapped_column()
    measured_omega_in: Mapped[Optional[float]] = mapped_column()
    measured_lambda_out: Mapped[Optional[float]] = mapped_column(doc="Λ_out")
    measured_lambda_in: Mapped[Optional[float]] = mapped_column(doc="Λ_in")
    
    # Postulate 1 validation
    residual: Mapped[Optional[float]] = mapped_column(doc="|measured_ratio - predicted_ratio|")
    
    # Energy spectrum
    E0: Mapped[Optional[float]] = mapped_column()
    E1: Mapped[Optional[float]] = mapped_column()
    E2: Mapped[Optional[float]] = mapped_column()
    E3: Mapped[Optional[float]] = mapped_column()
    E4: Mapped[Optional[float]] = mapped_column()
    E5: Mapped[Optional[float]] = mapped_column()
    gap01: Mapped[Optional[float]] = mapped_column()
    gap12: Mapped[Optional[float]] = mapped_column()
    gap23: Mapped[Optional[float]] = mapped_column()
    gap34: Mapped[Optional[float]] = mapped_column()
    gap45: Mapped[Optional[float]] = mapped_column()
    min_gap: Mapped[Optional[float]] = mapped_column()
    
    # Phase classification
    phase_status: Mapped[Optional[str]] = mapped_column(doc="✓|~|✗")
    
    # Agreement for this point
    alpha_error: Mapped[Optional[float]] = mapped_column(doc="|theory - measured| / theory")
    
    run: Mapped["Run"] = relationship(back_populates="lambda_points")
    
    def __repr__(self):
        return f"<LambdaPoint(run={self.run_id}, λ={self.lambda_val:.3f}, error={self.alpha_error:.1%})>"


class Validation(Base):
    """SRQID validators and final agreement metrics for a run."""
    
    __tablename__ = "validations"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), unique=True)
    
    # SRQID structural tests
    v_lr: Mapped[Optional[float]] = mapped_column(doc="Lieb-Robinson velocity")
    no_signalling_max: Mapped[Optional[float]] = mapped_column(doc="max|Δ⟨Z_r⟩|")
    energy_drift: Mapped[Optional[float]] = mapped_column(doc="Peak-to-peak energy variation")
    
    # Mean-field comparison (at lambda_focus)
    lambda_focus: Mapped[Optional[float]] = mapped_column()
    residual_at_focus: Mapped[Optional[float]] = mapped_column()
    measured_freq_out: Mapped[Optional[float]] = mapped_column()
    measured_freq_in: Mapped[Optional[float]] = mapped_column()
    predicted_freq_out: Mapped[Optional[float]] = mapped_column()
    predicted_freq_in: Mapped[Optional[float]] = mapped_column()
    
    # Spin-2 PSD analysis
    spin2_measured_power: Mapped[Optional[float]] = mapped_column(doc="Fitted PSD slope")
    spin2_expected_power: Mapped[float] = mapped_column(default=-2.0)
    spin2_residual: Mapped[Optional[float]] = mapped_column(doc="|measured - (-2)|")
    
    # Overall agreement
    overall_agreement: Mapped[Optional[float]] = mapped_column(doc="0-1 score")
    agreement_breakdown: Mapped[Optional[dict]] = mapped_column(JSON, doc="Per-metric scores")
    
    run: Mapped["Run"] = relationship(back_populates="validation")
    
    def __repr__(self):
        return f"<Validation(run={self.run_id}, agreement={self.overall_agreement:.1%})>"


class DerivationStep(Base):
    """Logs each step of parameter derivation for transparency."""
    
    __tablename__ = "derivation_steps"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    
    step_number: Mapped[int] = mapped_column(doc="Order in derivation")
    parameter_name: Mapped[str] = mapped_column(doc="α|δB|hotspot|κ|k0|a|τ")
    formula_used: Mapped[str] = mapped_column(doc="Name of formula/method")
    inputs: Mapped[Optional[dict]] = mapped_column(JSON, doc="Input values to formula")
    output_value: Mapped[float] = mapped_column()
    intermediate_steps: Mapped[Optional[str]] = mapped_column(Text, doc="Detailed calculation log")
    
    run: Mapped["Run"] = relationship(back_populates="derivation_steps")
    
    def __repr__(self):
        return f"<DerivationStep({self.parameter_name}={self.output_value:.4f})>"


# Database setup functions
def get_engine(db_path: str = "qca.db"):
    """Create SQLAlchemy engine with SQLite."""
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(engine=None):
    """Create all tables."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None) -> Session:
    """Get a database session."""
    if engine is None:
        engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


if __name__ == "__main__":
    # Initialize database
    engine = init_db()
    print("Database initialized at Simulator/qca.db")
    print("Tables created:", Base.metadata.tables.keys())
