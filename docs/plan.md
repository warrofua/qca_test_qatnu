# QATNU First Principles Simulator

## Vision

A **terminal-based first principles simulator** that derives all parameters from theory (not fitting) and validates whether the QATNU/SRQID physics hangs together. No hand-tuned knobs—all parameters are computed from the underlying mathematical framework.

**Core Question:** Does the theory I wrote in the paper actually produce the numbers I measured?

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Textual Frontend (Terminal UI)         │
│  ┌─────────────────────────────────┐    │
│  │  QATNU FIRST PRINCIPLES SIM     │    │
│  │                                 │    │
│  │  Ground Truth Inputs:           │    │
│  │  • Topology: [Path ▼]           │    │
│  │  • N (sites): [4]               │    │
│  │  • χ_max (cutoff): [4]          │    │
│  │  • G_eff (Newton): [6.67e-11]   │    │
│  │                                 │    │
│  │  [Validate Theory]              │    │
│  │                                 │    │
│  │  ═══════════════════════════    │    │
│  │  Agreement: 94.2% ✓             │    │
│  │  ═══════════════════════════    │    │
│  │                                 │    │
│  │  ┌─────────┬─────────┬────────┐ │    │
│  │  │ Theory  │ Measured│ Error  │ │    │
│  │  ├─────────┼─────────┼────────┤ │    │
│  │  │ λ_c1    │ 0.203   │ 0.198  │ │    │
│  │  │ λ_rev   │ 1.058   │ 1.062  │ │    │
│  │  └─────────┴─────────┴────────┘ │    │
│  │                                 │    │
│  │  [View Derivation Steps]        │    │
│  │  ▸ α=0.843 (Stinespring)        │    │
│  │  ▸ δB=5.0 (perturbative)        │    │
│  │  ▸ Hotspot=3.2 (⟨F⟩=0.90)       │    │
│  │                                 │    │
│  │  Scan: ████████████████████░░   │    │
│  │  λ=0.634, residual=0.134        │    │
│  └─────────────────────────────────┘    │
│                    │                    │
│                    │ WebSocket          │
┌────────────────────┴────────────────────┤
│  FastAPI Backend                        │
│  ┌─────────────────────────────────────┐│
│  │  Derivation Engine                  ││
│  │  • Stinespring α(λ)                 ││
│  │  • Scale anchoring (a, τ)           ││
│  │  • Perturbative validity (δB)       ││
│  │  • Frustration target (hotspot)     ││
│  │  • Newtonian limit (κ)              ││
│  └─────────────────────────────────────┘│
│  ┌─────────────────────────────────────┐│
│  │  Physics Runner                     ││
│  │  • Exact diagonalization            ││
│  │  • Real-time streaming              ││
│  │  • Measurement extraction           ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
│                    │
│                    ▼
┌─────────────────────────────────────────┐
│  SQLite Database (qca.db)               │
│  • runs (theory inputs + results)       │
│  • lambda_points (per-λ measurements)   │
│  • validations (agreement metrics)      │
│  • derivations (step-by-step logs)      │
└─────────────────────────────────────────┘
```

---

## Philosophy: Parameters Are Outputs, Not Inputs

### Traditional (Fitting) Mode ❌
```
Inputs: N=4, α=0.8, hotspot=3.0, δB=5.0, κ=0.1
↓
Does Postulate 1 hold?
↓
Yes/No (empirical)
```

### First Principles (Derivation) Mode ✅
```
Inputs: N=4, topology=path, χ_max=4, G_eff, c
↓
Compute: α(λ), δB(λ), hotspot(λ), κ(λ) from theory
↓
Run simulation with derived parameters
↓
Compare: Does measured α_eff(λ) match derived α_theory(λ)?
↓
Agreement score: 94.2%
```

---

## Derivation Engine

### 1. Scale Anchoring (Vacuum + Black Hole)

From `mega_document.md` Sec. 6:

```python
def compute_lattice_spacing(chi_max: int, beta: float = 1.0) -> float:
    """
    From Bekenstein-Hawking entropy anchor:
    
    S_QATNU = γ × N_∂ = ln(χ_max) × N_∂
    S_BH = A / (4ℓ_P²) = β × N_∂ × a² / (4ℓ_P²)
    
    Equating: γ = β × a² / (4ℓ_P²)
    
    Returns: a / ℓ_P (lattice spacing in Planck units)
    """
    gamma = np.log(chi_max)
    return 2 * np.sqrt(gamma / beta)

def compute_time_step(
    c_lattice: float,      # Group velocity in lattice units (~1/√2)
    a: float,              # Lattice spacing
    c_physical: float = 1.0  # Speed of light
) -> float:
    """τ = c_lattice × a / c_physical"""
    return c_lattice * a / c_physical
```

### 2. α from Stinespring (Non-Perturbative)

From `mega_document.md` Eq. 4.1 and Appendix:

```python
def compute_alpha_stinespring(
    qca: ExactQCA,
    lambda_val: float,
    bond_structure: Dict
) -> float:
    """
    1. Build promotion isometry V: H_matter → H_matter ⊗ H_bond
       V = Σ_m √(m+1) |m+1⟩⟨m| ⊗ F_ij
    
    2. Compute quantum channel: T(ρ) = Tr_bond(V ρ V†)
    
    3. Find fixed point ρ* = T(ρ*)
    
    4. Extract effective frequency from off-diagonal decay
       ω_eff = -Im(log(λ_2)) / dt  where λ_2 is subleading eigenvalue
    
    5. Numerical derivative: α = -∂_Λ log(ω_eff / ω)
    
    Returns: α(λ) from first principles
    """
    # Implementation requires new physics code
    pass

def compute_alpha_perturbative(
    lambda_val: float,
    omega: float,
    delta_eff: float,
    avg_frustration: float = 0.5
) -> float:
    """
    Fallback: perturbative formula (mega_document Eq. 144)
    
    α_pert ≈ (2λ² / (ω × Δ_eff)) × ⟨F⟩
    
    Used when Stinespring is too expensive or for validation.
    """
    return (2 * lambda_val**2) / (omega * delta_eff) * avg_frustration
```

### 3. δB from Perturbative Validity

```python
def compute_deltaB(
    omega: float,
    lambda_max: float,
    safety_factor: float = 5.0
) -> float:
    """
    Ensure perturbative regime: Δ_eff > safety_factor × λ_max
    
    Theory requirement: α_pert derivation requires Δ_eff ≫ λ
    
    Returns: bond energy spacing δB
    """
    return safety_factor * lambda_max * omega
```

### 4. Hotspot from Frustration Target

```python
def compute_hotspot_multiplier(
    qca: ExactQCA,
    lambda_nominal: float,
    target_frustration: float = 0.9,
    evolution_time: float = 1.0,
    max_multiplier: float = 5.0
) -> float:
    """
    Find M such that evolving ground state with λ = M × λ_nominal
    for t=evolution_time gives ⟨F_ij⟩ ≈ target_frustration
    
    Physics: Create "maximally frustrated but not saturated" initial state
    
    Returns: hotspot multiplier M
    """
    def measure_frustration_at_m(M: float) -> float:
        config = qca.config.copy()
        config['lambda'] = lambda_nominal * M
        test_qca = ExactQCA(qca.N, config, bond_cutoff=qca.bond_cutoff)
        
        ground = test_qca.get_ground_state()
        evolved = test_qca.evolve_state(ground, evolution_time)
        
        # Measure ⟨F_ij⟩ averaged over edges
        F_total = sum(
            measure_frustration(evolved, i, j)
            for (i, j) in qca.edges
        )
        return F_total / len(qca.edges)
    
    # Binary search for target frustration
    return binary_search(
        target=target_frustration,
        f=measure_frustration_at_m,
        low=1.0,
        high=max_multiplier
    )
```

### 5. κ from Newtonian Limit

From `mega_document.md` Appendix D:

```python
def compute_kappa_from_Geff(
    G_eff: float,      # Target Newton's constant (dimensionless)
    alpha: float,      # From Stinespring
    c: float,          # Speed of light (from anchoring)
    a: float           # Lattice spacing (from anchoring)
) -> float:
    """
    From Poisson equation: 4πG_eff = α × c² × κ / a²
    
    Solving: κ = 4πG_eff × a² / (α × c²)
    
    Physics: Links microscopic α to macroscopic G via κ
    """
    return 4 * np.pi * G_eff * a**2 / (alpha * c**2)
```

---

## Validation Loop

```python
class FirstPrinciplesValidator:
    def __init__(self, topology: str, N: int, chi_max: int):
        self.topology = topology
        self.N = N
        self.chi_max = chi_max
        
        # Compute scales once
        self.a = compute_lattice_spacing(chi_max)
        self.tau = compute_time_step(1/np.sqrt(2), self.a)
        
    def validate(self, lambda_range: Tuple[float, float]) -> ValidationReport:
        """
        Full validation pipeline:
        
        For each λ in range:
          1. Derive all parameters from theory
          2. Run exact simulation
          3. Measure effective behavior
          4. Compare theory vs measurement
        """
        report = ValidationReport()
        
        for lam in np.linspace(*lambda_range, 50):
            # STEP 1: DERIVE from theory
            derived = self.derive_parameters(lam)
            
            # STEP 2: SIMULATE with derived params
            observed = self.simulate(derived)
            
            # STEP 3: COMPARE
            agreement = self.compare(derived, observed)
            
            report.add_point(lam, derived, observed, agreement)
        
        return report
    
    def derive_parameters(self, lambda_val: float) -> DerivedParams:
        """Apply all theory constraints"""
        
        # Core derivations
        alpha = compute_alpha_perturbative(lambda_val, omega=1.0, delta_eff=5.0)
        # TODO: Use Stinespring when implemented
        
        deltaB = compute_deltaB(omega=1.0, lambda_max=lambda_val)
        
        hotspot = compute_hotspot_multiplier(
            self.qca, lambda_val, target_frustration=0.9
        )
        
        kappa = compute_kappa_from_Geff(
            G_eff=6.674e-11,  # Or dimensionless
            alpha=alpha,
            c=1.0,
            a=self.a
        )
        
        k0 = self.compute_target_degree()
        
        return DerivedParams(
            alpha=alpha,
            deltaB=deltaB,
            hotspot=hotspot,
            kappa=kappa,
            k0=k0,
            a=self.a,
            tau=self.tau
        )
    
    def compare(self, derived: DerivedParams, observed: ObservedParams) -> Agreement:
        """Compute agreement metrics"""
        return Agreement(
            alpha_error=abs(derived.alpha - observed.alpha_eff) / derived.alpha,
            critical_point_error=self.compare_critical_points(...),
            overall_score=...  # Weighted combination
        )
```

---

## Textual UI Components

### Main Screen Layout

```python
# textual_app.py
from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Static, DataTable, Button, 
    Input, Select, ProgressBar, Collapsible, Label
)
from textual.containers import Container, Horizontal, Vertical

class QATNUSimulator(App):
    CSS = """
    Screen { align: center middle; }
    
    .title { 
        text-align: center; 
        text-style: bold underline; 
        color: cyan;
        margin: 1 0;
    }
    
    .agreement { 
        text-align: center; 
        text-style: bold;
        padding: 1;
        border: solid cyan;
    }
    
    .agreement.good { color: green; }
    .agreement.warning { color: yellow; }
    .agreement.bad { color: red; }
    
    .derivation-step { 
        margin: 0 2;
        color: dim;
    }
    
    .derivation-step.active { 
        color: bright_white;
        text-style: bold;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container():
            # Title
            yield Static("QATNU FIRST PRINCIPLES SIMULATOR", classes="title")
            
            # Input panel
            with Container(id="inputs"):
                with Horizontal():
                    yield Select(
                        [(t, t) for t in ["path", "cycle", "star", "diamond"]],
                        id="topology",
                        value="path"
                    )
                    yield Input(placeholder="N", value="4", id="n-input")
                    yield Input(placeholder="χ_max", value="4", id="chi-input")
                    yield Input(placeholder="G_eff", value="6.67e-11", id="g-input")
                yield Button("Validate Theory", id="validate-btn", variant="primary")
            
            # Results panel (hidden initially)
            with Container(id="results"):
                yield Static(id="agreement-display", classes="agreement")
                
                yield DataTable(id="comparison-table")
                
                yield ProgressBar(id="scan-progress", total=50)
                
                with Collapsible(title="Derivation Steps", collapsed=True):
                    yield Static(id="derivation-log")
            
            # Status bar
            yield Static(id="status-bar")
        
        yield Footer()
```

### Widgets

**DataTable for comparisons:**
```
┌──────────┬─────────┬─────────┬────────┬────────┐
│ λ        │ α_theory│ α_meas  │ Error  │ Status │
├──────────┼─────────┼─────────┼────────┼────────┤
│ 0.100    │ 0.020   │ 0.021   │ 5.0%   │ ✓      │
│ 0.500    │ 0.500   │ 0.480   │ 4.0%   │ ✓      │
│ 1.000    │ 2.000   │ 1.500   │ 25.0%  │ ⚠      │
└──────────┴─────────┴─────────┴────────┴────────┘
```

**Collapsible derivation logs:**
```
▶ Derivation Steps (click to expand)
  ▸ α = 0.843 (Stinespring construction)
    └─▶ Isometry V: 1024×64, computed ω_eff = 0.731
  ▸ δB = 5.0 (Perturbative validity constraint)
    └─▶ Δ_eff/λ_max = 5.2 ✓
  ▸ Hotspot = 3.2 (Target ⟨F⟩ = 0.90)
    └─▶ Binary search converged in 4 iterations
  ▸ κ = 0.15 (From 4πG_eff = αc²κ/a²)
    └─▶ Using a = 1.67 ℓ_P, G_eff = 6.67e-11
```

**Progress with sparkline:**
```
Scan Progress: ████████████████████░░░░ 67%
λ = 0.634  ▁▂▄▅▇▇▇█▇▇▅▄▃▂▁  (residual history)
```

---

## File Structure

```
qatnu_simulator/
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app
│   ├── derivation/                # First principles calculations
│   │   ├── __init__.py
│   │   ├── anchoring.py           # Scale setting (a, τ)
│   │   ├── stinespring.py         # α from isometry
│   │   ├── perturbative.py        # Fallback formulas
│   │   ├── frustration.py         # Hotspot from ⟨F⟩
│   │   └── newtonian.py           # κ from G_eff
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── runner.py              # Pausable runner
│   │   └── measurement.py         # Extract observables
│   ├── models.py                  # SQLAlchemy tables
│   └── api/
│       ├── runs.py                # HTTP endpoints
│       └── websocket.py           # Real-time streaming
│
├── frontend/                      # Textual UI
│   ├── __init__.py
│   ├── app.py                     # Main Textual app
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── comparison_table.py    # Theory vs sim table
│   │   ├── derivation_log.py      # Collapsible steps
│   │   ├── progress_sparkline.py  # ASCII progress
│   │   └── agreement_gauge.py     # Big percentage
│   └── screens/
│       ├── __init__.py
│       ├── main_screen.py         # Primary interface
│       └── derivation_detail.py   # Full derivation view
│
├── qca_core/                      # EXISTING CODE (reuse)
│   ├── core_qca.py
│   ├── scanners.py
│   ├── phase_analysis.py
│   ├── srqid.py
│   ├── tester.py
│   ├── geometry.py
│   └── topologies.py
│
├── docs/
│   ├── plan.md                    # This file
│   └── plan_web_dashboard.md      # Previous web UI plan
│
├── qca.db                         # SQLite database
├── run.py                         # Entry point
└── requirements.txt
```

---

## Implementation Phases

### Phase 0: Foundation (Days 1-2)
- [ ] SQLite schema
- [ ] FastAPI skeleton
- [ ] Textual "Hello World" layout
- [ ] WebSocket connection test

### Phase 1: Derivation Engine (Days 3-6)
- [ ] Scale anchoring (a, τ)
- [ ] Perturbative α(λ)
- [ ] Perturbative validity δB
- [ ] Frustration-target hotspot
- [ ] Newtonian κ
- [ ] **Stinespring α (stretch goal)**

### Phase 2: Physics Integration (Days 7-10)
- [ ] Wrap existing QCA code
- [ ] Pause/resume/cancel logic
- [ ] Real-time measurement extraction
- [ ] Database streaming
- [ ] Agreement metric computation

### Phase 3: Textual UI (Days 11-14)
- [ ] Main screen layout
- [ ] Input widgets
- [ ] Comparison table
- [ ] Derivation log (collapsible)
- [ ] Progress sparkline
- [ ] Agreement gauge

### Phase 4: Validation & Polish (Days 15-18)
- [ ] Compare to existing CSV runs
- [ ] Calibrate derivation formulas
- [ ] Error handling
- [ ] Export results (JSON/Markdown)
- [ ] Documentation

### Phase 5: Stinespring (Days 19-30)
- [ ] Implement isometry construction
- [ ] Quantum channel computation
- [ ] Fixed-point solver
- [ ] Compare perturbative vs non-perturbative α
- [ ] Document differences

---

## Success Metrics

1. **All parameters derived** — No hand-tuned values in simulation
2. **Theory vs simulation agreement >90%** — For perturbative formulas
3. **Real-time validation** — See agreement score update as scan runs
4. **Derivation transparency** — Every parameter shows its formula and intermediate values
5. **Calibrated against existing data** — Matches your existing CSV runs when using same theory
6. **Agent accessible** — I can read terminal output and validate physics without screenshots

---

## Migration from Existing Data

Existing CSV outputs serve as **ground truth** for validating the derivation engine:

```python
# calibration/validate_derivations.py

def calibrate_alpha_formula():
    """Compare perturbative α to measured α_eff from CSV"""
    for csv_file in glob("outputs/run_*/scan_*.csv"):
        df = pd.read_csv(csv_file)
        
        # Compute theoretical α
        alpha_theory = compute_alpha_perturbative(
            lambda_val=df['lambda'],
            omega=1.0,
            delta_eff=5.0
        )
        
        # Compare to measured
        alpha_measured = extract_alpha_from_frequencies(
            df['omega_out'], df['omega_in'],
            df['lambda_out'], df['lambda_in']
        )
        
        # Plot: Theory vs Measured
        # Should cluster around y=x line
```

**The app answers:** "If I derive all parameters from first principles, do I recover the behavior I empirically observed?"

---

## Key Difference from Previous Plan

| Aspect | Web Dashboard (plan_web_dashboard.md) | First Principles Simulator (this plan) |
|--------|--------------------------------------|----------------------------------------|
| **Primary goal** | Visualize many runs, compare parameters | Validate theory, derive parameters |
| **Parameters** | User inputs (fitting mode) | Computed from theory (derivation mode) |
| **Frontend** | React + D3.js (browser) | Textual (terminal) |
| **Key output** | Pretty charts | Agreement scores, derivation logs |
| **Agent accessibility** | Screenshots needed | Terminal text I can read |
| **Build time** | 3-4 weeks | 2-3 weeks |
| **Physics rigor** | Medium (can tweak knobs) | High (all from first principles) |

---

