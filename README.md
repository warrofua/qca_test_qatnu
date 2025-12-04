# QATNU / SRQID Simulation Framework

This repository houses the `app.py` workflow that unifies the exact QCA diagonalization, λ scans, SRQID structural tests, mean-field overlays, and spin-2 diagnostics used in the recent QATNU/SRQID studies.

## Physics overview (why this exists)
The simulator implements the August 2025 QATNU/SRQID construction in which a measurement-free promotion ladder renormalizes local clock rates according to Postulate 1,
$$
\omega_{\mathrm{eff}}^{(i)} = \frac{\omega}{1+\alpha \Lambda_i}
$$
with $\Lambda_i$ extracted from the logarithmic bond-depth profile of the SRQID Hamiltonian.  Each production run follows the “GR-style” workflow documented in `docs/status_202511.tex`:

1. **Geometry from source:** build the exact many-body Hamiltonian on the requested graph, evolve its ground state under a hotspot (3λ) promotion, and measure $\Lambda_{\text{out/in}}$ from that frustrated background.
2. **Test clocks vs. embedded clocks:** by default (Protocol A) the Ramsey probes are clean π/2 excitations on the ground state, mirroring GR’s “test clock” assumption.  The optional `--embedded-clocks` flag switches to Protocol B, exciting the probes directly on the promoted background to explore the breakdown of Postulate 1 in the fully many-body regime.
3. **Postulate 1 validation:** for each λ sample we compare the measured frequency ratio (FFT of the Ramsey traces) to $(1+\alpha\Lambda_{\text{out}})/(1+\alpha\Lambda_{\text{in}})$, producing the residual/phase diagrams cited in the status paper.
4. **SRQID + spin‑2 checks:** Lieb–Robinson velocity, no-signalling, energy drift, mean-field overlays, and χ→spin-2 PSD diagnostics run within the same Hamiltonian so the entire QATNU/SRQID story stays in one script.

The goal is to give the “unified app” promised in the preprints: phase diagrams, critical points, Ramsey overlays, SRQID metrics, and spin‑2 figures all emerge from a single exact-diagonalization workflow while remaining faithful to the theory narrative.

## Key capabilities
- **Exact QCA engine** (`core_qca.py`) takes the many-body Hamiltonian, constructs the full matter+bond Hilbert space, and exposes time-evolution and observable evaluation.
- **Parallel λ scans** (`scanners.py`) sweep promotion strength to generate phase diagrams with critical points, ω scaling, and phase classification (figures saved under `figures/run_*`).
- **SRQID validators** (`srqid.py`) run Lieb–Robinson, no-signalling, and energy-drift checks in the same Hamiltonian.
- **Mean-field comparator** (`mean_field.py`, `tester.py`) overlays exact Ramsey traces with the Λ-informed semiclassical chain.
- **Spin-2 PSD** (`geometry.py`) derives a PSD directly from the measured χ profile to cross-check emergent-geometry expectations.

## Environment setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Platform-specific behavior

**macOS (Apple Silicon/Intel):**
- Automatically uses Accelerate framework for BLAS operations (optimal performance)
- Sets `forkserver` multiprocessing start method (required for stability)
- **Important**: Serial execution by default due to NumPy `longdouble` issues in multiprocessing
  - To enable parallel scans: `export QATNU_FORCE_MP=1` (may cause instability)
  - Serial mode typically adds ~20-30% runtime but guarantees correctness

**Linux/Windows:**
- Uses OpenBLAS or system BLAS
- Parallel execution enabled by default (6 workers)
- Set `VECLIB_MAXIMUM_THREADS` environment variable to control BLAS threading

**Performance expectations:**
| Configuration | 100-point scan |
|---------------|----------------|
| N=4, macOS serial | ~2-3 minutes |
| N=4, parallel (6 cores) | ~60-90 seconds |
| N=5, macOS serial | ~20 hours |
| N=5, parallel (6 cores) | ~3-5 hours |

## Running the production workflow
```bash
python app.py --N 4 --alpha 0.8 --points 100 --phase-space
```

### Command-line options

**Core parameters:**
- `--N` : Number of matter sites (default: 5, tested up to N=5)
  - N=4: ~1-2 min per λ point, recommended for rapid iteration
  - N=5: ~10-12 min per λ point, use `--points 40-50` for faster runs
- `--alpha` : Postulate‑1 coupling coefficient in `ω/(1+αΛ)` (default: 0.8)
  - Typical values: 0.6–1.2 to explore phase boundary dependence
- `--points` : Number of λ sample points in the 1-D scan (default: 100)
  - Auto-densified around critical regions (λ∈[0.55,0.8]) for default range

**Scan configuration:**
- `--lambda-min` : Minimum λ for the 1-D scan (default: 0.1)
- `--lambda-max` : Maximum λ for the 1-D scan (default: 1.5)
  - Use these to "zoom" into specific λ regions of interest
  - Example: `--lambda-min 0.5 --lambda-max 0.7 --points 50`
- `--phase-space` : Additionally generate the 2-D (α, λ) heatmap
  - Runs 12×6 grid by default (α∈[0.2,1.2], λ∈[0.1,1.2])
  - Warning: significantly increases runtime (~72 parameter points)

**Topology and probes:**
- `--graph` : Topology of the matter-bond network (default: `path`)
  - `path`: Linear chain (N-1 edges)
  - `cycle`: Closed loop, requires N≥3 (N edges)
  - `diamond`: Rhombus geometry, N=4 only (4 edges)
  - `bowtie`: Two triangles sharing a vertex, N=4 only (5 edges)
  - `pyramid` / `star`: Hub-and-spoke with central vertex, N≥2 (N-1 edges)
- `--probes OUTER INNER` : Override probe vertex indices (default: topology-dependent)
  - Format: two integers specifying which sites to use as outer/inner clocks
  - Example: `--probes 0 3` uses vertices 0 and 3
  - Defaults: path/cycle→(0,1), diamond→(0,2), others→(0,1)

**Additional outputs:**
- `--legacy-viz` : Also run the Dataverse legacy visualization scripts (`qatnu_poc.py`)
  - Generates light-cone, dispersion, and MERA figures for comparison

### Example usage patterns

**Quick exploratory run (N=4, standard settings):**
```bash
python app.py --N 4 --alpha 0.8 --points 100
```

**Zoom into critical region around λ≈0.6:**
```bash
python app.py --N 4 --alpha 0.8 --lambda-min 0.5 --lambda-max 0.7 --points 50
```

**Test different topology (cycle graph, N=4):**
```bash
python app.py --N 4 --graph cycle --alpha 0.8 --points 80
```

**Full 2D phase space exploration:**
```bash
python app.py --N 4 --alpha 0.8 --points 100 --phase-space
```

**High-precision N=5 run (long runtime, use reduced points):**
```bash
python app.py --N 5 --alpha 0.8 --points 40
```

**Custom probe locations on diamond topology:**
```bash
python app.py --N 4 --graph diamond --probes 0 2 --alpha 0.8
```

### Output structure

Each run creates timestamped directories with the naming pattern `run_YYYYMMDD-HHMMSS_N{N}_{graph}_alpha{α}`:

**Generated files:**
```
outputs/run_*/
  ├── scan_run_*.csv           # Parameter scan results with all observables
  ├── summary_run_*.txt        # Human-readable summary with critical points
  └── phase_space_run_*.csv    # 2D heatmap data (if --phase-space used)

figures/run_*/
  ├── phase_diagram_run_*.png  # Six-panel phase diagram with critical points
  ├── ramsey_overlay_run_*.png # Mean-field comparison at λ_focus
  ├── spin2_psd_run_*.png      # Emergent geometry PSD analysis
  └── phase_space_run_*.png    # 2D (α,λ) heatmap (if --phase-space used)
```

**CSV output columns:**
- `lambda`: Promotion strength parameter
- `residual`: Postulate 1 residual error |measured - predicted|
- `omega_out`, `omega_in`: Measured effective frequencies from FFT of Ramsey traces
- `predicted_omega_out`, `predicted_omega_in`: Theoretical predictions from ω/(1+αΛ)
- `lambda_out`, `lambda_in`: Circuit depth proxy Λ = log₂(1+degree)
- `E0`–`E5`: Ground state and first five excited state energies
- `gap01`–`gap45`: Energy gaps between adjacent low-lying levels
- `min_gap`: Minimum gap in the low-lying spectrum
- `status`: Phase classification (✓ emergent <5%, ~ borderline 5–10%, ✗ violated >10%)

**Summary file contents:**
- Critical points: λ_c1 (breakdown), λ_revival (quantum revival), λ_c2 (catastrophic)
- SRQID validation metrics: Lieb-Robinson velocity, no-signalling violation, energy drift
- Mean-field comparison at selected λ_focus
- Spin-2 PSD fit quality (measured vs. expected power-law slope)

### Understanding the phase diagram output

The six-panel phase diagram (`phase_diagram_run_*.png`) provides comprehensive visualization:

1. **Top panel**: Postulate 1 residual vs. λ showing phase boundaries
   - Green region (Phase I): Emergent regime where ω/(1+αΛ) holds (residual < 5%)
   - Yellow region (Phase II): Breakdown regime (5-10%)
   - Red region (Phase IV): Catastrophic failure (>50%)
   - Purple vertical line: Quantum revival point (local minimum in residual)

2. **Middle-left panel**: Frequency scaling showing measured vs. predicted ω_eff
   - Tests whether observed frequencies match theoretical predictions
   - Bare frequency ω=1.0 marked for reference

3. **Middle-center panel**: Frequency inversion (high-λ regime)
   - Shows where ω_in > ω_out (breakdown of gravitational analogy)
   - Highlights non-monotonic behavior

4. **Middle-right panel**: Entanglement proxy Λ (circuit depth)
   - Tracks bond complexity at outer/inner probe sites
   - Directly feeds into Postulate 1 predictions

5. **Bottom-left panel**: Frequency ratio deviation
   - Compares measured ratio (ω_in/ω_out) vs. predicted ratio
   - Highlights where the theory breaks down

6. **Bottom-right panel**: Phase classification timeline
   - Color-coded status markers: ✓ (emergent), ~ (borderline), ✗ (violated)
   - Provides quick visual summary of regime boundaries

### Automatic validation checks

Every run performs the following SRQID structural validation tests:

**Lieb-Robinson velocity (v_LR):**
- Measures information propagation speed via commutator growth
- Expected: v_LR ≈ 1.9–2.1 (finite light-cone)
- Extracted from [Z(0,0), Z(r,t)] growth threshold

**No-signalling test:**
- Verifies that local quenches don't instantaneously affect distant observables
- Reports max|Δ⟨Z_r⟩| for a quench at site 0 measured at site N-1
- Expected: < 10⁻¹⁴ (enforced by exact unitary evolution)

**Energy conservation:**
- Tracks ⟨ψ(t)|H|ψ(t)⟩ over time evolution
- Reports peak-to-peak energy drift
- Expected: ~ 10⁻¹⁴ (machine precision, validates unitarity)

These tests run automatically using the same Hamiltonian as the main scan and are reported in the summary file.

## Project layout
```
app.py                # orchestration CLI
core_qca.py           # exact diagonalization engine
mean_field.py         # semiclassical comparator
scanners.py           # parallel λ scans
phase_analysis.py     # plotting + critical point logic
srqid.py              # structural validators (LR velocity, no-signalling, energy drift)
geometry.py           # spin-2 PSD utilities
tester.py             # single-point experiments & overlays
topologies.py         # graph topology definitions
runtime_config.py     # platform detection and BLAS configuration
dataverse_files/      # ancillary SRQID/QATNU scripts from Harvard Dataverse
figures/              # generated plots (timestamped subfolders)
outputs/              # CSVs & summary logs (timestamped subfolders)
```

## Engine architecture & capabilities
- **Dynamic geometry & gauge promotion:** The exact Hamiltonian couples matter qubits to finite-dimensional bond registers on edges via $H_{int}$ so the bond dimension $\chi$ can occupy superposition states (e.g., $|1\rangle + |2\rangle$). Frustration tiles $F_{ij}$ trigger promotions/demotions that let the lattice geometry rewrite itself in response to the state, matching the “gauge promotion” narrative of SRQID/QATNU.
- **Absolute unitary integrity:** Time evolution uses full spectral decomposition ($U(t)=e^{-iHt}$ via `scipy.linalg.eigh`), guaranteeing $U^\dagger U=I$ to machine precision ($\sim 10^{-15}$). There is no integration drift or stochastic noise, which is crucial for validating SRQID’s information-theoretic claims.
- **Bespoke observable extraction:** The code includes custom observables beyond standard spin-chain tooling: the circuit-depth proxy $\Lambda$ (local bond complexity), Lieb–Robinson velocity from commutator growth, and automatic no-signalling / energy-drift checks that run inside the exact Hamiltonian.
- **Hardware-aware parallelism:** `ParameterScanner` detects Apple Silicon/Accelerate stacks, sets `VECLIB_MAXIMUM_THREADS`, and enforces the `forkserver` multiprocessing start method so λ sweeps saturate local cores without manual tuning. Everything runs on consumer hardware—no cluster scheduler required.

## Current technical constraints & limitations
- **Exponential Hilbert scaling:** Because each edge carries an active bond register, the Hilbert space grows as $2^N \times \chi_{\max}^{|E|}$. Exact diagonalization is practical up to roughly $N\approx 5$ (depending on topology/bond cutoff); larger systems would need tensor-network or Krylov reductions.
- **Dense-matrix bottleneck:** Full `eigh` has $O(D^3)$ complexity in the total Hilbert dimension $D$. This is a deliberate trade-off for unitarity and reproducibility; runtime scales steeply with both $N$ and bond cutoff.
- **Bond-dimension truncation:** Runs must specify a fixed `bond_cutoff` ($\chi_{\max}$). This acts as a curvature ceiling—if the dynamics would populate higher tiers, they get truncated, which can set artificial bounds in high-energy regimes.
- **Topological rigidity (current release):** The index-mapping logic is tailored to 1D-like graphs (paths, cycles, bowties, stars). Extending to full 2D/3D lattices would require reworking the basis indexing and neighbor tables.

## Troubleshooting

**"command not found: python" error:**
```bash
# Use python3 instead, or activate the venv first:
source venv/bin/activate
python app.py --N 4 --alpha 0.8
```

**Slow performance on macOS:**
- The code runs serially by default on macOS to avoid NumPy `longdouble` multiprocessing bugs
- To enable parallel execution (experimental, may be unstable):
  ```bash
  export QATNU_FORCE_MP=1
  python app.py --N 4 --alpha 0.8
  ```

**Out of memory errors (N≥5):**
- Reduce `--points` (try 40-50 instead of 100)
- Use `--lambda-min` and `--lambda-max` to focus on a smaller λ range
- Check available RAM: N=5 requires ~4-8 GB per worker

**Phase diagram shows no critical points:**
- Try expanding the λ range: `--lambda-min 0.1 --lambda-max 2.0`
- Check if your topology/parameters moved the critical region
- Examine the CSV directly to see residual values

**Import errors (missing dependencies):**
```bash
# Reinstall requirements
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

**Timestamp directories have spaces (legacy outputs-figures/):**
- Use quotes when accessing: `cd "legacy outputs-figures/"`
- Or use tab completion in your shell

## Ancillary Dataverse scripts
The `dataverse_files/` folder contains the original toy-model numerics (Ising SRQID checks, Hadamard light-cone, stochastic spin-2 PSD, etc.). They are preserved for reference but are not part of the main workflow. You can run them via the included `Makefile` inside that folder if you need the legacy comparison plots.
