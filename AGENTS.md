# Agent Playbook – QATNU/SRQID Repo

## Mission
Model and validate the QATNU/SRQID physics claims: a measurement-free promotion mechanism that slows local clocks according to Postulate 1,
\[
\omega_{\mathrm{eff}} = \frac{\omega}{1+\alpha \Lambda},
\]
while respecting Lieb–Robinson locality and providing evidence for a spin-2 channel. The codebase runs exact diagonalization of the SRQID Hamiltonian, performs λ scans, overlays mean-field comparators, and exports figures/logs for documentation.

## Repo Layout Cheat Sheet
```
app.py                 – CLI orchestration (entry point)
core_qca.py            – exact Hamiltonian, evolution, Λ extraction
scanners.py            – parallel λ scans
phase_analysis.py      – critical point detection + plots
srqid.py               – LR velocity, no-signalling, energy drift checks
mean_field.py/tester.py– mean-field comparator + single-run helpers
geometry.py            – χ-driven spin-2 PSD
docs/                  – README, results snapshot, status report
legacy outputs-figures/– reference outputs from appv3.py (N=4/N=5 figures)
dataverse_files/       – ancillary toy scripts (light cone, dispersion, etc.)
outputs/run_*          – per-run CSVs, summaries
figures/run_*          – per-run PNGs (phase diagram, overlay, PSD)
```

## How to Run
Canonical N=4 run:
```bash
python app.py --N 4 --alpha 0.8 --points 100 --phase-space
```
- Generates `outputs/run_<timestamp>_N4_alpha0.80/` with CSV + summary.
- Figures stored in `figures/run_<timestamp>_N4_alpha0.80/`.

N=5 guidance:
- Runtime explodes (≈10–12 min per λ point). Use `--points 40-50`.
- Expect 8× larger Hilbert space and >50× runtime relative to N=4.

After major changes to docs, recompile `docs/status_202511.tex`:
```bash
cd docs && pdflatex -interaction=nonstopmode status_202511.tex
```

## Validation Checklist
1. **Phase diagram** (`phase_analysis.py`):
   - Check residual curve, critical points, and annotated revival.
2. **SRQID validators** (`srqid.py`):
   - v_LR ≈ 1.9–2.1; max |Δ⟨Z_r⟩| < 10⁻¹⁵; ΔE ~ 10⁻¹⁴.
3. **Ramsey overlays** (`tester.py`):
   - Ensure new vs legacy traces match at λ≈1.058 (N=4).
4. **Spin-2 PSD** (`geometry.py`):
   - Currently underperforms (slope ≈ 0); note any improvements/regressions.
5. **Docs**:
   - Update `docs/results.md` with new run tags.
   - Keep `docs/status_202511.tex` aligned with the physics narrative.

## What Not to Break
- `Λ` extraction in `core_qca.py`: Postulate 1 residuals rely on it.
- `phase_analysis.py`: replicates the preprint's six-panel figure; keep critical point logic intact.
- `srqid.py`: must continue running inside the exact Hamiltonian to maintain SRQID claims.
- Legacy outputs in `legacy outputs-figures/`: reference data for comparison; don't overwrite.
- `dataverse_files/`: contains toy scripts used in the papers.

## Open Tasks / TODO

### High Priority (Physics-Critical)
1. **Fix lambda conversion bug in `tester.py:153-154`** — `measured_lambda_out = np.log2(1.0 + 2.0**depthOut - 1.0)` simplifies to just `depthOut`. This may invalidate Postulate 1 residuals. Verify the intended Λ extraction formula.
2. **Improve χ→PSD mapping** — Current slope ≈ 0 vs expected 1/k². The stochastic toy model in `geometry.py` decouples promotions, but real Hamiltonian has correlated promotions via F_ij. Need bond-bond correlators instead of raw χ profile.
3. **Test hotspot multiplier sensitivity** — The 3.0× multiplier for frustrated background is hardcoded in `scanners.py:63`. Test 2.0× and 4.0× to verify λ_revival doesn't shift; document physical justification.

### Medium Priority (Robustness & Completeness)
4. **Hamiltonian parameter sensitivity scan** — Test sensitivity to `deltaB` (bond spacing), `kappa` (degree penalty), `k0` (target degree). Currently fixed at 5.0, 0.1, 4. Are these scale-setting or tuning parameters?
5. **Perform α sweeps at N=4** — Test α ∈ {0.6, 1.0, 1.2} to map phase boundaries and test Postulate 1 rigidity across regimes.
6. **Implement Stinespring derivation** — Compute α directly from promotion isometry rather than fitting. Documented in `docs/mega_document_expanded_full.md` but not implemented.
7. **Test bond_cutoff dependence** — Sweep χ_max ∈ {3,4,6,8,12} at N=4 to check if phase boundaries (λ_c1, λ_rev, λ_c2) are truncation artifacts. Initial runs show identical results for χ=3 and χ=4 (both give λ_c1≈0.21, λ_rev≈0.34).

### Lower Priority (Features & Tech Debt)
8. **Build Dashboard Application** — See `docs/plan.md` for full specification. FastAPI + WebSocket + D3.js SVG dashboard with SQLite backend. Single-window real-time visualization with pause/resume/stop/start controls.
9. **Add unit tests** — pytest suite for `core_qca` Hamiltonian construction, observable extraction, and validators. Current validation is integration-test only.
10. **Add non-1D graph support** — Square grid, triangular lattice, "house" topology. Requires neighbor lookup overhaul in `core_qca.py`.
11. **Profile memory for N=5** — Consider sparse matrix methods or tensor network approximations. Currently O(D³) dense `eigh` is bottleneck.
12. **Add checkpointing** — For long N=5 scans, save intermediate results to resume on crash.

---

## Theory vs Implementation: Known Divergences

The `mega_document_expanded_full.md` describes an abstract Hamiltonian framework, but the implementation added several phenomenological parameters not in the theory. These are **engineering choices** to stabilize numerics, not derived from first principles.

### Parameters Added for Numerical Stability

| Parameter | Code Value | Theory Status | Purpose |
|-----------|-----------|---------------|---------|
| `deltaB = 5.0` | Bond energy spacing | Maps to $\Delta_{\text{eff}}$ in perturbative α formula | Sets promotion energy cost |
| `kappa = 0.1` | Degree penalty | ❌ **Not in theory** | Prevents runaway promotions |
| `k0 = 4` | Target degree | ❌ **Not in theory** | Preferred coordination (irrelevant for 1D paths) |
| Hotspot multiplier `3.0×` | Frustration protocol | ❌ **Not in theory** | Creates promotion cluster heuristically |

### The Hotspot Protocol Issue

The theory (Sec 3.2) mentions:
> "an inner site $i_{\text{in}}$ near the center of the **promotion cluster** with higher $\Lambda_{i_{\text{in}}}$"

But **never specifies how to create this cluster**. The code uses:
```python
hotspot_config["lambda"] = lambda_param * 3.0  # Heuristic!
```

This is an **implementation choice** without theoretical justification. The 3.0× value was likely found empirically to maximize frustration while avoiding premature saturation.

### Best Path Forward

**Immediate (validation)**:
1. **Test hotspot multiplier sensitivity** — Run λ scans with multipliers 2.0, 3.0, 4.0. If λ_revival shifts, the choice is arbitrary. If stable, document as "empirically determined."
2. **Test kappa sensitivity** — If results depend on degree penalty, the model isn't robust.

**Short-term (alignment)**:
3. **Lock deltaB to omega** — Scale invariance suggests `deltaB = c * omega` for constant c.
4. **Derive kappa from theory** — If degree penalty is needed for stability, derive it from ladder physics or remove it for 1D topologies where degree ∈ {0,1,2} << k0=4.

**Long-term (theory)**:
5. **Self-consistent frustration** — Instead of 3.0× multiplier, iterate until Λ converges (expensive but principled).
6. **Document divergences** — Add "Implementation Notes" to preprint explaining these engineering choices.

---

## Future Architecture: Database + Web App

**Current pain points**:
- Individual Python runs generate scattered output folders
- Manual eyeballing of PNGs/CSVs to compare results
- No systematic tracking of which parameters were tested
- Difficult to spot trends across parameter sweeps

**Proposed solution**: Replace file-based outputs with a database-backed web application.

**Benefits**:
- **Queryable results**: "Show me all N=4 runs with α > 0.8 and residual < 0.1"
- **Real-time monitoring**: Watch λ scans progress live
- **Automatic comparison**: Diff phase diagrams across bond_cutoff values
- **Reproducibility**: Exact parameter configs stored with each run
- **Collaboration**: Multiple users can queue/view experiments

**Tech stack options**:
- **SQLite/PostgreSQL** + **FastAPI** + **React/Vue frontend**
- **DuckDB** (embedded analytics) + **Streamlit** (quick prototype)
- **MongoDB** (flexible schema for varying topologies) + **FastAPI**

**Database schema sketch**:
```sql
runs (id, timestamp, N, alpha, bond_cutoff, topology, status, ...)
lambda_points (id, run_id, lambda, residual, omega_out, omega_in, ...)
critical_points (id, run_id, lambda_c1, lambda_revival, lambda_c2, ...)
validations (id, run_id, v_lr, no_signalling_violation, energy_drift, ...)
```

**Next steps**: Design schema, choose stack, migrate existing CSV data, build minimal UI for queuing runs and viewing results.

## Environment Notes
- Python venv in `venv/` (activate before running).
- macOS uses `forkserver`; `app.py` enforces it automatically.
- `.gitignore` excludes TeX build artifacts; rerun `pdflatex` as needed.
- Be mindful of directories with spaces (e.g., `"legacy outputs-figures"`).
