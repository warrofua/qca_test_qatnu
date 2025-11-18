# Agent Playbook – QATNU/SRQID Repo

## Mission
Model and validate the QATNU/SRQID physics claims: a measurement-free promotion mechanism that slows local clocks according to Postulate 1,
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
- `Λ` extraction in `core_qca.py`: Postulate 1 residuals rely on it.
- `phase_analysis.py`: replicates the preprint’s six-panel figure; keep critical point logic intact.
- `srqid.py`: must continue running inside the exact Hamiltonian to maintain SRQID claims.
- Legacy outputs in `legacy outputs-figures/`: reference data for comparison; don’t overwrite.
- `dataverse_files/`: contains toy scripts used in the papers.

## Open Tasks / TODO
- Improve the χ→PSD mapping to recover the 1/k² slope or juxtapose toy/data-driven PSDs in the docs.
- Perform α sweeps at N=4 (e.g., α=0.6, 1.0) to map phase boundaries and test Postulate 1 rigidity.
- Implement the linear-response/Stinespring derivation to compute α directly from the promotion isometry (remove manual fitting).
- Add support for non-1D graphs (star, cycle/pentagon, “house”) starting with an N=5 star graph and N=4 cycle: run ~40-point λ scans and log their critical points.

## Environment Notes
- Python venv in `venv/` (activate before running).
- macOS uses `forkserver`; `app.py` enforces it automatically.
- `.gitignore` excludes TeX build artifacts; rerun `pdflatex` as needed.
- Be mindful of directories with spaces (e.g., `"legacy outputs-figures"`).
