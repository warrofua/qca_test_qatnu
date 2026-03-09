# Decision Log: Robustness Sweep Over deltaB, kappa, bond_cutoff

Date: 2026-02-23
Scope: Determine whether phase landmarks remain stable once hotspot reporting is locked to global post-violation revival.

## Mnemolog-style decision protocol
Bootstrap references used as decision contract:
1. `GET https://mnemolog.com/robots.txt`
2. `GET https://mnemolog.com/.well-known/agent.json`
3. `GET https://mnemolog.com/agents.txt`
4. `GET https://mnemolog.com/api/agents/capabilities`

Execution pattern: bounded experiment matrix, explicit baseline, reproducible artifacts.

## Objective
Test whether hotspot sensitivity is robust physics or parameter-tuning artifact by varying:
- `deltaB` (bond spacing),
- `kappa` (degree penalty),
- `bond_cutoff` (chi cutoff),
while using the locked official revival rule (`lambda_revival = global post-violation minimum`).

## Implementation updates
- `scanners.py` now accepts parameterized `deltaB`, `kappa`, and `k0` in scan routines.
- Added reproducible matrix runner:
  - `scripts/robustness_sweep.py`
  - outputs summary CSV, delta-vs-baseline CSV, span CSV, heatmaps, markdown report.

## Runtime note
- Initial matrix with `bond_cutoff={3,4,6}` was started but `chi=6` proved too slow for a same-session completion.
- Run was intentionally restarted with `bond_cutoff={3,4,5}` to keep the matrix controlled but tractable.

## Final matrix executed
- Run tag: `robustness_sweep_20260222-201426_N4_alpha0.80_pts24`
- Configuration:
  - `N=4`, `alpha=0.8`, `points=24`, `hotspot_multiplier=3.0`
  - `deltaB ∈ {3,5,7}`
  - `kappa ∈ {0.05,0.1,0.2}`
  - `bond_cutoff ∈ {3,4,5}`
  - total combinations: 27

## Key outcomes
- Global spans:
  - span(`lambda_c1`) = `0.625`
  - span(`lambda_revival`) = `1.400`
  - span(`lambda_c2`) = `1.34375`
  - span(`mean_residual`) = `0.22304`
- Mean-of-means sensitivity:
  - `deltaB` strongly shifts `lambda_revival` and `lambda_c1`.
  - `kappa` strongly shifts `lambda_c1` and mean residual.
  - `bond_cutoff` has comparatively small impact on `lambda_c1` and mean residual in this coarse sweep.

Conclusion:
- Phase landmarks are not robust under simultaneous `deltaB`/`kappa` variation in this tested window.
- The dominant uncertainty source has moved from detector ambiguity to Hamiltonian-parameter sensitivity.

## Artifact index
- Report: `outputs/robustness_sweep_20260222-201426_N4_alpha0.80_pts24/robustness_report.md`
- Summary CSV: `outputs/robustness_sweep_20260222-201426_N4_alpha0.80_pts24/robustness_summary.csv`
- Deltas CSV: `outputs/robustness_sweep_20260222-201426_N4_alpha0.80_pts24/robustness_summary_with_deltas.csv`
- Spans CSV: `outputs/robustness_sweep_20260222-201426_N4_alpha0.80_pts24/robustness_spans.csv`
