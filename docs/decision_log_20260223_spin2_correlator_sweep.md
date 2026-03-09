# Decision Log: Spin-2 Correlator Sweep (Topology + Hotspot Sensitivity)

Date: 2026-02-23  
Scope: Replace chi-profile toy PSD readout with bond-correlator proxy and test where slopes are closest to `-2`.

## Objective
Evaluate whether the correlator-based spin-2 proxy reveals topology-dependent movement toward the target long-wavelength slope (`-2`) under the currently locked Hamiltonian family.

## Implementation updates
- Added new sweep runner:
  - `scripts/spin2_correlator_topology_sweep.py`
- Added correlator-based spin-2 extraction in:
  - `geometry.py` (`spin2_from_bond_correlators`)
- Wired simulator/backend reporting of spin-2 proxy:
  - `Simulator/backend/main.py`
  - `Simulator/frontend/app.py`

## Sweep A (N4, hotspot=1.5)
Command:
` .venv/bin/python scripts/spin2_correlator_topology_sweep.py --workers 6`

Artifacts:
- `outputs/spin2_correlator_sweep_20260223-100252/spin2_points.csv`
- `outputs/spin2_correlator_sweep_20260223-100252/spin2_summary.csv`
- `outputs/spin2_correlator_sweep_20260223-100252/spin2_slope_vs_lambda.png`
- `outputs/spin2_correlator_sweep_20260223-100252/spin2_report.md`

Best-by-scenario:
- `N4_star_alpha0.8`: best slope `-2.258`, best error `0.258`
- `N4_path_alpha0.8`: best slope `-3.615`, best error `1.615`
- `N4_cycle_alpha0.8`: best slope `-3.685`, best error `1.685`

## Sweep B (N4, hotspot=3.0)
Command:
` .venv/bin/python scripts/spin2_correlator_topology_sweep.py --workers 6 --hotspot-multiplier 3.0`

Artifacts:
- `outputs/spin2_correlator_sweep_20260223-101138/spin2_points.csv`
- `outputs/spin2_correlator_sweep_20260223-101138/spin2_summary.csv`
- `outputs/spin2_correlator_sweep_20260223-101138/spin2_slope_vs_lambda.png`
- `outputs/spin2_correlator_sweep_20260223-101138/spin2_report.md`

Best-by-scenario:
- `N4_star_alpha0.8`: best slope `-2.260`, best error `0.260`
- `N4_cycle_alpha0.8`: best slope `-3.580`, best error `1.580`
- `N4_path_alpha0.8`: best slope `-3.772`, best error `1.772`

## Sweep C (N5 path sparse anchor, hotspot=1.5)
Command:
` .venv/bin/python scripts/spin2_correlator_topology_sweep.py --scenarios N5:path:0.8 --lambdas 0.2,0.6,1.0 --workers 3`

Artifacts:
- `outputs/spin2_correlator_sweep_20260223-100445/spin2_points.csv`
- `outputs/spin2_correlator_sweep_20260223-100445/spin2_summary.csv`
- `outputs/spin2_correlator_sweep_20260223-100445/spin2_slope_vs_lambda.png`
- `outputs/spin2_correlator_sweep_20260223-100445/spin2_report.md`

Result:
- `N5_path_alpha0.8`: best slope `-3.118`, best error `1.118`

## Sweep D (N5 dense topologies, hotspot=1.5, bond_cutoff=3)
Command:
` .venv/bin/python scripts/spin2_correlator_topology_sweep.py --scenarios N5:path:0.8,N5:cycle:0.8,N5:star:0.8 --lambdas 0.2,0.4,0.6,0.8,1.0,1.2 --bond-cutoff 3 --workers 3`

Artifacts:
- `outputs/spin2_correlator_sweep_20260223-101802/spin2_points.csv`
- `outputs/spin2_correlator_sweep_20260223-101802/spin2_summary.csv`
- `outputs/spin2_correlator_sweep_20260223-101802/spin2_slope_vs_lambda.png`
- `outputs/spin2_correlator_sweep_20260223-101802/spin2_report.md`

Best-by-scenario:
- `N5_star_alpha0.8`: best slope `-2.407`, best error `0.407`
- `N5_path_alpha0.8`: best slope `-3.118`, best error `1.118`
- `N5_cycle_alpha0.8`: best slope `-3.755`, best error `1.755`

## Sweep E (Cutoff robustness, N4)
Goal:
Test whether topology-dependent spin-2 slopes are stable under bond cutoff changes.

### E1: N4 all topologies, chi={3,4}
Command:
` .venv/bin/python scripts/spin2_correlator_topology_sweep.py --bond-cutoffs 3,4 --lambdas 0.2,0.4,0.6,0.8,1.0 --workers 6 --output-dir outputs/spin2_correlator_sweep_20260223-robust_N4_chi34`

Artifacts:
- `outputs/spin2_correlator_sweep_20260223-robust_N4_chi34/spin2_points.csv`
- `outputs/spin2_correlator_sweep_20260223-robust_N4_chi34/spin2_summary.csv`
- `outputs/spin2_correlator_sweep_20260223-robust_N4_chi34/spin2_slope_vs_lambda.png`
- `outputs/spin2_correlator_sweep_20260223-robust_N4_chi34/spin2_report.md`

Best-by-scenario:
- `N4_star_alpha0.8, chi=3`: best slope `-2.523`, best error `0.523`
- `N4_star_alpha0.8, chi=4`: best slope `-2.326`, best error `0.326`
- `N4_path_alpha0.8, chi=3`: best slope `-3.415`, best error `1.415`
- `N4_path_alpha0.8, chi=4`: best slope `-3.718`, best error `1.718`
- `N4_cycle_alpha0.8, chi=3`: best slope `-3.449`, best error `1.449`
- `N4_cycle_alpha0.8, chi=4`: best slope `-3.685`, best error `1.685`

### E2: N4 path/star extension, chi=5
Command:
` .venv/bin/python scripts/spin2_correlator_topology_sweep.py --scenarios N4:path:0.8,N4:star:0.8 --bond-cutoffs 5 --lambdas 0.2,0.4,0.6,0.8,1.0 --workers 6 --output-dir outputs/spin2_correlator_sweep_20260223-robust_N4_pathstar_chi5`

Artifacts:
- `outputs/spin2_correlator_sweep_20260223-robust_N4_pathstar_chi5/spin2_points.csv`
- `outputs/spin2_correlator_sweep_20260223-robust_N4_pathstar_chi5/spin2_summary.csv`
- `outputs/spin2_correlator_sweep_20260223-robust_N4_pathstar_chi5/spin2_slope_vs_lambda.png`
- `outputs/spin2_correlator_sweep_20260223-robust_N4_pathstar_chi5/spin2_report.md`

Best-by-scenario:
- `N4_star_alpha0.8, chi=5`: best slope `-2.376`, best error `0.376`
- `N4_path_alpha0.8, chi=5`: best slope `-3.805`, best error `1.805`

## Sweep F (Cutoff robustness, N5 path/star)
Goal:
Check whether N5 star advantage persists when increasing cutoff from chi=3 to chi=4.

Command:
` .venv/bin/python scripts/spin2_correlator_topology_sweep.py --scenarios N5:path:0.8,N5:star:0.8 --bond-cutoffs 3,4 --lambdas 0.2,0.6,1.0 --workers 2 --output-dir outputs/spin2_correlator_sweep_20260223-robust_N5_pathstar_chi34`

Artifacts:
- `outputs/spin2_correlator_sweep_20260223-robust_N5_pathstar_chi34/spin2_points.csv`
- `outputs/spin2_correlator_sweep_20260223-robust_N5_pathstar_chi34/spin2_summary.csv`
- `outputs/spin2_correlator_sweep_20260223-robust_N5_pathstar_chi34/spin2_slope_vs_lambda.png`
- `outputs/spin2_correlator_sweep_20260223-robust_N5_pathstar_chi34/spin2_report.md`

Best-by-scenario:
- `N5_star_alpha0.8, chi=3`: best slope `-2.536`, best error `0.536`
- `N5_star_alpha0.8, chi=4`: best slope `-2.754`, best error `0.754`
- `N5_path_alpha0.8, chi=3`: best slope `-3.118`, best error `1.118`
- `N5_path_alpha0.8, chi=4`: best slope `-3.118`, best error `1.118`

## Sweep G (Spin-2 semantic lock + N5 chi=5 feasibility guard)
Goal:
Unify spin-2 metric semantics end-to-end, then preflight `N5 chi=5` dense runs safely.

Implementation:
- Normalized simulator spin-2 semantics to **power convention**:
  - `S(k) ~ 1/k^p`, target `p=2`, slope derived as `-p`.
- Updated simulator outputs to expose both power and slope consistently:
  - `Simulator/backend/main.py` (`get_run`, `get_run_agreement`, `spin2_complete` payload)
  - `Simulator/backend/models.py` (validation defaults/docs)
  - `Simulator/frontend/app.py` (status/log display)
- Hardened sweep runner:
  - `scripts/spin2_correlator_topology_sweep.py`
  - Added `--max-dense-gib` preflight skip for infeasible dense Hamiltonians.
  - Added explicit `spin2_skipped.csv` / `spin2_errors.csv` outputs.
  - Added per-future exception isolation.

Feasibility table:
- `outputs/spin2_n5_chi5_feasibility_20260223.csv`
  - `N5 path chi=5`: `total_dim=20000`, dense estimate `2.98 GiB`
  - `N5 star chi=5`: `total_dim=20000`, dense estimate `2.98 GiB`
  - `N5 cycle chi=5`: `total_dim=100000`, dense estimate `74.51 GiB`

Command (explicit skip artifact):
` .venv/bin/python scripts/spin2_correlator_topology_sweep.py --scenarios N5:cycle:0.8 --bond-cutoffs 5 --lambdas 0.6 --workers 1 --max-dense-gib 32 --output-dir outputs/spin2_correlator_sweep_20260223-N5_cycle_chi5_skip`

Artifacts:
- `outputs/spin2_correlator_sweep_20260223-N5_cycle_chi5_skip/spin2_summary.csv`
- `outputs/spin2_correlator_sweep_20260223-N5_cycle_chi5_skip/spin2_skipped.csv`
- `outputs/spin2_correlator_sweep_20260223-N5_cycle_chi5_skip/spin2_report.md`

Verification addendum:
- Semantic smoke run passed:
  - `outputs/spin2_correlator_sweep_20260223-verify_semantics/spin2_summary.csv`
- Preflight skip behavior re-validated:
  - `outputs/spin2_correlator_sweep_20260223-verify_skip/spin2_summary.csv`
  - `outputs/spin2_correlator_sweep_20260223-verify_skip/spin2_skipped.csv`
- Runtime probe (`N5 path/star`, `chi=5`, one λ each) did not reach first completed point within short wall-clock window on current hardware; run was terminated intentionally to avoid long blocking:
  - `outputs/spin2_correlator_sweep_20260223-N5_pathstar_chi5_probe/`

## Consolidated table
- `outputs/spin2_correlator_comparison_20260223.csv`
- `outputs/spin2_correlator_cutoff_sensitivity_20260223.csv`

## Interpretation
1. Topology is currently the dominant discriminator for the new spin-2 proxy.
2. Star topology repeatedly lands near `-2.26` (close to target), while path/cycle remain around `-3.6` to `-4.0`.
3. N5 path improves relative to N4 path (`-3.12` vs `~ -3.6 to -3.8`) but is still far from `-2`.
4. N5 star remains much closer to target (`-2.41`) than N5 path/cycle, so the topology signal persists at larger N.
5. The star advantage is robust to hotspot multiplier change from `1.5` to `3.0` (tested at N4).
6. Cutoff robustness (new):
   - N4 star stays in the near-target band across chi `{3,4,5}` (`-2.52`, `-2.33`, `-2.38`).
   - N4/N5 path remains steep and non-spin-2-like as cutoff increases.
   - N5 path is nearly cutoff-invariant (`-3.118` at chi `3` and `4` for tested lambdas).
   - N5 star degrades at chi `4` but still remains clearly separated from path.
7. Semantic lock (new): simulator now reports spin-2 in one convention (`power` target `2.0`) with derived slope aliases, removing previous sign mismatch between websocket and DB/API paths.
8. Feasibility boundary (new): `N5 cycle chi=5` dense diagonalization is currently out-of-budget (`~74.5 GiB` matrix estimate), so decisive `chi=5` topology-wide checks require sparse/iterative eigensolvers or stronger hardware.

## Immediate next experiment
Run dense `N5 path/star` at `chi=5` on upgraded hardware with 6-9 λ points, and in parallel implement a sparse/iterative eigensolver path for `N5 cycle chi=5` so topology-complete `chi=5` comparison becomes tractable.
