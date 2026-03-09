# Decision Log: No-Retuning Holdout (Topology Transfer)

Date: 2026-02-23  
Scope: High-impact transfer test after hotspot and robustness sweeps.

## Decision objective
Test whether a single locked parameterization can reproduce phase-structure behavior across out-of-sample topologies without retuning.

## Locked parameterization
- `deltaB = 6.5`
- `kappa = 0.2`
- `hotspot_multiplier = 1.5`
- `k0 = 4`
- `bond_cutoff = 4`

## Holdouts executed
- `N4_cycle_alpha0.8` (targeted)
- `N4_star_alpha0.8` (targeted)
- `N4_path_alpha1.0` (untargeted transfer)

## Protocol implementation changes
- Updated `scripts/no_retuning_holdout_test.py`:
  - Added adaptive structural refinement (`--max-refinements`, `--refine-factor`, `--refine-max-points`, `--no-refine`).
  - Added explicit metadata fields:
    - `points_initial`, `points_final`, `refinement_steps`
    - `resolution_ok`, `insufficient_resolution`, `no_violation_detected`
  - Added per-attempt scan tags so refinement attempts are preserved as distinct scan CSVs.

## Execution command
`QATNU_FORCE_MP=1 .venv/bin/python scripts/no_retuning_holdout_test.py --scenarios N4_cycle_alpha0.8,N4_star_alpha0.8,N4_path_alpha1.0 --max-points 6 --max-refinements 1 --refine-factor 2.0 --refine-max-points 12 --quiet-scanner`

## Results
- `OVERALL=FAIL`
- `N4_path_alpha1.0`: PASS (`lambda_c1=0.38`, `lambda_revival=0.94`, `lambda_c2=1.22`, `residual_min=0.112`)
- `N4_star_alpha0.8`: FAIL (residual gate fails and target mismatch)
- `N4_cycle_alpha0.8`: FAIL with `no_violation_detected=True`
  - residual remains at machine floor (`~1e-16`)
  - no 10% violation crossing, so revival analysis never triggers

## Interpretation
This lock does not transfer across topologies.  
The dominant failure mode is no longer only detector ambiguity; the model behavior itself changes qualitatively under topology transfer.

## Artifact index
- `outputs/no_retuning_holdout_20260223-092900/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260223-092900/preregistered_protocol.json`
- `outputs/no_retuning_holdout_20260223-092900/holdout_report.md`
