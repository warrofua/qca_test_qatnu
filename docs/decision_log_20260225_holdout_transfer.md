# Decision Log: Locked-Parameter Topology Transfer Holdout (2026-02-25)

## Objective
Run a no-retuning transfer check with one locked parameterization informed by the Feb 25 sensitivity results, then determine whether topology transfer passes without additional tuning.

## Locked parameterization
- `deltaB = 5.0`
- `kappa = 0.1`
- `hotspot_multiplier = 3.0`
- `k0 = 4`
- `bond_cutoff = 4`

Rationale:
- Central/default point in the tested sensitivity envelope.
- Avoids selecting edge values that only maximize one topology.

## Runs

### A) Quick all-topology screen (`8` points, no refine)
Command:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/no_retuning_holdout_test.py \
  --scenarios N4_cycle_alpha0.8,N4_star_alpha0.8,N4_path_alpha1.0 \
  --deltaB 5.0 \
  --kappa 0.1 \
  --hotspot-multiplier 3.0 \
  --k0 4 \
  --bond-cutoff 4 \
  --max-points 8 \
  --no-refine \
  --quiet-scanner \
  --output-dir outputs/no_retuning_holdout_20260225_locked_d5_k01_h3_quick8
```

Artifacts:
- `outputs/no_retuning_holdout_20260225_locked_d5_k01_h3_quick8/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260225_locked_d5_k01_h3_quick8/holdout_report.md`

Readout:
- `N4_cycle_alpha0.8`: `no_violation_detected=True` (no triplet).
- `N4_path_alpha1.0`: `lambda_c1=lambda_revival=0.3` (coarse-grid alias).
- `N4_star_alpha0.8`: alias + target mismatch.
- Overall: **FAIL**.

### B) Path+star disambiguation (`16` points, no refine)
Command:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/no_retuning_holdout_test.py \
  --scenarios N4_star_alpha0.8,N4_path_alpha1.0 \
  --deltaB 5.0 \
  --kappa 0.1 \
  --hotspot-multiplier 3.0 \
  --k0 4 \
  --bond-cutoff 4 \
  --max-points 16 \
  --no-refine \
  --quiet-scanner \
  --output-dir outputs/no_retuning_holdout_20260225_locked_d5_k01_h3_pathstar16
```

Artifacts:
- `outputs/no_retuning_holdout_20260225_locked_d5_k01_h3_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260225_locked_d5_k01_h3_pathstar16/holdout_report.md`

Readout:
- `N4_path_alpha1.0`: **PASS**
  - `lambda_c1=0.19`, `lambda_revival=0.37`, `lambda_c2=0.657`
  - `residual_min=0.136`
- `N4_star_alpha0.8`: **FAIL (target mismatch)**
  - `lambda_c1=0.19`, `lambda_revival=0.28`, `lambda_c2=0.37`
  - `residual_min=0.160`
  - errors: `|Δc1|=0.026`, `|Δrev|=1.02`, `|Δc2|=1.03`
- Overall: **FAIL**.

## Decision
Single-lock transfer remains non-robust across topologies under the current phenomenological structure:
- path can pass structural criteria,
- star misses target landmarks by large margins,
- cycle can sit in no-violation regime where revival logic does not engage.

Therefore:
- do not claim no-retuning topology transfer as validated;
- prioritize topology-conditioned mechanisms (or structural model changes) over further scalar-parameter tuning.
